"""Pure deterministic Tier-1 signal functions (no capability calls): empty-segment detection, bidirectional boundary punctuation/capitalization heuristics, forced-alignment coverage flags, positional cross-transcriber diff, phonetic + edit-distance variant clustering, and the event-proposal overlay (leg 4: the finetuned detector's spans anchored onto the spine). The worklist is recomputed from these each session; revolution-1 builds ZERO new capabilities."""

import json
import re
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cjm_transcript_correction_core.models import SpineSegment


def detect_empty_segments(
    segments: List[SpineSegment],  # Ordered spine segments
) -> List[int]:  # Positions (in `segments`) of empty-text segments
    """Find empty-text segments (silence VAD chunks with no aligned words; decomp D14)."""
    return [i for i, s in enumerate(segments) if s.is_empty]


_TERMINAL_PUNCT = (".", "!", "?", "。", "！", "？")  # incl. CJK full-stop / ! / ?


def _ends_terminal(text: str) -> bool:  # True if text ends with sentence-terminal punctuation
    """Whether a segment's text ends with terminal punctuation (trailing quotes/brackets ignored)."""
    t = (text or "").rstrip().rstrip("\"')”’")
    return t.endswith(_TERMINAL_PUNCT)


def _starts_upper(text: str) -> bool:  # True if the first alphabetic char is uppercase
    """Whether a segment's text starts with an uppercase letter (leading quotes/brackets ignored)."""
    for ch in (text or "").lstrip("\"'(“‘"):
        if ch.isalpha():
            return ch.isupper()
        if not ch.isspace():
            return False
    return False


def boundary_punct_caps_flags(
    segments: List[SpineSegment],  # Ordered spine segments
) -> Dict[int, List[str]]:  # segment index -> boundary flags
    """Bidirectional boundary punctuation/capitalization heuristics (in-segment only).

    At each border (seg[i] -> seg[i+1]) flag the two error directions a downstream
    grouping workflow cares about, WITHOUT ever merging across audio segments:
      - "boundary-missing-terminal": seg[i] lacks terminal punctuation and seg[i+1]
        starts uppercase -> a sentence may end here but is missing a period.
      - "boundary-terminal-then-lowercase": seg[i] ends terminal but seg[i+1] starts
        lowercase -> one sentence may have been split across the border.
    Empty neighbours are skipped (handled by the prune).
    """
    flags: Dict[int, List[str]] = {}
    for i in range(len(segments) - 1):
        a, b = segments[i], segments[i + 1]
        if a.is_empty or b.is_empty:
            continue
        bt = b.text.strip()
        if not _ends_terminal(a.text) and _starts_upper(b.text):
            flags.setdefault(i, []).append("boundary-missing-terminal")
        if _ends_terminal(a.text) and bt[:1].isalpha() and not _starts_upper(b.text):
            flags.setdefault(i, []).append("boundary-terminal-then-lowercase")
    return flags


def fa_coverage_flags(
    segments: List[SpineSegment],  # Ordered spine segments
) -> Dict[int, List[str]]:  # segment index -> coverage flags
    """Flag segments whose forced-alignment coverage looks suspect (Tier-1).

    Empty-text segments (no aligned words) and segments missing source-coordinate
    timing are flagged; both are alignment-failure signals shared by text and
    segmentation errors.
    """
    flags: Dict[int, List[str]] = {}
    for i, s in enumerate(segments):
        if s.is_empty:
            flags.setdefault(i, []).append("empty-text")
        if s.start_time is None or s.end_time is None:
            flags.setdefault(i, []).append("missing-timing")
    return flags


def levenshtein(
    a: str,  # First string
    b: str,  # Second string
) -> int:  # Edit distance
    """Levenshtein edit distance (pure, in-core; variant-clustering primitive)."""
    a, b = a or "", b or ""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # alphabetic word tokens (unicode-aware)


def phonetic_key(
    word: str,  # A single word token
) -> str:  # A coarse phonetic key (Soundex-like, in-core)
    """Compute a coarse phonetic key for a word (groups like-sounding variants).

    A lightweight Soundex-style reduction (first letter + consonant codes, vowels
    dropped): enough to bucket transcription variants of one entity for
    fix-one-fix-all, without a phonetics dependency.
    """
    w = "".join(ch for ch in (word or "").lower() if ch.isalpha())
    if not w:
        return ""
    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
             **dict.fromkeys("dt", "3"), **dict.fromkeys("l", "4"),
             **dict.fromkeys("mn", "5"), **dict.fromkeys("r", "6")}
    first = w[0]
    tail: List[str] = []
    prev = codes.get(first, "")
    for ch in w[1:]:
        c = codes.get(ch, "")
        if c and c != prev:
            tail.append(c)
        prev = c
    return (first + "".join(tail) + "000")[:4]


def _normalize_text(text: str) -> str:  # Lowercased alphabetic word tokens, space-joined
    """Normalize segment text for cross-transcriber comparison."""
    return " ".join(_WORD_RE.findall((text or "").lower()))


def variant_divergence(
    segments: List[SpineSegment],            # Layer-0 spine (authoritative text)
    variants: Dict[str, Dict[str, str]],     # segment_id -> {transcriber: chunk text} (from the graph)
) -> Dict[int, Tuple[str, str]]:  # spine index -> (authoritative_text, first divergent variant)
    """Within-segment cross-transcriber divergence (stage 5: intra-graph).

    The shared-skeleton model stores every transcriber's chunk text as a slice
    on ONE segment, so divergence is a WITHIN-NODE comparison now (C14 realized)
    — no second spine, no positional join. Proper-noun / error sites concentrate
    where the normalized texts diverge (the force-multiplier signal); the
    authoritative transcriber's own variant compares equal by construction.
    """
    diffs: Dict[int, Tuple[str, str]] = {}
    for i, s in enumerate(segments):
        auth_norm = _normalize_text(s.text)
        for t, vtext in (variants.get(s.id) or {}).items():
            if _normalize_text(vtext) != auth_norm:
                diffs[i] = (s.text or "", vtext)
                break
    return diffs


def cluster_variants(
    words: List[str],    # Candidate word tokens (e.g. divergent proper nouns)
    max_edits: int = 2,  # Max edit distance to join two words into one cluster
) -> List[List[str]]:  # Clusters (size > 1) of like-sounding / near-spelled variants
    """Cluster word variants by phonetic key + edit distance (fix-one-fix-all).

    Buckets transcription variants of one entity so a single decision can map them
    all to a canonical form. Pure, in-core (no phonetics dependency).
    """
    uniq = list(dict.fromkeys(w.strip() for w in words if w and w.strip()))
    clusters: List[List[str]] = []
    keys: List[str] = []
    for w in uniq:
        k = phonetic_key(w)
        placed = False
        for ci, ck in enumerate(keys):
            if k and k == ck and levenshtein(w.lower(), clusters[ci][0].lower()) <= max_edits:
                clusters[ci].append(w)
                placed = True
                break
        if not placed:
            clusters.append([w])
            keys.append(k)
    return [c for c in clusters if len(c) > 1]


def compute_signal_flags(
    segments: List[SpineSegment],                       # Ordered layer-0 spine
    variants: Optional[Dict[str, Dict[str, str]]] = None,  # segment_id -> {transcriber: text} (intra-graph)
) -> Dict[int, List[str]]:  # segment index -> combined Tier-1 flags
    """Combine all deterministic Tier-1 signals into per-segment flags.

    The worklist is RECOMPUTED from this each session (only decisions persist);
    new signals join here and are picked up automatically. Stage 5: the
    transcriber-divergence signal reads the segments' own variant slices
    (intra-graph), not a second decomp spine.
    """
    flags: Dict[int, List[str]] = {}

    def add(idx: int, fl: List[str]) -> None:
        bucket = flags.setdefault(idx, [])
        for f in fl:
            if f not in bucket:
                bucket.append(f)

    for idx, fl in fa_coverage_flags(segments).items():
        add(idx, fl)
    for idx, fl in boundary_punct_caps_flags(segments).items():
        add(idx, fl)
    if variants:
        for idx in variant_divergence(segments, variants):
            add(idx, ["transcriber-divergence"])
    return flags


def speaker_turn_proposals(
    segments: List[SpineSegment],  # Ordered spine segments (source-coordinate times)
    turns: List[Dict[str, Any]],   # Diarization turns [{start, end, speaker, ...}], source coordinates
) -> Dict[str, Dict[str, Any]]:  # segment id -> {"cluster", "overlap", "coverage"}
    """Dominant diarization cluster per segment — the assign lane's proposal paint.

    Pure time-overlap dominance: accumulate overlap seconds per anonymous
    cluster label across the (possibly overlapping) turns; the label with the
    most overlap wins. `coverage` = dominant overlap / segment duration — what
    the painter dims on and the accept op snapshots. Segments with no time
    span, no overlapping turn, or NO TEXT get NO proposal (the lane shows ∅):
    text is the unit of attribution supervision, so empty chunks — silence,
    inhale/bookend inserts — never propose (and never ride a bulk accept);
    a text-bearing synthetic (a split half, an e-typed missed-speech insert)
    proposes like any chunk (drive ask 2026-07-27). Cluster labels are
    result-scoped (never identities) — binding them to Entities is the accept
    gesture's job (DEC 8a4df244 cluster-name-once)."""
    out: Dict[str, Dict[str, Any]] = {}
    ts = sorted((float(t.get("start") or 0.0), float(t.get("end") or 0.0),
                 str(t.get("speaker") or "")) for t in (turns or []))
    if not ts:
        return out
    lo = 0
    for seg in segments:
        if seg.start_time is None or seg.end_time is None:
            continue
        if not (seg.text or "").strip():
            continue
        s, e = float(seg.start_time), float(seg.end_time)
        if e <= s:
            continue
        # turns sorted by start: one ending at/before this segment's start can
        # never overlap a LATER segment either — safe to retire it.
        while lo < len(ts) and ts[lo][1] <= s:
            lo += 1
        overlap: Dict[str, float] = {}
        j = lo
        while j < len(ts) and ts[j][0] < e:
            t_s, t_e, label = ts[j]
            dur = min(e, t_e) - max(s, t_s)
            if dur > 0 and label:
                overlap[label] = overlap.get(label, 0.0) + dur
            j += 1
        if not overlap:
            continue
        cluster, dom = max(overlap.items(), key=lambda kv: kv[1])
        out[seg.id] = {"cluster": cluster, "overlap": round(dom, 3),
                       "coverage": round(min(1.0, dom / (e - s)), 3)}
    return out


# Format tag of a consumed proposal set (leg 4, DEC 8e05b87b): the manifest
# chain's inference-run record — capability-owned today, read by the workflow
# BY FORMAT TAG (generalized to a workflow-generic seam at n=2 proposal
# producers, the 16159e09 rule).
EVENT_PROPOSAL_SET_FORMAT = "cjm-capability-pyannote/proposal-set-manifest"


def load_event_proposal_set(
    ws_root: str,                         # Workspace root (proposal sets live under <root>/proposals/)
    content_hash: Optional[str] = None,   # Source content hash to match (preferred join key)
    source_id: Optional[str] = None,      # Source node id to match (fallback join key)
) -> Optional[Dict[str, Any]]:  # {"manifest": ..., "proposals": [...]} for the LATEST match, or None
    """Find the latest proposal set for a source (the turns-artifact discovery
    pattern: workspace + source identity name the artifact; no artifact = no
    proposals and the walk stays manual). Malformed sets are skipped, never
    fatal — a broken artifact must not take down the TUI open."""
    root = Path(ws_root) / "proposals"
    if not root.is_dir():
        return None
    best: Optional[Dict[str, Any]] = None
    for mp in sorted(root.glob("*/manifest.json")):
        try:
            m = json.loads(mp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("format") != EVENT_PROPOSAL_SET_FORMAT:
            continue
        src = m.get("source") or {}
        if content_hash and src.get("content_hash") == content_hash:
            pass
        elif source_id and src.get("source_id") == source_id:
            pass
        else:
            continue
        if best is None or float(m.get("created_at") or 0) > float(best["manifest"].get("created_at") or 0):
            best = {"manifest": m, "path": str(mp)}
    if best is None:
        return None
    data_file = Path(best["path"]).parent / str((best["manifest"].get("files") or {}).get("proposals") or "proposals.jsonl")
    try:
        proposals = [json.loads(line) for line in data_file.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    return {"manifest": best["manifest"], "proposals": proposals}


def event_span_proposals(
    segments: List[SpineSegment],        # Ordered spine segments (source-coordinate times)
    proposals: List[Dict[str, Any]],     # Proposal spans [{proposal_id,label,start_time,end_time,score}]
    occupied: Optional[List[Tuple[float, float]]] = None,  # Active insert spans (already-materialized time ranges)
) -> Dict[str, List[Dict[str, Any]]]:  # anchor segment id -> pending proposals (time order)
    """Anchor pending event proposals onto the spine — the propose lane's paint.

    Each proposal anchors to the segment it would be inserted AFTER: the last
    segment whose start_time <= the proposal's start (the chunk-insert
    after-anchor convention, DEC 3d3fa2a8). Proposals overlapping an ALREADY
    MATERIALIZED insert span are dropped — they were accepted in some session;
    the verdict join (not this paint) is where accept/edit/reject derive
    (DEC 8e05b87b). Proposals starting before the first timed segment anchor
    to it."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    timed = [(float(s.start_time), s.id) for s in segments if s.start_time is not None]
    if not timed:
        return out
    starts = [t for t, _ in timed]
    occ = sorted(occupied or [])
    occ_starts = [s for s, _ in occ]
    for p in sorted(proposals or [], key=lambda d: float(d.get("start_time") or 0.0)):
        ps, pe = float(p.get("start_time") or 0.0), float(p.get("end_time") or 0.0)
        if pe <= ps:
            continue
        j = bisect_right(occ_starts, pe) - 1
        if j >= 0 and occ[j][1] > ps:  # overlaps a materialized insert — already decided
            continue
        i = max(0, bisect_right(starts, ps) - 1)
        out.setdefault(timed[i][1], []).append(p)
    return out
