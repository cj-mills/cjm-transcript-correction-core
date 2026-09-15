"""Pure deterministic Tier-1 signal functions (no capability calls): empty-segment detection, bidirectional boundary punctuation/capitalization heuristics, forced-alignment coverage flags, positional cross-transcriber diff, phonetic + edit-distance variant clustering, and the event-proposal overlay (leg 4: the finetuned detector's spans anchored onto the spine). The worklist is recomputed from these each session; revolution-1 builds ZERO new capabilities."""

import difflib
import json
import re
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def attention_fa_marks(
    segments: List[SpineSegment],       # Effective spine (source-coordinate times), text-bearing segments considered
    fa_words: List[Dict[str, Any]],     # Aligned words in SOURCE seconds: [{"s", "e", "text"}], any order
    events: Optional[List[Dict[str, Any]]] = None,  # Active event inserts [{"start", "end", "label"}] (inhale, noise ...)
    thresholds: Optional[Dict[str, float]] = None,  # Overrides of ATTENTION_THRESHOLDS
) -> List[Dict[str, Any]]:  # Mark rows: {"mark_class", "anchor", "t", "rationale", "key"}
    """The forced-alignment signals of the attention tier (item 3758f6cb signal 1), pure.

    Three classes, each a mechanical boundary-shift / nudge / omission candidate the
    walk lane otherwise finds only by listening: `fa-mid-word-boundary` — a boundary
    between two text-bearing segments falls INSIDE an aligned word (past `word_margin_s`
    from its edges; boundary anchor); `fa-trailing-audio` / `fa-leading-audio` — a
    segment's audio runs on for `tail_s` after the last aligned speech in it (or before
    the first) with no event insert explaining the stretch (segment anchor: text the
    transcriber may have dropped). Aligned speech is measured by OVERLAP clipped to the
    segment, so a word straddling a boundary counts on both sides and never fakes a tail."""
    th = {**ATTENTION_THRESHOLDS, **(thresholds or {})}
    words = sorted((w for w in (fa_words or []) if w.get("s") is not None and w.get("e") is not None),
                   key=lambda w: float(w["s"]))
    starts = [float(w["s"]) for w in words]
    evs = sorted(((float(e.get("start") or 0.0), float(e.get("end") or 0.0), str(e.get("label") or ""))
                  for e in (events or [])), key=lambda e: e[0])

    def _covered(a: float, b: float) -> float:  # seconds of [a, b] an event insert covers
        return sum(max(0.0, min(b, ee) - max(a, es)) for es, ee, _ in evs if es < b and ee > a)

    def _word_at(t: float) -> Optional[Dict[str, Any]]:  # the word strictly containing t (past the margin)
        i = bisect_right(starts, t) - 1
        while i >= 0 and float(words[i]["s"]) > t - 5.0:   # words are short: a few seconds back is enough
            w = words[i]
            if float(w["s"]) + th["word_margin_s"] < t < float(w["e"]) - th["word_margin_s"]:
                return w
            i -= 1
        return None

    rows: List[Dict[str, Any]] = []
    timed = [s for s in segments if s.start_time is not None and s.end_time is not None
             and (s.text or "").strip()]
    for a, b in zip(timed, timed[1:]):
        for t, side in ((float(a.end_time), "left"), (float(b.start_time), "right")):
            w = _word_at(t)
            if w is None:
                continue
            ws_, we_ = float(w["s"]), float(w["e"])
            # No stretched-word class here any more: a long aligned word across the boundary
            # was the fa-stretched-word mark of re-tune 1 (b7e9838a) — removed by ruling
            # 328c48ae (every standalone instance dismissed on the second walk, evidence 1dbbc512).
            rows.append({"mark_class": "fa-mid-word-boundary",
                         "anchor": {"kind": "boundary", "boundary_after": a.id, "right_segment_id": b.id},
                         "t": t,
                         "rationale": f"boundary at {t:.2f}s ({side} edge) falls inside aligned word "
                                      f"'{w.get('text', '')}' {ws_:.2f}-{we_:.2f}s",
                         "key": f"fa-mid-word-boundary@{a.id}|{b.id}"})
            break   # one mark per boundary
    for s in timed:
        s0, s1 = float(s.start_time), float(s.end_time)
        lo = bisect_right(starts, s0 - 10.0)
        over = [w for w in words[lo:] if float(w["s"]) < s1 and float(w["e"]) > s0]
        if not over:
            continue
        covered_end = max(min(float(w["e"]), s1) for w in over)
        covered_start = min(max(float(w["s"]), s0) for w in over)
        last_w = max(over, key=lambda w: float(w["e"]))
        first_w = min(over, key=lambda w: float(w["s"]))
        tail = s1 - covered_end
        if tail >= th["tail_s"] and _covered(covered_end, s1) < tail * 0.5:
            rows.append({"mark_class": "fa-trailing-audio",
                         "anchor": {"kind": "segment", "segment_id": s.id}, "t": covered_end,
                         "rationale": f"{tail:.2f}s of audio after the last aligned word "
                                      f"'{last_w.get('text', '')}' ({covered_end:.2f}s) before the segment ends "
                                      f"at {s1:.2f}s — no event insert explains it",
                         "key": f"fa-trailing-audio@{s.id}"})
        lead = covered_start - s0
        if lead >= th["tail_s"] and _covered(s0, covered_start) < lead * 0.5:
            rows.append({"mark_class": "fa-leading-audio",
                         "anchor": {"kind": "segment", "segment_id": s.id}, "t": s0,
                         "rationale": f"{lead:.2f}s of audio from the segment start ({s0:.2f}s) before the "
                                      f"first aligned word '{first_w.get('text', '')}' ({covered_start:.2f}s) — "
                                      f"no event insert explains it",
                         "key": f"fa-leading-audio@{s.id}"})
    return rows


def attention_boundary_marks(
    segments: List[SpineSegment],       # Effective spine (source-coordinate times)
    events: Optional[List[Dict[str, Any]]] = None,  # Active event inserts [{"start", "end", "label"}]
    fa_words: Optional[List[Dict[str, Any]]] = None,  # Aligned words in SOURCE seconds (gap + cut signals)
    thresholds: Optional[Dict[str, float]] = None,  # Overrides of ATTENTION_THRESHOLDS
) -> List[Dict[str, Any]]:  # Mark rows: {"mark_class", "anchor", "t", "rationale", "key"}
    """The boundary signals of the attention tier (item 3758f6cb signals 1+3 and the folded
    65791933 Tier-1 flags), pure. Over consecutive TEXT-BEARING segments: `speech-in-gap`
    — aligned words lie WHOLLY inside the gap between two segments (the fold homed text
    into a neighbour whose audio lacks it: VAD-missed speech or a stray carve — the
    scan-mishomed shape at the walk lane's grain; a word straddling the gap's edge is the
    mid-word boundary's finding, not this one, and a word an event insert mostly covers is
    the breath the aligner absorbed, not speech — first-walk sighting, an inhale flagged
    as 'to'); `segment-overlap`
    — the times cross by more than `overlap_s`; `cut-in-speech` — a gap below `cut_gap_s`
    with no event insert within `event_reach_s` and aligned speech within `cut_word_s` on
    BOTH sides (a boundary cut through continuous speech — the breath structure says no
    pause was here); `numeral-adjacency` — a number ends the left text or starts the right
    (the boundary that splits a figure from its unit)."""
    th = {**ATTENTION_THRESHOLDS, **(thresholds or {})}
    evs = sorted(((float(e.get("start") or 0.0), float(e.get("end") or 0.0), str(e.get("label") or ""))
                  for e in (events or [])), key=lambda e: e[0])
    ws = sorted(((float(w["s"]), float(w["e"]), str(w.get("text") or "")) for w in (fa_words or [])
                 if w.get("s") is not None and w.get("e") is not None), key=lambda p: p[0])
    w_ends = sorted(e for _, e, _ in ws)
    w_starts = [s for s, _, _ in ws]
    num_tail = re.compile(r"(\d[\d,.]*)\s*[%$]?[\"')\]]*$")
    num_head = re.compile(r"^[\"'(\[]*[$€£]?\d")

    def _covered(a: float, b: float) -> float:
        return sum(max(0.0, min(b, ee) - max(a, es)) for es, ee, _ in evs if es < b and ee > a)

    def _event_near(t: float) -> bool:
        r = th["event_reach_s"]
        return any(es - r <= t <= ee + r for es, ee, _ in evs)

    def _words_in(a: float, b: float) -> List[str]:  # words lying WHOLLY inside (a, b) — a straddler is mid-word's finding
        tol = th["overlap_s"]
        lo = bisect_right(w_starts, a - tol)
        out: List[str] = []
        for i in range(lo, len(ws)):
            s, e, text = ws[i]
            if s >= b:
                break
            if s >= a - tol and e <= b + tol and _covered(s, e) < 0.5 * max(e - s, 1e-6):
                out.append(text)   # a word an event insert (inhale …) mostly covers is the breath the aligner absorbed
        return out

    def _word_end_before(t: float) -> Optional[float]:
        i = bisect_right(w_ends, t) - 1
        return w_ends[i] if i >= 0 else None

    def _word_start_after(t: float) -> Optional[float]:
        i = bisect_right(w_starts, t)
        return w_starts[i] if i < len(w_starts) else None

    rows: List[Dict[str, Any]] = []
    timed = [s for s in segments if s.start_time is not None and s.end_time is not None
             and (s.text or "").strip()]
    for a, b in zip(timed, timed[1:]):
        a_end, b_start = float(a.end_time), float(b.start_time)
        anchor = {"kind": "boundary", "boundary_after": a.id, "right_segment_id": b.id}
        gap = b_start - a_end
        if gap < -th["overlap_s"]:
            rows.append({"mark_class": "segment-overlap", "anchor": anchor, "t": b_start,
                         "rationale": f"segments overlap by {-gap:.2f}s ({b_start:.2f}s starts before "
                                      f"{a_end:.2f}s ends)",
                         "key": f"segment-overlap@{a.id}|{b.id}"})
        elif gap >= th["gap_word_s"] and ws:
            inside = _words_in(a_end, b_start)
            if inside:
                rows.append({"mark_class": "speech-in-gap", "anchor": anchor, "t": a_end,
                             "rationale": f"{len(inside)} aligned word(s) inside the {gap:.2f}s gap "
                                          f"{a_end:.2f}-{b_start:.2f}s: '{' '.join(inside)[:60]}'",
                             "key": f"speech-in-gap@{a.id}|{b.id}"})
        # A long silence between text-bearing segments (the unexplained-gap class) no longer
        # marks — ruling 328c48ae on evidence 1dbbc512: a pause is not a finding.
        if ws and 0.0 <= gap < th["cut_gap_s"] and not _event_near(a_end):
            we, wsn = _word_end_before(a_end + th["cut_word_s"]), _word_start_after(b_start - th["cut_word_s"])
            if (we is not None and a_end - we <= th["cut_word_s"]
                    and wsn is not None and wsn - b_start <= th["cut_word_s"]):
                rows.append({"mark_class": "cut-in-speech", "anchor": anchor, "t": a_end,
                             "rationale": f"boundary at {a_end:.2f}s sits in continuous speech: aligned words "
                                          f"within {abs(a_end - we):.2f}s before and {abs(wsn - b_start):.2f}s "
                                          f"after, no breath/event insert within {th['event_reach_s']:.2f}s",
                             "key": f"cut-in-speech@{a.id}|{b.id}"})
        left, right = (a.text or "").rstrip(), (b.text or "").lstrip()
        if num_tail.search(left) or num_head.search(right):
            rows.append({"mark_class": "numeral-adjacency", "anchor": anchor, "t": a_end,
                         "rationale": f"a number touches the boundary: '…{left[-24:]}' | '{right[:24]}…'",
                         "key": f"numeral-adjacency@{a.id}|{b.id}"})
    return rows


def attention_divergence_marks(
    segments: List[SpineSegment],            # Effective spine (authoritative text)
    variants: Dict[str, Dict[str, str]],     # segment_id -> {transcriber: chunk text} (load_variant_texts)
    thresholds: Optional[Dict[str, float]] = None,  # Overrides of ATTENTION_THRESHOLDS
) -> List[Dict[str, Any]]:  # Mark rows: {"mark_class", "anchor", "t", "rationale", "key"}
    """The two-transcriber disagreement signal of the attention tier (item 3758f6cb signal
    2), pure — narrowed to what the WALK lane acts on: `asr-extra-words` on a segment whose
    second transcriber heard CONTENT words the authority lacks AT THE SEGMENT'S EDGES — a
    text-PLACEMENT disagreement (the other transcriber put those words in this chunk, the
    authority in a neighbour: the boundary shifts the first walk made, evidence 974f5500);
    mid-segment extra words are the fidelity lane's and do not mark. Filler-only differences (uh, um, like, the
    hedges a lightweight transcriber drops by design) and substitutions — same-length
    replacements, or a longer one whose letters are still half the authority's ('carpet
    the' for Karpathy: a mishearing re-split, the fidelity lane's material) — do not mark. A RUNAWAY variant is not a
    disagreement (finding 84f466bb): a token repeated `degenerate_run` times in a row, or
    a variant longer than `degenerate_ratio` times the authority, is skipped. The rationale
    quotes the extra words."""
    th = {**ATTENTION_THRESHOLDS, **(thresholds or {})}
    run_n, ratio = int(th["degenerate_run"]), float(th["degenerate_ratio"])
    min_extra = int(th.get("extra_words", 1))

    def _toks(text: str) -> List[str]:
        return _normalize_text(text or "").split()

    def _degenerate(auth: List[str], var: List[str]) -> bool:
        if len(var) > ratio * max(1, len(auth)):
            return True
        run = 1
        for x, y in zip(var, var[1:]):
            run = run + 1 if x == y else 1
            if run >= run_n:
                return True
        return False

    rows: List[Dict[str, Any]] = []
    for s in segments:
        if not (s.text or "").strip():
            continue
        auth = _toks(s.text)
        auth_set = set(auth)
        for transcriber, vtext in (variants.get(s.id) or {}).items():
            var = _toks(vtext)
            if var == auth or not var or _degenerate(auth, var):
                continue
            sm = difflib.SequenceMatcher(a=auth, b=var, autojunk=False)
            extra: List[str] = []
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag not in ("insert", "replace"):
                    continue
                if j1 != 0 and j2 != len(var):
                    continue   # mid-segment extra words are the fidelity lane's; placement shows at an edge
                if tag == "replace":
                    if (i2 - i1) >= (j2 - j1):
                        continue   # a substitution, not extra speech
                    a_join, b_join = "".join(auth[i1:i2]), "".join(var[j1:j2])
                    if levenshtein(a_join, b_join) <= 0.5 * max(len(a_join), len(b_join), 1):
                        continue   # a mishearing that merely re-splits the sound ('carpet the' for Karpathy)
                words = [w for w in var[j1:j2] if w not in ATTENTION_FILLERS and len(w) >= 3
                         and w not in auth_set]
                if words:
                    extra.append(" ".join(var[j1:j2]))
            if len(extra) < min_extra:
                continue
            shown = "; ".join(f"'{e[:40]}'" for e in extra[:3])
            rows.append({"mark_class": "asr-extra-words",
                         "anchor": {"kind": "segment", "segment_id": s.id},
                         "t": float(s.start_time) if s.start_time is not None else 0.0,
                         "rationale": f"{transcriber} heard words the authority lacks: {shown}",
                         "key": f"asr-extra-words@{s.id}"})
            break   # one mark per segment
    return rows


def attention_speaker_marks(
    segments: List[SpineSegment],  # Effective spine (source-coordinate times)
    turns: List[Dict[str, Any]],   # Diarization turns [{start, end, speaker}], source coordinates
) -> List[Dict[str, Any]]:  # Mark rows: {"mark_class", "anchor", "t", "rationale", "key"}
    """The next-speaker-change landmark of the folded item 65791933, pure: `speaker-change`
    on the boundary between consecutive text-bearing segments whose dominant diarization
    cluster differs (`speaker_turn_proposals` decides dominance). Navigation, not
    suspicion — opt-in (`speaker` is not a default signal), because an eight-speaker
    lecture would otherwise swamp the ⚑ set the walk lane jumps through."""
    dom = speaker_turn_proposals(segments, turns)
    rows: List[Dict[str, Any]] = []
    timed = [s for s in segments if s.start_time is not None and s.end_time is not None
             and (s.text or "").strip()]
    for a, b in zip(timed, timed[1:]):
        ca, cb = (dom.get(a.id) or {}).get("cluster"), (dom.get(b.id) or {}).get("cluster")
        if ca and cb and ca != cb:
            rows.append({"mark_class": "speaker-change",
                         "anchor": {"kind": "boundary", "boundary_after": a.id, "right_segment_id": b.id},
                         "t": float(a.end_time),
                         "rationale": f"dominant cluster changes {ca} -> {cb} at {float(a.end_time):.2f}s",
                         "key": f"speaker-change@{a.id}|{b.id}"})
    return rows


def attention_marks(
    segments: List[SpineSegment],                       # Effective spine (project_effective_spine output)
    *,
    fa_words: Optional[List[Dict[str, Any]]] = None,    # Aligned words in SOURCE seconds (None = FA signals off)
    events: Optional[List[Dict[str, Any]]] = None,      # Active event inserts [{start, end, label}]
    turns: Optional[List[Dict[str, Any]]] = None,       # Diarization turns (None = speaker signal off)
    signals: Optional[Sequence[str]] = None,            # Subset of ATTENTION_SIGNALS (default: ATTENTION_DEFAULT_SIGNALS)
    thresholds: Optional[Dict[str, float]] = None,      # Overrides of ATTENTION_THRESHOLDS
) -> List[Dict[str, Any]]:  # Mark rows in source-time order, deduplicated by key
    """Compose the attention tier (item 3758f6cb): every enabled signal's mark rows,
    merged, sorted by time, one row per (class, anchor) key. A signal whose input is
    absent contributes nothing — the tier degrades to what the stack has, never
    refuses. Pure: the CLI verb lands the rows as marks; this only derives them."""
    want = set(signals or ATTENTION_DEFAULT_SIGNALS)
    unknown = want - set(ATTENTION_SIGNALS)
    if unknown:
        raise ValueError(f"unknown attention signal(s): {sorted(unknown)} — "
                         f"known: {list(ATTENTION_SIGNALS)}")
    rows: List[Dict[str, Any]] = []
    if "fa" in want and fa_words:
        rows += attention_fa_marks(segments, fa_words, events=events, thresholds=thresholds)
    if want & {"gap", "cut", "numeral"}:
        allowed = {"gap": {"segment-overlap", "speech-in-gap"},
                   "cut": {"cut-in-speech"}, "numeral": {"numeral-adjacency"}}
        keep = set().union(*(allowed[s] for s in want if s in allowed))
        rows += [r for r in attention_boundary_marks(segments, events=events,
                                                     fa_words=fa_words if want & {"cut", "gap"} else None,
                                                     thresholds=thresholds)
                 if r["mark_class"] in keep]
    # The two-transcriber `divergence` signal (asr-extra-words) left the tier by ruling
    # 328c48ae (evidence 1dbbc512: dismissed whenever it stood alone); attention_divergence_marks
    # stays a pure helper for other consumers.
    if "speaker" in want and turns:
        rows += attention_speaker_marks(segments, turns)
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for r in sorted(rows, key=lambda r: (float(r.get("t") or 0.0), r["mark_class"])):
        if r["key"] in seen:
            continue
        seen.add(r["key"])
        out.append(r)
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


# ---- the walk-lane ATTENTION TIER (item 3758f6cb; ruling f400d2c3): derived marks, no model ----
ATTENTION_ACTOR = "capability:attention-tier"   # the actor every tier mark + dismissal carries (idempotency key)
ATTENTION_SIGNALS = ("fa", "gap", "numeral", "cut", "speaker")  # open set; `speaker` + `cut` are opt-in; `divergence` retired by ruling 328c48ae
ATTENTION_DEFAULT_SIGNALS = ("fa", "gap", "numeral")   # cut-in-speech opt-in since evidence 974f5500 (4/19); stretched-word / unexplained-gap / asr-extra-words removed by 328c48ae (evidence 1dbbc512)
ATTENTION_THRESHOLDS: Dict[str, float] = {
    "word_margin_s": 0.06,    # a boundary is INSIDE a word only past this margin from the word's edges (FA jitter; 0.08 missed a real cut — 974f5500)
    "tail_s": 0.6,            # audio after the last aligned word (or before the first) worth a listen
    "gap_word_s": 0.15,       # a gap this wide with an aligned word INSIDE it = speech the fold mis-homed
    "overlap_s": 0.02,        # consecutive segments whose times cross by more than this
    "cut_gap_s": 0.12,        # a cut in continuous speech: gap below this ...
    "cut_word_s": 0.15,       # ... with aligned words this close on both sides of the boundary
    "event_reach_s": 0.25,    # an event insert this near a boundary explains the cut
    "degenerate_run": 6,      # a token repeated this many times in a row = runaway variant, not disagreement
    "degenerate_ratio": 2.5,  # a variant this many times longer than the authority = runaway, not disagreement
    "extra_words": 1,         # extra-content-word runs a variant needs before it marks
}
ATTENTION_FILLERS = frozenset((  # tokens a lightweight transcriber drops by design — never "extra words"
    "uh", "um", "uhm", "hmm", "mm", "ah", "oh", "like", "so", "yeah", "okay", "ok", "right", "well",
    "you", "know", "i", "mean", "kind", "of", "sort", "just", "and", "the", "a", "an", "to", "it",
    "that", "this", "is", "was", "are", "be", "we", "they", "he", "she", "but", "or", "in", "on",
    "at", "for", "with", "there", "s", "t", "re", "ve", "ll", "d", "m", "then", "now", "also", "very",
))
