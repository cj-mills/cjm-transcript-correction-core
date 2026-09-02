"""Strata — the filtering lane's domain half (DECs 304fd984 + 9d4c0a38; work items
55bcc3c5 / 014a31b7).

Filtering IS the first pass of semantic classification: the domain op asserts a
CATEGORY (a stratum) over a run of spine segments — never a boolean irrelevance,
never a cut. "Filtered" becomes a PER-CONSUMER QUERY over strata (the notes
projection excludes tangents + sponsors + apparatus; a research agent pulls exactly
tool-mentions; a detector extract pulls disfluency runs). The immutable skeleton
holds; strata are Correction overlay nodes like marks, but they are STATE (consumers
project on them), not routed attention.

The lane is AGENT-FIRST from birth (DEC 3e1c260f) behind ONE interchangeable seam:

    pack  ->  proposer  ->  proposal set  ->  confirm (accept = stratum op)

* the PACK is what a proposer reads — the effective spine of one source window,
  numbered, with the vocabulary and the output contract rendered in (the read-trace
  of every proposal is the pack it came from, digest-bound);
* the PROPOSER is anything that turns a pack into proposal rows: a Claude Code
  sub-agent for pass 1, an in-core API call or a local model later — the contract
  never names the proposer kind beyond provenance;
* the PROPOSAL SET is durable inference output in the proposal-set-manifest shape the
  inhale lane already walks (dual-tier, windowed, source-bound) — verdicts are NEVER
  stored (DEC 8e05b87b): accept = a stratum op carrying the proposal id; rejects stay
  unmarked; accept/edit/relabel/reject DERIVE from joining the set against the live
  strata below the lane's watermark (`bench_filter_proposals`).

Windows ride the pack and the set from birth, so partitioning a multi-hour source
across proposers (item 90114c29) is additive, not a format break.
"""

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cjm_substrate.core.workspace import relativize_recorded
from cjm_transcript_correction_core.models import RECOMMENDED_STRATUM_CLASSES, SpineSegment

FILTER_PACK_FORMAT = "cjm-transcript-correction-core/filter-pack"
FILTER_PACK_VERSION = "0.1.0"
FILTER_PROPOSAL_SET_FORMAT = "cjm-transcript-correction-core/filtering-proposal-set"
FILTER_PROPOSAL_SET_VERSION = "0.1.0"
FILTER_LANE = "filter"   # The lane tag on the per-spine gate's watermark assertions

# Glosses for the recommended slate — rendered into every pack so a cold proposer
# reads the SAME class semantics the human confirms against (open vocabulary: a
# proposer may mint a new kebab-case class; it lands as data, never a release).
STRATUM_GLOSSES: Dict[str, str] = {
    "tangent": "an aside off the main topic — interesting or not, it is not what the source is about here",
    "tool-mention": "an off-hand mention of a tool / product / service worth pulling out for research",
    "sponsor": "a sponsor read or advertisement (products in it may still be research-worthy)",
    "research-mark": "a claim, citation, name, or reference a research pass should follow up",
    "disfluency": "hesitations, false starts, repeats — timestamp-detector training feedstock",
    "apparatus": "publishing apparatus: credits, dedication, legal, acknowledgments, chapter boilerplate",
}


def _fmt_ts(seconds: float) -> str:  # mm:ss.s for the rendered pack
    m, s = divmod(max(0.0, float(seconds)), 60.0)
    return f"{int(m):02d}:{s:04.1f}"


def _is_class_token(value: str) -> bool:  # Same rule as mark classes: letter/digit-led, non-empty
    v = (value or "").strip()
    return bool(v) and v[:1].isalnum()


def new_pack_id() -> str:  # e.g. "pack_20260901_180000_1a2b3c4d"
    """Generate a unique, sortable pack id."""
    return f"pack_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def build_filter_pack(
    source_id: str,                          # The Source node id
    title: str,                              # Display title (rendered; not identity)
    skeleton_hash: Optional[str],            # The spine the pack reads (None = legacy)
    segments: List[SpineSegment],            # The EFFECTIVE spine (corrections applied), index order
    *,
    content_hash: Optional[str] = None,      # Source media content hash (run-independent binding)
    window: Optional[Tuple[float, Optional[float]]] = None,  # (start, end) source seconds; None = whole spine
    strata: Optional[List[Dict[str, Any]]] = None,  # Active stratum correction dicts (rendered as context)
    vocabulary: Optional[List[str]] = None,  # Category slate (default: RECOMMENDED_STRATUM_CLASSES)
) -> Dict[str, Any]:  # The pack (JSON-serializable)
    """Build the proposer's input: one source window's text-bearing effective
    segments, numbered 0..n-1 in pack order, plus the vocabulary and the
    existing strata over the same window.

    Pack position `i` is the proposer's coordinate (rows reference from_i/to_i);
    the pack keeps each row's segment id + spine index + times, so ingest maps
    positions back to spine identity without a graph read. Empty-text segments
    (silence chunks, wordless inserts) are left out — nothing to classify — so
    a stratum spans text segments only."""
    w_start = float(window[0]) if window and window[0] is not None else None
    w_end = float(window[1]) if window and window[1] is not None else None
    rows: List[Dict[str, Any]] = []
    for s in segments:
        if s.is_empty:
            continue
        st = float(s.start_time) if s.start_time is not None else None
        en = float(s.end_time) if s.end_time is not None else None
        if w_start is not None and en is not None and en <= w_start:
            continue
        if w_end is not None and st is not None and st >= w_end:
            continue
        rows.append({"i": len(rows), "id": s.id, "index": s.index,
                     "start": st, "end": en, "text": s.text})
    timed_starts = [r["start"] for r in rows if r["start"] is not None]
    timed_ends = [r["end"] for r in rows if r["end"] is not None]
    win = {"start": (w_start if w_start is not None else (min(timed_starts) if timed_starts else 0.0)),
           "end": (w_end if w_end is not None else (max(timed_ends) if timed_ends else None))}
    vocab = list(vocabulary) if vocabulary else list(RECOMMENDED_STRATUM_CLASSES)
    pos_by_id = {r["id"]: r["i"] for r in rows}
    existing: List[Dict[str, Any]] = []
    for c in (strata or []):
        p = c.get("payload") or {}
        ids = [sid for sid in (p.get("segment_ids") or []) if sid in pos_by_id]
        if not ids:
            continue
        existing.append({"stratum_id": c.get("id"), "category": p.get("category"),
                         "from_i": min(pos_by_id[i] for i in ids),
                         "to_i": max(pos_by_id[i] for i in ids),
                         "actor": c.get("actor")})
    pack = {
        "format": FILTER_PACK_FORMAT,
        "version": FILTER_PACK_VERSION,
        "pack_id": new_pack_id(),
        "created_at": time.time(),
        "source": {"source_id": source_id, "title": title,
                   "content_hash": content_hash, "skeleton_hash": skeleton_hash},
        "window": win,
        "vocabulary": [{"category": c, "gloss": STRATUM_GLOSSES.get(c, "")} for c in vocab],
        "existing_strata": existing,
        "segments": rows,
    }
    pack["digest"] = pack_digest(pack)
    return pack


def pack_digest(pack: Dict[str, Any]) -> str:  # "sha256:<hex>" over the content a proposer read
    """Digest the READ content (source binding + window + numbered segments) —
    what a proposal set records so the read-trace is verifiable, independent of
    pack id / timestamps."""
    body = {"source": pack.get("source"), "window": pack.get("window"),
            "segments": [[r["i"], r["id"], r["start"], r["end"], r["text"]]
                         for r in pack.get("segments") or []]}
    h = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return f"sha256:{h.hexdigest()}"


OUTPUT_CONTRACT = """\
## Output contract

Write ONE JSON object per line (JSONL). Each row proposes ONE stratum over a run of
consecutive pack lines:

    {"category": "<class>", "from_i": <int>, "to_i": <int>, "tier": 1|2,
     "confidence": <0.0-1.0>, "rationale": "<one sentence>", "quote": "<short verbatim>"}

* `category`: a class from the vocabulary above, or a NEW kebab-case class when none fits
  (say why in the rationale). Do NOT propose the main topic — absence of a stratum IS
  main-topic.
* `from_i`/`to_i`: inclusive pack line numbers (the `[i]` prefixes). Runs must not
  overlap another row of the same category.
* `tier`: 1 = you are confident enough that a human batch-accept is reasonable;
  2 = borderline, audition only (dim in the walk, never batch-accepted).
* `confidence`: your own calibration, 0..1.
* `quote`: a few verbatim words from the run, for the human to find it fast.

Rows only — no prose before or after, no code fences.
"""


def render_filter_pack(pack: Dict[str, Any]) -> str:  # The proposer brief (markdown)
    """Render a pack as the brief a proposer reads: identity + window, the class
    slate with glosses, the strata already asserted over this window, the output
    contract, then the numbered segment lines. Deterministic for a given pack."""
    src = pack.get("source") or {}
    win = pack.get("window") or {}
    end_txt = _fmt_ts(win["end"]) if win.get("end") is not None else "end"
    lines: List[str] = [
        f"# Filtering pack `{pack.get('pack_id')}`",
        "",
        f"Source: **{src.get('title') or src.get('source_id')}**  "
        f"(`{src.get('source_id')}`; spine `{(src.get('skeleton_hash') or 'legacy')[-12:]}`)",
        f"Window: {_fmt_ts(win.get('start') or 0.0)} – {end_txt}  ·  "
        f"{len(pack.get('segments') or [])} text segments  ·  digest `{pack.get('digest', '')[-12:]}`",
        "",
        "## Task",
        "",
        "Read the numbered transcript lines below and propose STRATA: runs of lines that",
        "belong to one of the classes in the vocabulary. Everything you do not mark is",
        "main-topic content. Prefer precise runs over generous ones; a run may be one line.",
        "",
        "## Vocabulary",
        "",
    ]
    for v in pack.get("vocabulary") or []:
        lines.append(f"- `{v['category']}` — {v.get('gloss') or ''}".rstrip(" —"))
    existing = pack.get("existing_strata") or []
    lines += ["", "## Already asserted over this window", ""]
    if existing:
        for e in existing:
            lines.append(f"- `{e['category']}` lines {e['from_i']}–{e['to_i']} "
                         f"(by {e.get('actor') or '?'}) — do not re-propose")
    else:
        lines.append("- (none)")
    lines += ["", OUTPUT_CONTRACT, "## Transcript", ""]
    for r in pack.get("segments") or []:
        st = _fmt_ts(r["start"]) if r.get("start") is not None else "--:--"
        en = _fmt_ts(r["end"]) if r.get("end") is not None else "--:--"
        lines.append(f"[{r['i']}] {st}–{en}  {r['text']}")
    return "\n".join(lines) + "\n"


def validate_proposal_rows(
    rows: List[Dict[str, Any]],  # Raw proposer output rows (parsed JSONL)
    pack: Dict[str, Any],        # The pack the rows reference
) -> List[Dict[str, Any]]:  # Normalized rows (category/from_i/to_i/tier/confidence/rationale/quote)
    """Validate + normalize proposer rows against their pack — loud on the first
    bad row (row number in the message). Enforces the contract: class token,
    in-range inclusive run, tier in {1, 2}, confidence in [0, 1], no same-class
    overlap between rows."""
    n = len(pack.get("segments") or [])
    out: List[Dict[str, Any]] = []
    seen: Dict[str, List[Tuple[int, int]]] = {}
    for k, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"row {k}: not an object")
        cat = str(raw.get("category") or "").strip()
        if not _is_class_token(cat):
            raise ValueError(f"row {k}: category must be a letter/digit-led class token, got {cat!r}")
        try:
            fi, ti = int(raw.get("from_i")), int(raw.get("to_i"))
        except (TypeError, ValueError):
            raise ValueError(f"row {k}: from_i/to_i must be integers")
        if not (0 <= fi <= ti < n):
            raise ValueError(f"row {k}: run {fi}..{ti} outside the pack (0..{n - 1}) or inverted")
        tier = int(raw.get("tier", 1) or 1)
        if tier not in (1, 2):
            raise ValueError(f"row {k}: tier must be 1 or 2, got {tier}")
        conf = raw.get("confidence")
        if conf is not None:
            conf = float(conf)
            if not (0.0 <= conf <= 1.0):
                raise ValueError(f"row {k}: confidence must be within [0, 1], got {conf}")
        for (a, b) in seen.get(cat, []):
            if fi <= b and ti >= a:
                raise ValueError(f"row {k}: {cat} run {fi}..{ti} overlaps an earlier {cat} run {a}..{b}")
        seen.setdefault(cat, []).append((fi, ti))
        out.append({"category": cat, "from_i": fi, "to_i": ti, "tier": tier,
                    "confidence": conf,
                    "rationale": str(raw.get("rationale") or "").strip() or None,
                    "quote": str(raw.get("quote") or "").strip() or None})
    return out


def proposals_from_rows(
    rows: List[Dict[str, Any]],  # validate_proposal_rows output
    pack: Dict[str, Any],        # The pack the rows reference
) -> List[Dict[str, Any]]:  # Proposal-set rows (time-ordered), pack positions resolved to spine identity
    """Resolve validated rows to proposal-set rows: proposal id, category, source
    times, the covered segment ids, tier, confidence, rationale, and the evidence
    read-trace (pack id + run + quote)."""
    segs = pack.get("segments") or []
    out: List[Dict[str, Any]] = []
    for r in rows:
        run = segs[r["from_i"]:r["to_i"] + 1]
        starts = [s["start"] for s in run if s.get("start") is not None]
        ends = [s["end"] for s in run if s.get("end") is not None]
        out.append({
            "proposal_id": str(uuid.uuid4()),
            "category": r["category"],
            "label": r["category"],          # the propset walkers' generic key
            "start_time": (round(min(starts), 4) if starts else None),
            "end_time": (round(max(ends), 4) if ends else None),
            "segment_ids": [s["id"] for s in run],
            "tier": r["tier"],
            "confidence": r.get("confidence"),
            "score": r.get("confidence"),   # the propset walkers' generic key
            "rationale": r.get("rationale"),
            "evidence": {"pack_id": pack.get("pack_id"), "from_i": r["from_i"],
                         "to_i": r["to_i"], "quote": r.get("quote")},
        })
    out.sort(key=lambda p: (p["start_time"] if p["start_time"] is not None else 0.0,
                            p["evidence"]["from_i"]))
    return out


def write_filter_propset(
    pack: Dict[str, Any],                 # The pack the proposals came from
    proposals: List[Dict[str, Any]],      # proposals_from_rows output
    *,
    out_root: Path,                       # Proposal-set root (<workspace>/proposals)
    proposer: Dict[str, Any],             # Provenance: {"kind": "claude-code-subagent" | "api" | ..., "name": ..., ...}
    ws: Any = None,                       # Resolved workspace (relativize_recorded) or None
) -> Dict[str, Any]:  # {"set_id","set_dir","manifest_path","classes","counts","tier2_counts"}
    """Write one filtering proposal set: `<out_root>/<set_id>/manifest.json` +
    `proposals.jsonl` — the durable half of the derived-verdicts contract. The
    manifest binds the SOURCE (id + content hash + skeleton), the WINDOW, the
    PROPOSER and the PACK DIGEST it read, so the bench join and the provenance
    pane can name exactly what was proposed, by whom, over what."""
    started = time.time()
    set_id = (f"propset_{time.strftime('%Y%m%d_%H%M%S', time.localtime(started))}"
              f"_{uuid.uuid4().hex[:8]}")
    set_dir = Path(out_root) / set_id
    set_dir.mkdir(parents=True)
    with open(set_dir / "proposals.jsonl", "w") as f:
        for p in proposals:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    counts: Dict[str, int] = {}
    tier2: Dict[str, int] = {}
    for p in proposals:
        bucket = counts if int(p.get("tier", 1)) == 1 else tier2
        bucket[p["category"]] = bucket.get(p["category"], 0) + 1
    src = dict(pack.get("source") or {})
    manifest = {
        "format": FILTER_PROPOSAL_SET_FORMAT,
        "version": FILTER_PROPOSAL_SET_VERSION,
        "proposal_set_id": set_id,
        "created_at": started,
        "config": {"lane": FILTER_LANE,
                   "vocabulary": [v["category"] for v in (pack.get("vocabulary") or [])]},
        "model": dict(proposer),
        "pack": {"pack_id": pack.get("pack_id"), "digest": pack.get("digest"),
                 "segments": len(pack.get("segments") or [])},
        "source": src,
        "window": dict(pack.get("window") or {}),
        "classes": sorted(set(counts) | set(tier2)),
        "files": {"proposals": "proposals.jsonl"},
        "counts": counts,
        "tier2_counts": tier2,
    }
    manifest_path = set_dir / "manifest.json"
    manifest_path.write_text(json.dumps(relativize_recorded(manifest, ws), indent=2,
                                        ensure_ascii=False))
    return {"set_id": set_id, "set_dir": str(set_dir), "manifest_path": str(manifest_path),
            "classes": manifest["classes"], "counts": counts, "tier2_counts": tier2}


def render_filter_propset_markdown(
    manifest: Dict[str, Any],                 # The proposal set's manifest
    proposals: List[Dict[str, Any]],          # Its rows
    pack: Optional[Dict[str, Any]] = None,    # The pack the rows reference (None = quote-only rendering)
    context: int = 1,                         # Pack lines of context shown before/after each run
) -> str:  # A human-readable projection of the set (markdown)
    """Project a proposal set for a HUMAN to check against the source in the
    correction app: one block per proposal — category, source times, the SPINE
    INDEX range (the app's coordinate), tier, confidence, the full rationale,
    the quote, and the verbatim segment run from the pack with a line of
    context either side. Time order. Nothing is decided here; this is the
    printed worklist the CLI's list mode cannot fit on one line."""
    src = manifest.get("source") or {}
    model = manifest.get("model") or {}
    win = manifest.get("window") or {}
    rows = sorted(proposals or [],
                  key=lambda p: (float(p.get("start_time") or 0.0),
                                 int(((p.get("evidence") or {}).get("from_i")) or 0)))
    segs = (pack or {}).get("segments") or []
    tag = (src.get("skeleton_hash") or "legacy").split(":")[-1][:8]
    end_txt = _fmt_ts(win["end"]) if win.get("end") is not None else "end"
    counts = manifest.get("counts") or {}
    t2 = manifest.get("tier2_counts") or {}
    lines: List[str] = [
        f"# Filtering proposals — {src.get('title') or src.get('source_id')}",
        "",
        f"Set `{manifest.get('proposal_set_id')}` · spine `{tag}` · window "
        f"{_fmt_ts(win.get('start') or 0.0)}–{end_txt} · proposer "
        f"{model.get('kind') or '?'}:{model.get('name') or '?'}"
        + (f" ({model.get('model')})" if model.get("model") else ""),
        f"Tier 1: {' · '.join(f'{k}×{v}' for k, v in sorted(counts.items())) or 'none'}  ·  "
        f"Tier 2: {' · '.join(f'{k}×{v}' for k, v in sorted(t2.items())) or 'none'}"
        + ("" if pack else "  ·  (pack not found — runs shown by quote only)"),
        "",
        "Spine index = the segment position the correction app shows; times are source "
        "seconds. `?` = tier 1 (batch-acceptable), `??` = tier 2 (audition).",
        "",
    ]
    for n, p in enumerate(rows, start=1):
        ev = p.get("evidence") or {}
        fi, ti = ev.get("from_i"), ev.get("to_i")
        run = segs[fi:ti + 1] if (segs and fi is not None and ti is not None) else []
        idx_txt = (f"spine {run[0]['index']}–{run[-1]['index']}" if run
                   else f"pack lines {fi}..{ti}")
        tier = "??" if int(p.get("tier", 1)) == 2 else "?"
        conf = p.get("confidence")
        st = _fmt_ts(p["start_time"]) if p.get("start_time") is not None else "--:--"
        en = _fmt_ts(p["end_time"]) if p.get("end_time") is not None else "--:--"
        lines.append(f"## {n}. `{tier}` **{p.get('category')}** · {st}–{en} · {idx_txt}"
                     + (f" · c={conf:.2f}" if isinstance(conf, (int, float)) else "")
                     + f" · id `…{str(p.get('proposal_id') or '')[-8:]}`")
        lines.append("")
        if p.get("rationale"):
            lines.append(f"**Why:** {p['rationale']}")
            lines.append("")
        if ev.get("quote"):
            lines.append(f"**Quote:** “{ev['quote']}”")
            lines.append("")
        if run:
            lo = max(0, fi - context)
            hi = min(len(segs), ti + 1 + context)
            for r in segs[lo:hi]:
                inside = fi <= r["i"] <= ti
                rst = _fmt_ts(r["start"]) if r.get("start") is not None else "--:--"
                mark = "**" if inside else ""
                pre = "" if inside else "_"
                post = "" if inside else "_"
                lines.append(f"> {mark}[{r['i']}]{mark} {rst} · spine {r['index']} — "
                             f"{pre}{r['text']}{post}")
            lines.append("")
    if not rows:
        lines.append("_(no proposals in this set)_")
        lines.append("")
    return "\n".join(lines)


def load_filter_proposal_sets(
    ws_root: str,                         # Workspace root (sets live under <root>/proposals/)
    source_id: str,                       # Source node id to match
    skeleton_hash: Optional[str] = None,  # Restrict to sets bound to this spine (None = any)
) -> List[Dict[str, Any]]:  # [{"manifest","path","proposals"}] newest first; malformed sets skipped
    """Every filtering proposal set for a source (and optionally one spine),
    newest first — several coexist by design (proposers, windows, generations),
    so the caller picks; `latest` = index 0."""
    root = Path(ws_root) / "proposals"
    if not root.is_dir():
        return []
    found: List[Dict[str, Any]] = []
    for mp in sorted(root.glob("*/manifest.json")):
        try:
            m = json.loads(mp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("format") != FILTER_PROPOSAL_SET_FORMAT:
            continue
        src = m.get("source") or {}
        if src.get("source_id") != source_id:
            continue
        if skeleton_hash is not None and src.get("skeleton_hash") not in (None, skeleton_hash):
            continue
        data_file = mp.parent / str((m.get("files") or {}).get("proposals") or "proposals.jsonl")
        try:
            proposals = [json.loads(line) for line in data_file.read_text().splitlines()
                         if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        found.append({"manifest": m, "path": str(mp), "proposals": proposals})
    found.sort(key=lambda d: float(d["manifest"].get("created_at") or 0.0), reverse=True)
    return found


def active_strata(
    corrections: List[Dict[str, Any]],  # Corrections (e.g. from load_source_corrections)
    superseded_ids: set,                # Ids that are SUPERSEDES targets
) -> List[Dict[str, Any]]:  # The live stratum corrections, time order
    """The live strata: stratum corrections neither superseded (reclassified /
    retracted) nor still `proposed`."""
    out = [c for c in corrections
           if c.get("correction_type") == "stratum"
           and c.get("id") not in superseded_ids
           and c.get("status") != "proposed"]
    out.sort(key=lambda c: float((c.get("payload") or {}).get("start_time") or 0.0))
    return out


def strata_index(
    strata: List[Dict[str, Any]],  # active_strata output
) -> Dict[str, List[str]]:  # segment id -> categories asserted over it
    """Segment-keyed view of the live strata (the consumer query's index)."""
    out: Dict[str, List[str]] = {}
    for c in strata:
        p = c.get("payload") or {}
        for sid in p.get("segment_ids") or []:
            cats = out.setdefault(sid, [])
            if p.get("category") not in cats:
                cats.append(p.get("category"))
    return out


def exclude_strata(
    segments: List[SpineSegment],       # The effective spine
    strata: List[Dict[str, Any]],       # active_strata output
    categories: List[str],              # Categories a consumer EXCLUDES (e.g. tangent + sponsor + apparatus)
) -> List[SpineSegment]:  # The spine minus segments under any excluded stratum
    """The per-consumer filtered projection (DEC 9d4c0a38): "filtered" is a
    query, so each consumer names what it excludes; the skeleton is untouched."""
    idx = strata_index(strata)
    drop = set(categories)
    return [s for s in segments if not (drop & set(idx.get(s.id, [])))]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:  # Seconds of overlap
    return max(0.0, min(a1, b1) - max(a0, b0))


def _iou(a0: float, a1: float, b0: float, b1: float) -> float:  # Intersection over union, 0 on degenerate
    inter = _overlap(a0, a1, b0, b1)
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def materialized_mark_ids(
    corrections: List[Dict[str, Any]],  # Corrections (e.g. from load_source_corrections)
    superseded_ids: set,                # Ids that are SUPERSEDES targets
) -> set:  # proposal ids that OPEN marks carry (mark-family rows accepted AS marks)
    """Class-family routing's other half: a proposer's mark-family row (an ASR
    error, a suspect noun) is accepted as a MARK, never a stratum — the mark's
    payload carries the proposal id, so the worklist and the bench see it as
    materialized exactly like a stratum accept."""
    return {(c.get("payload") or {}).get("proposal_id") for c in corrections
            if c.get("correction_type") == "mark"
            and c.get("id") not in superseded_ids
            and (c.get("payload") or {}).get("proposal_id")}


def pending_filter_proposals(
    proposals: List[Dict[str, Any]],  # A proposal set's rows
    strata: List[Dict[str, Any]],     # active_strata output
    show_tier2: bool = False,         # Include the audition tier
    materialized: Optional[set] = None,  # Extra proposal ids already materialized (e.g. as marks)
) -> List[Dict[str, Any]]:  # Proposals not yet materialized, time order
    """The headless worklist: proposals with NO live stratum carrying their id and
    NO same-category stratum overlapping them in time (accepted in some session
    — the verdict join owns history). Tier 2 hides by default (dual-tier
    doctrine a475ccd6: audition never joins the primary walk uninvited)."""
    by_pid = {(c.get("payload") or {}).get("proposal_id") for c in strata} | set(materialized or ())
    spans = [((c.get("payload") or {}).get("category"),
              float((c.get("payload") or {}).get("start_time") or 0.0),
              float((c.get("payload") or {}).get("end_time") or 0.0)) for c in strata]
    out: List[Dict[str, Any]] = []
    for p in sorted(proposals, key=lambda d: float(d.get("start_time") or 0.0)):
        if not show_tier2 and int(p.get("tier", 1)) == 2:
            continue
        if p.get("proposal_id") in by_pid:
            continue
        ps, pe = float(p.get("start_time") or 0.0), float(p.get("end_time") or 0.0)
        if any(cat == p.get("category") and _overlap(ps, pe, s0, s1) > 0.0
               for cat, s0, s1 in spans):
            continue
        out.append(p)
    return out


def bench_filter_proposals(
    proposals: List[Dict[str, Any]],       # A proposal set's rows
    strata: List[Dict[str, Any]],          # active_strata output (the live final state)
    window: Tuple[float, Optional[float]], # (start, end) the proposals covered; end None = unbounded
    *,
    watermark: Optional[float] = None,     # The lane's annotated_through (None = nothing visited)
    iou_tolerance: float = 0.9,            # Same-category overlap at/above which a match is ACCEPTED (else EDITED)
    mark_ids: Optional[set] = None,        # Proposal ids materialized AS MARKS (class-family routing) — ACCEPTED, family mark
) -> Dict[str, Any]:  # {"counts", "rates", "verdicts", "missed"}
    """Derive the filtering verdicts (DEC 8e05b87b, the bench_event_proposals
    sibling) — pure, nothing stored.

    Per proposal: a live stratum carrying its proposal id (or the best time-
    overlapping stratum) decides — same category with IoU >= tolerance =
    ACCEPTED; same category but boundaries moved = EDITED; a different category
    = RELABELED. No match BELOW the watermark = REJECTED (absence is a verdict
    only where the human walked); no match above it = UNVISITED. Tier-2 rows
    join the remainder and read 'unaccepted' instead of rejected (an unshown
    audition is not a verdict). MISSED = live strata inside the window that no
    proposal matched — classes the proposer failed to raise. Rates are the
    tier-1 operating-point contract."""
    w0 = float(window[0])
    w1 = float(window[1]) if window[1] is not None else None

    def in_window(t: float) -> bool:
        return t >= w0 and (w1 is None or t < w1)

    live = []
    for c in strata:
        p = c.get("payload") or {}
        if p.get("start_time") is None:
            continue
        s0 = float(p["start_time"])
        if in_window(s0):
            live.append({"stratum_id": c.get("id"), "category": p.get("category"),
                         "start": s0, "end": float(p.get("end_time") or s0),
                         "proposal_id": p.get("proposal_id"), "actor": c.get("actor")})
    unmatched = {id(r): r for r in live}
    rows = sorted(proposals or [], key=lambda d: float(d.get("start_time") or 0.0))
    tier1 = [p for p in rows if int(p.get("tier", 1)) == 1]
    tier2 = [p for p in rows if int(p.get("tier", 1)) == 2]

    def join(ordered: List[Dict[str, Any]], no_match: str) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        matches: Dict[int, Dict[str, Any]] = {}
        # pass 0: exact proposal-id carry (the accept gesture records it)
        for p in ordered:
            for key, r in list(unmatched.items()):
                if r["proposal_id"] and r["proposal_id"] == p.get("proposal_id"):
                    matches[id(p)] = unmatched.pop(key)
                    break
        # pass 1: same-category best IoU; pass 2: any-category best IoU (relabeled)
        for same in (True, False):
            pairs: List[Tuple[float, int, int]] = []
            for p in ordered:
                if id(p) in matches:
                    continue
                ps, pe = float(p.get("start_time") or 0.0), float(p.get("end_time") or 0.0)
                for key, r in unmatched.items():
                    if same and r["category"] != p.get("category"):
                        continue
                    iou = _iou(ps, pe, r["start"], r["end"])
                    if iou > 0.0:
                        pairs.append((iou, id(p), key))
            for _iou_v, pid, key in sorted(pairs, key=lambda t: -t[0]):
                if pid in matches or key not in unmatched:
                    continue
                matches[pid] = unmatched.pop(key)
        counts = {"accepted": 0, "edited": 0, "relabeled": 0, no_match: 0, "unvisited": 0}
        verdicts: List[Dict[str, Any]] = []
        for p in ordered:
            ps, pe = float(p.get("start_time") or 0.0), float(p.get("end_time") or 0.0)
            m = matches.get(id(p))
            if m is None and p.get("proposal_id") in (mark_ids or ()):
                counts["accepted"] += 1
                verdicts.append({"proposal_id": p.get("proposal_id"), "category": p.get("category"),
                                 "start_time": ps, "end_time": pe, "tier": int(p.get("tier", 1)),
                                 "confidence": p.get("confidence"), "verdict": "accepted",
                                 "family": "mark"})
                continue
            if m is None:
                verdict = (no_match if (watermark is not None and ps < float(watermark))
                           else "unvisited")
            elif m["category"] != p.get("category"):
                verdict = "relabeled"
            elif _iou(ps, pe, m["start"], m["end"]) >= iou_tolerance:
                verdict = "accepted"
            else:
                verdict = "edited"
            counts[verdict] += 1
            verdicts.append({"proposal_id": p.get("proposal_id"), "category": p.get("category"),
                             "start_time": ps, "end_time": pe, "tier": int(p.get("tier", 1)),
                             "confidence": p.get("confidence"), "verdict": verdict,
                             **({"stratum_id": m["stratum_id"], "stratum_category": m["category"],
                                 "stratum_start": m["start"], "stratum_end": m["end"]} if m else {})})
        return counts, verdicts

    c1, v1 = join(tier1, "rejected")
    c2, v2 = join(tier2, "unaccepted")
    decided = c1["accepted"] + c1["edited"] + c1["relabeled"] + c1["rejected"]
    rates = ({k: round(c1[k] / decided, 3) for k in ("accepted", "edited", "relabeled", "rejected")}
             if decided else {})
    missed = sorted(unmatched.values(), key=lambda r: r["start"])
    return {"counts": {"tier1": c1, "tier2": c2}, "rates": rates,
            "verdicts": v1 + v2, "missed": missed}
