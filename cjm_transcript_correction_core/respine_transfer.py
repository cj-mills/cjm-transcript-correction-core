"""The chunk-scoped transfer a chunk respine calls (work item 7a5e9c84; ruling 0b4d5cfa (5),
amendment 4a7ec4f8): re-home the two SOURCE-truth correction classes from a chunk's old
segments onto its re-derived segments BY TIME, in one CorrectionSession.

  events    accepted, wordless, labeled chunk inserts (the propose lane's event layer —
            inhale spans and their kin) re-anchor by source time between the new
            layer-0 segments through the wordless-transfer planner (plan_transfer_rows:
            the donor set is the effective wordless layer, time nudges applied,
            dup-guarded), exactly as a whole-spine transfer does — chunk-scoped.
  speakers  speaker assignments are anchored on SEGMENT IDS, but who speaks at time t
            is source-truth: the chunk's active assignments project onto a per-span
            speaker map from the old segments' effective spans, and each new segment
            takes the one entity whose span covers it. A new segment that STRADDLES
            two donor speakers stays UNASSIGNED (a listed gap the assign lane
            surfaces, never a guess). Carried assignments commit as fresh
            speaker_assign Corrections whose proposal snapshot names the transfer and
            the donor correction, so the flywheel row stays honest.

Discovered by decomp-core through the `cjm_transcript_decomp_core.chunk_transfer`
entry-point group (pyproject) — the replay-registry shape, so decomp never depends on
this core. What stays behind by design: marks, reviews, nudges, text edits, prunes,
splits, word-bearing inserts — spine-truth the escalated text replaces (the verb
strands them on the operator's explicit say-so)."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from cjm_context_graph_layer.ops import graph_task
from cjm_context_graph_primitives.query import NodeQuery, PropertyPredicate
from cjm_transcript_correction_core.graph import (_row_to_spine_segment, _SPINE_PROJECTION,
                                                  active_speaker_assignments,
                                                  commit_chunk_insert_correction,
                                                  commit_speaker_assign_correction,
                                                  load_source_corrections, project_effective_spine,
                                                  set_session_status, start_session)
from cjm_transcript_correction_core.models import SpineSegment
from cjm_transcript_graph_schema.schema import SEGMENT_SUPERSEDED_BY_PROP, TranscriptGraphLabels

logger = logging.getLogger(__name__)

TRANSFER_PURPOSE = "chunk-respine-transfer"  # The CorrectionSession purpose tag the carried rows land under
STRADDLE_EPS_S = 0.05  # An overlap shorter than this is a boundary sliver, not coverage


# ---- pure planners -------------------------------------------------------------------------

def speaker_spans(
    old_units: List[SpineSegment],          # The chunk's old segments, EFFECTIVE times (nudges applied)
    assignments: Dict[str, Dict[str, Any]],  # active_speaker_assignments output (segment id -> {entity_id, verdict, correction_id})
) -> List[Dict[str, Any]]:  # [{start, end, entity_id, verdict, correction_id}] for every assigned old segment, time order
    """The per-span speaker map the carry reads (pure)."""
    out: List[Dict[str, Any]] = []
    for u in old_units:
        a = assignments.get(u.id)
        if not a or not a.get("entity_id") or u.start_time is None or u.end_time is None:
            continue
        out.append({"start": float(u.start_time), "end": float(u.end_time),
                    "entity_id": str(a["entity_id"]), "verdict": str(a.get("verdict") or "name"),
                    "correction_id": str(a.get("correction_id") or "")})
    out.sort(key=lambda r: r["start"])
    return out


def plan_speaker_carry(
    spans: List[Dict[str, Any]],        # speaker_spans output
    new_segments: List[SpineSegment],   # The replacement segments, index order (layer-0 times)
    eps: float = STRADDLE_EPS_S,        # Minimum overlap that counts as coverage
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:  # (rows {segment_id, entity_id, verdict, donors}, straddles, uncovered)
    """Which new segment takes which speaker (pure; 4a7ec4f8 (2)/(3)): a segment whose
    extent overlaps exactly ONE donor entity takes it; two or more distinct entities =
    a STRADDLE, left unassigned; none = uncovered (the old segment there carried no
    assignment). Boundary slivers under `eps` never count — a cut a few ms into the
    neighbour's turn is alignment noise, not a second speaker."""
    rows: List[Dict[str, Any]] = []
    straddles: List[str] = []
    uncovered: List[str] = []
    for s in new_segments:
        if s.start_time is None or s.end_time is None:
            uncovered.append(s.id)
            continue
        st, en = float(s.start_time), float(s.end_time)
        hits: Dict[str, Dict[str, Any]] = {}
        for r in spans:
            ov = min(en, r["end"]) - max(st, r["start"])
            # Coverage = more than a sliver (float-safe: 0.05 - 0.05 rounds to ~1e-11);
            # a segment shorter than the sliver itself counts any real overlap.
            if ov - eps > 1e-6 or (ov > 1e-6 and (en - st) <= eps):
                h = hits.setdefault(r["entity_id"], {"verdict": r["verdict"], "donors": [], "overlap": 0.0})
                if r["correction_id"] not in h["donors"]:
                    h["donors"].append(r["correction_id"])
                h["overlap"] += ov
        if not hits:
            uncovered.append(s.id)
        elif len(hits) == 1:
            eid, h = next(iter(hits.items()))
            rows.append({"segment_id": s.id, "entity_id": eid, "verdict": h["verdict"],
                         "donors": [d for d in h["donors"] if d]})
        else:
            straddles.append(s.id)
    return rows, straddles, uncovered


def group_speaker_rows(
    rows: List[Dict[str, Any]],        # plan_speaker_carry rows, new-segment index order
) -> List[Dict[str, Any]]:  # [{entity_id, verdict, segment_ids, donors}] — one per contiguous same-entity run
    """Fold per-segment rows into turn-shaped commits (pure): contiguous same-entity
    segments become ONE speaker_assign Correction, the shape the assign lane's own
    accept gesture mints."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        if out and out[-1]["entity_id"] == r["entity_id"]:
            out[-1]["segment_ids"].append(r["segment_id"])
            for d in r["donors"]:
                if d not in out[-1]["donors"]:
                    out[-1]["donors"].append(d)
            continue
        out.append({"entity_id": r["entity_id"], "verdict": r["verdict"],
                    "segment_ids": [r["segment_id"]], "donors": list(r["donors"])})
    return out


# ---- graph reads ------------------------------------------------------------------------------

async def load_segments_by_ids(
    queue: Any, graph_id: str,
    ids: List[str],  # Segment ids (any spine, live or superseded — the respine reads both sides)
) -> List[SpineSegment]:  # SpineSegments in index order
    """Segments by id — the ONE read that bypasses the live predicate on purpose: the
    transfer runs while both the old and the new segments exist."""
    out: List[SpineSegment] = []
    for i in range(0, len(ids), 500):
        q = NodeQuery(ids=list(ids[i:i + 500]), label=TranscriptGraphLabels.SEGMENT,
                      project=list(_SPINE_PROJECTION))
        res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
        out.extend(_row_to_spine_segment(r) for r in (res.rows or []))
    return sorted(out, key=lambda s: s.index)


async def live_neighbour(
    queue: Any, graph_id: str,
    source_id: str,        # The Source
    skeleton_hash: str,    # The live spine
    index: int,            # The neighbour's index
) -> Optional[SpineSegment]:  # The live segment at that index (None at the spine head)
    """The live segment right before the chunk — the flank a donor that starts before
    the new first segment anchors on (plan_transfer_rows's 'unanchored' otherwise)."""
    if index < 0:
        return None
    q = NodeQuery(label=TranscriptGraphLabels.SEGMENT, project=list(_SPINE_PROJECTION),
                  where=[PropertyPredicate("source_id", "eq", source_id),
                         PropertyPredicate("skeleton_hash", "eq", skeleton_hash),
                         PropertyPredicate("index", "eq", int(index)),
                         PropertyPredicate(SEGMENT_SUPERSEDED_BY_PROP, "is_null")])
    res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
    rows = list(res.rows or [])
    return _row_to_spine_segment(rows[0]) if rows else None


async def _segment_props(queue: Any, graph_id: str, segment_id: str) -> Dict[str, Any]:
    q = NodeQuery(ids=[segment_id], project=["skeleton_hash", "source_id", "rendition_id", "index"])
    res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
    rows = list(res.rows or [])
    return dict(rows[0]) if rows else {}


# ---- the handler -------------------------------------------------------------------------------

async def chunk_respine_transfer(
    queue: Any,                     # Started job queue (decomp-core's open stack)
    graph_id: str,                  # Graph-storage capability id
    *,
    source_id: str,                 # The Source
    old_segment_ids: List[str],     # The chunk's old (about-to-be-superseded) segments, index order
    new_segment_ids: List[str],     # The replacement segments, index order
    journal_path: Optional[str],    # Sidecar journal
    actor: str,                     # Who (the operator running the respine)
    run_id: Optional[str] = None,   # The decomp respine run id (rides the proposal snapshots)
    tolerance: float = 0.05,        # Event dup-guard window (s)
) -> Dict[str, Any]:  # {"session_id","carried","straddles","uncovered","events","speakers","dups","unanchored","word_bearing"}
    """The registered chunk-scoped transfer (the entry point decomp-core discovers).
    Reads the chunk's ACTIVE overlay, plans both classes purely, commits everything
    under ONE CorrectionSession (purpose chunk-respine-transfer), returns the ids the
    respine's fact op records as `carried` + the straddles it lists."""
    # Local import: cli.py imports graph.py; the planners live beside the CLI verb.
    from cjm_transcript_correction_core.cli import plan_transfer_rows, wordless_donors

    old_segs = await load_segments_by_ids(queue, graph_id, list(old_segment_ids))
    new_segs = await load_segments_by_ids(queue, graph_id, list(new_segment_ids))
    if not old_segs or not new_segs:
        raise ValueError(f"chunk transfer needs both sides on the graph (old {len(old_segs)}, new {len(new_segs)})")
    corrections, superseded = await load_source_corrections(queue, graph_id, source_id)
    active = [c for c in corrections if c["id"] not in superseded and c.get("status") != "proposed"]
    insert_meta = {c["id"]: (c.get("payload") or {}) for c in active
                   if c.get("correction_type") == "insertion"
                   and (c.get("payload") or {}).get("operation") == "chunk_insert"}
    eff_old = project_effective_spine(old_segs, active)

    # Events: the chunk's effective wordless layer, re-anchored by time onto the new run
    # (+ the live neighbour before the chunk, so a donor cut ahead of the new first
    # segment still anchors instead of reading as unanchored).
    donors, word_bearing = wordless_donors(eff_old, insert_meta)
    to_l0 = [s for s in new_segs if s.start_time is not None]
    first_new = min((s.index for s in new_segs), default=0)
    props = await _segment_props(queue, graph_id, new_segs[0].id)
    prev = None
    if props.get("skeleton_hash"):
        prev = await live_neighbour(queue, graph_id, source_id, str(props["skeleton_hash"]), first_new - 1)
    anchors = ([prev] if prev is not None and prev.id not in set(old_segment_ids) else []) + to_l0
    plan, dups, unanchored = plan_transfer_rows(donors, [], anchors, tolerance)

    # Speakers: the per-span map from the old segments' EFFECTIVE spans -> the new extents.
    assignments = active_speaker_assignments(corrections, superseded)
    spans = speaker_spans([u for u in eff_old if u.id in set(old_segment_ids)], assignments)
    rows, straddles, uncovered = plan_speaker_carry(spans, new_segs)
    turns = group_speaker_rows(rows)

    if not plan and not turns:
        return {"session_id": None, "carried": [], "straddles": straddles, "uncovered": uncovered,
                "events": 0, "speakers": 0, "dups": dups, "unanchored": unanchored,
                "word_bearing": word_bearing}
    sess = await start_session(queue, graph_id, [source_id], journal_path=journal_path,
                               purpose=TRANSFER_PURPOSE, actor=actor)
    carried: List[str] = []
    for p in plan:
        cid = await commit_chunk_insert_correction(
            queue, graph_id, source_id, p["after_id"], p["start"], p["end"], sess.id,
            before_segment_id=p["before_id"], label=p["label"], rank=p["rank"],
            actor=actor, journal_path=journal_path)
        carried.append(cid)
    for t in turns:
        cid = await commit_speaker_assign_correction(
            queue, graph_id, source_id, t["segment_ids"], t["entity_id"], sess.id,
            verdict=t["verdict"],
            proposal={"transfer": "chunk-respine", "run_id": run_id, "donor_correction_ids": t["donors"]},
            actor=actor, journal_path=journal_path)
        carried.append(cid)
    await set_session_status(queue, graph_id, sess.id, "completed", journal_path=journal_path, actor=actor)
    logger.info(f"chunk-respine transfer: {len(plan)} event insert(s) + {len(turns)} speaker turn(s) "
                f"carried onto {len(new_segs)} new segment(s); {len(straddles)} straddle(s), "
                f"{len(uncovered)} uncovered, {dups} dup(s), {unanchored} unanchored")
    return {"session_id": sess.id, "carried": carried, "straddles": straddles, "uncovered": uncovered,
            "events": len(plan), "speakers": len(turns), "dups": dups, "unanchored": unanchored,
            "word_bearing": word_bearing}
