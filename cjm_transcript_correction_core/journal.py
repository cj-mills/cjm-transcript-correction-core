"""Live append-through for the correction verbs — the workflow journal's domain half.

Every correction commit appends its op to the sidecar journal (DEC ccbab9f5): the
ENVELOPE (op id = minted Correction id, actor, set = session id, run-independent
segment ANCHOR) keeps each op a legible dataset row; the exact WIRES committed make
replay faithful to what WAS written (builder evolution can never drift a rebuild —
store the effect, not the recipe). Replay handlers register onto
`cjm_context_graph_layer.journal.replay_journal` (replay stays domain-owned); the
append discipline lives in `cjm_context_graph_primitives.journal`.
"""

from typing import Any, Dict, List, Optional

from cjm_context_graph_layer.ops import extend_graph, graph_task
from cjm_context_graph_primitives.journal import append_op
from cjm_context_graph_primitives.query import NodeQuery


async def segment_anchor(
    queue: Any,               # Started job queue
    graph_id: str,            # Graph-storage capability id
    segment_ids: List[str],   # The Segment ids the correction touches
) -> Dict[str, Any]:  # Run-independent anchor: source hashes + per-segment time span + verbatim text
    """The run-independent anchor stamped on every correction op (DEC ccbab9f5 point 5).

    Segment ids are THIS pipeline run's coordinates; (source content hash, time span,
    verbatim layer-0 text) survive any re-run — the coordinates the orphan detector's
    propose/confirm rebind (DEC point 6) matches against. ONE batched segments read
    per call; Source media hashes are immutable, so they read once per process."""
    res = await graph_task(queue, graph_id, "query_nodes",
                           query=NodeQuery(ids=list(segment_ids)).to_dict())
    segs = sorted(res.nodes or [], key=lambda n: n.properties.get("index") or 0)
    source_ids = sorted({s.properties.get("source_id") for s in segs
                         if s.properties.get("source_id")})
    missing = [sid for sid in source_ids if sid not in _SOURCE_HASH_CACHE]
    if missing:
        sres = await graph_task(queue, graph_id, "query_nodes",
                                query=NodeQuery(ids=missing).to_dict())
        for s in (sres.nodes or []):
            hashes = [r.content_hash for r in s.sources if getattr(r, "content_hash", None)]
            _SOURCE_HASH_CACHE[s.id] = hashes[0] if hashes else None
    src_hash = {sid: _SOURCE_HASH_CACHE.get(sid) for sid in source_ids}
    return {
        "sources": [{"id": sid, "content_hash": src_hash.get(sid)} for sid in source_ids],
        "segments": [{"id": s.id, "index": s.properties.get("index"),
                      "start": s.properties.get("start_time"),
                      "end": s.properties.get("end_time"),
                      "text": s.properties.get("text")} for s in segs],
    }


def journal_correction_op(
    journal_path: str,             # Sidecar journal path
    verb: str,                     # Domain verb ("boundary-shift" | "text-correction" | ...)
    actor: str,                    # Who decided ("human" | "agent:*" | "import:*")
    session_id: str,               # The op's SET identity (per-run CorrectionSession)
    args: Dict[str, Any],          # The semantic call args (the dataset row)
    nodes: List[Dict[str, Any]],   # The exact node wires committed
    edges: List[Dict[str, Any]],   # The exact edge wires committed
    anchor: Optional[Dict[str, Any]] = None,  # Run-independent segment anchor
    op_id: Optional[str] = None,   # Minted Correction node id (exact-once dedup lane)
) -> bool:  # True if appended
    """Append one correction op — envelope + semantic args + the EXACT wires committed.

    Wires make replay faithful to what WAS written (a builder change can never drift a
    rebuild); args + anchor keep the op a legible dataset row (the north-star pullable-
    datasets requirement). Appended AFTER the commit succeeds, mirroring the dev-graph
    append-on-success discipline. dedup=False: the append must stay UI-cheap (a dedup
    rescan over a genesis-scale journal cost 111ms/op — the TUI keystroke lag finding);
    id-carrying ops mint a fresh Correction uuid per commit so duplicates are
    structurally impossible live, and replay converges on collisions regardless."""
    op: Dict[str, Any] = {"verb": verb, "actor": actor, "set": session_id,
                          "args": args, "wires": {"nodes": nodes, "edges": edges}}
    if op_id:
        op["id"] = op_id
    if anchor:
        op["anchor"] = anchor
    return append_op(journal_path, op, dedup=False)


def correction_replay_handlers() -> Dict[str, Any]:  # verb -> async handler(queue, graph_id, op)
    """The correction core's registered replay vocabulary (replay stays DOMAIN-OWNED).

    Pass to `cjm_context_graph_layer.journal.replay_journal(handlers=...)`: wire-carrying
    ops re-apply through the layer's idempotent extend (replayed wires collide into
    verified no-ops); `session-status` is the core's only update_node write, replayed
    last-wins in append order."""
    async def _apply_wires(queue: Any, graph_id: str, op: Dict[str, Any]) -> None:
        w = op.get("wires") or {}
        await extend_graph(queue, graph_id, w.get("nodes") or [], w.get("edges") or [])

    async def _apply_session_status(queue: Any, graph_id: str, op: Dict[str, Any]) -> None:
        a = op["args"]
        await graph_task(queue, graph_id, "update_node", node_id=a["session_id"],
                         properties={"status": a["status"], "updated_at": a["updated_at"]})

    return {"session-start": _apply_wires, "boundary-shift": _apply_wires,
            "text-correction": _apply_wires, "prune-amendment": _apply_wires,
            "mark": _apply_wires, "mark-dismiss": _apply_wires,
            "review-markers": _apply_wires, "session-status": _apply_session_status}


# A Source's content hash is IMMUTABLE — read once per process, not per commit
# (the per-keystroke Source re-read priced the TUI; ids are uuids, so cross-db
# collisions within one process cannot occur).
_SOURCE_HASH_CACHE: Dict[str, Optional[str]] = {}
