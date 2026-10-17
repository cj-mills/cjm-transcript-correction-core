"""The correction verbs on the write order — the workflow journal's domain half.

Every correction commit journals its op to the sidecar journal, then applies it (DEC
ccbab9f5, ruling 6683f49e): the ENVELOPE (op id = minted Correction id, actor, set =
session id, run-independent segment ANCHOR) keeps each op a legible dataset row; the exact WIRES committed make
replay faithful to what WAS written (builder evolution can never drift a rebuild —
store the effect, not the recipe). Replay handlers register onto
`cjm_context_graph_layer.journal.replay_journal` (replay stays domain-owned); the
append discipline lives in `cjm_context_graph_primitives.journal`.
"""

from typing import Any, Dict, List, Optional

from cjm_context_graph_layer.journal import journal_op, wires_handlers, write_window
from cjm_context_graph_layer.ops import graph_task
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


async def journal_correction_op(
    queue: Any,                    # Started job queue
    graph_id: str,                 # Graph-storage capability id
    verb: str,                     # Domain verb ("boundary-shift" | "text-correction" | ...)
    actor: str,                    # Who decided ("human" | "agent:*" | "import:*")
    session_id: str,               # The op's SET identity (per-run CorrectionSession)
    args: Dict[str, Any],          # The semantic call args (the dataset row)
    nodes: List[Dict[str, Any]],   # The exact node wires to commit
    edges: List[Dict[str, Any]],   # The exact edge wires to commit
    anchor_segments: Optional[List[str]] = None,  # Segment ids the run-independent anchor is read from (None = no anchor)
    op_id: Optional[str] = None,   # Minted Correction node id (exact-once dedup lane)
    journal_path: Optional[str] = None,  # Sidecar journal (None refuses unless `unjournaled`)
    unjournaled: bool = False,     # Explicitly commit with no journal (scratch graphs, tests)
) -> Dict[str, Any]:  # The op as journaled (as applied, when unjournaled)
    """Journal one correction op, then apply it — envelope + semantic args + the EXACT wires.

    Wires make replay faithful to what WAS written (a builder change can never drift a
    rebuild); args + anchor keep the op a legible dataset row (the north-star pullable-
    datasets requirement). THE WRITE ORDER (ruling 6683f49e (1)): the op is APPENDED, then
    APPLIED through its verb's own replay handler (`apply_wires`, or the session / entity
    property merges) — the commit IS the apply, so live and rebuild run one code path — under
    the family's writer, which the verb's `write_window` (an app's gesture window, in turn) is
    joined by. The anchor is read inside that writer, on the healed db, and only when the op is
    journaled (it is the journal row's coordinate). No content dedup: id-carrying ops mint a
    fresh Correction uuid per commit, so duplicates are structurally impossible live, and the
    apply converges on collisions regardless."""
    async with write_window(queue, graph_id, journal_path, unjournaled=unjournaled, what=verb):
        op: Dict[str, Any] = {"verb": verb, "actor": actor, "set": session_id,
                              "args": args, "wires": {"nodes": nodes, "edges": edges}}
        if op_id:
            op["id"] = op_id
        if journal_path and anchor_segments is not None:
            op["anchor"] = await segment_anchor(queue, graph_id, list(anchor_segments))
        return await journal_op(queue, graph_id, op, journal_path, unjournaled=unjournaled)


def correction_replay_handlers() -> Dict[str, Any]:  # verb -> async handler(queue, graph_id, op)
    """The correction core's registered replay vocabulary (replay stays DOMAIN-OWNED).

    Exported through the `cjm_context_graph_layer.replay` entry-point group (DEC
    426658f1) and unioned by `composed_replay_handlers`: wire-carrying verbs register
    the layer's shared `apply_wires` — the single wires-replay authority, identity-
    comparable across cores, which keeps cross-core verb collisions checkable;
    `session-status` is the core's only update_node write, replayed last-wins in
    append order."""
    async def _apply_session_status(queue: Any, graph_id: str, op: Dict[str, Any]) -> None:
        a = op["args"]
        await graph_task(queue, graph_id, "update_node", node_id=a["session_id"],
                         properties={"status": a["status"], "updated_at": a["updated_at"]})

    async def _apply_session_purpose(queue: Any, graph_id: str, op: Dict[str, Any]) -> None:
        a = op["args"]
        await graph_task(queue, graph_id, "update_node", node_id=a["session_id"],
                         properties={"purpose": a["purpose"], "updated_at": a["updated_at"]})

    async def _apply_entity_rename(queue: Any, graph_id: str, op: Dict[str, Any]) -> None:
        a = op["args"]
        props: Dict[str, Any] = {"canonical_name": a["canonical_name"],
                                 "updated_at": a["updated_at"]}
        if a.get("provisional") is not None:
            props["provisional"] = a["provisional"]
        await graph_task(queue, graph_id, "update_node", node_id=a["entity_id"],
                         properties=props)

    handlers = wires_handlers("session-start", "boundary-shift", "text-correction",
                              "prune", "prune-amendment", "mark", "mark-dismiss", "mark-discharge",
                              "review-markers",
                              "time-nudge", "chunk-insert", "chunk-insert-remove",
                              "chunk-split", "chunk-split-remove",
                              "speech-overlay", "speech-overlay-remove",
                              "speaker-entity", "speaker-assign", "extraction-gate",
                              "stratum", "stratum-retract")
    handlers["session-status"] = _apply_session_status
    handlers["session-purpose"] = _apply_session_purpose
    handlers["entity-rename"] = _apply_entity_rename
    return handlers


# A Source's content hash is IMMUTABLE — read once per process, not per commit
# (the per-keystroke Source re-read priced the TUI; ids are uuids, so cross-db
# collisions within one process cannot occur).
_SOURCE_HASH_CACHE: Dict[str, Optional[str]] = {}
