"""The correction overlay's graph I/O: targeted (scale-shaped) reads of a committed spine via the graph-storage query action, construction of Correction / CorrectionSession nodes + CORRECTS / SUPERSEDES / DERIVED_FROM / REVIEWED edges, the in-core effective-spine projection (layer-0 + applied corrections), and commit through the job queue. Hand-rolled (revolution-1) = direct CR-18 spec material; append-only on layer-0 (never update/delete a Segment)."""

import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from cjm_context_graph_layer.edits import (project_effective_spine as layer_project_effective_spine,
                                           resolve_active, SpineEdit, SpineUnit)
from cjm_context_graph_layer.grammar import make_edge, OverlayRelations, SpineRelations
from cjm_context_graph_layer.ops import extend_graph, graph_task
from cjm_context_graph_primitives.graph import GraphNode
from cjm_context_graph_primitives.locators import locator_from_dict
from cjm_context_graph_primitives.query import (EdgeQuery, EdgeQueryResult, NodeQuery,
                                                NodeQueryResult, OrderBy, PropertyPredicate,
                                                RelationPredicate, SourcePredicate)
from cjm_substrate.core.queue import JobQueue, JobStatus
from cjm_transcript_correction_core.journal import journal_correction_op, segment_anchor
from cjm_transcript_correction_core.models import (Correction, CorrectionRelations,
                                                   CorrectionSession, SpineSegment)
from cjm_transcript_graph_schema.schema import TranscriptGraphLabels

# Stage 4: the typed query surface — importing the result classes IS the
# host-side wire registration (F8); the tuple keeps these SIDE-EFFECT imports
# referenced so the canonical emit cannot prune them. Stage 5 (CR-18): the
# layer owns the shared plumbing + the effective-view machinery this core
# hand-rolled at revolution 1 (C5/C11/C16 migrated).
_REGISTERED_WIRE_KINDS = (NodeQueryResult, EdgeQueryResult)


async def submit_and_wait(
    queue: JobQueue,                  # Started job queue
    instance_id: str,                 # Capability instance to invoke
    *,
    timeout: Optional[float] = None,  # Seconds to wait; None = no limit
    **kwargs,                         # Forwarded to the capability action
) -> Any:  # Completed job result payload
    """Submit one capability job, wait for it, return its result (raise on failure).

    (Restored as its own cell after the stage-2 field_of retirement removed
    the shared cell that bundled both functions — the loop-back harness
    caught the casualty; one-fn-per-cell prevents the recurrence.)
    """
    job_id = await queue.submit(instance_id, **kwargs)
    job = await queue.wait_for_job(job_id, timeout=timeout)
    if job.status != JobStatus.completed:
        raise RuntimeError(f"{instance_id} job {job_id} {job.status}: {job.error}")
    return job.result


# Targeted, scale-shaped reads of one SOURCE's fine spine. The fine spine hangs
# under an AudioRendition (raw | vocals | ...) under an AudioSegment under the
# Source, so scoping is: coarse read (AudioSegments) -> rendition resolve (which
# rendition's spine) -> batched far-end constraint over the rendition ids.

_SPINE_PROJECTION = ["index", "text", "start_time", "end_time", "text_from", "sources"]  # + structural "id"

LEGACY_SKELETON = "legacy"  # --skeleton selector naming the pre-split spine (segments without a skeleton_hash prop)


def _row_to_spine_segment(row: Dict[str, Any]) -> SpineSegment:  # One projected row -> SpineSegment
    """Build a SpineSegment from a projected row (audio anchor + per-transcriber slices)."""
    sources = row.get("sources") or []
    text_from = row.get("text_from")
    audio = next((s for s in sources if (s.get("slice") or {}).get("kind") == "time"), None)
    slices: List[Dict[str, Any]] = []
    auth_hash = None
    for s in sources:
        sl = s.get("slice") or {}
        if sl.get("kind") != "char":
            continue
        tid = (s.get("locator") or {}).get("node_id")
        slices.append({"transcript": tid, "start": sl.get("start"), "end": sl.get("end"),
                       "content_hash": s.get("content_hash")})
        if tid and tid == text_from:
            auth_hash = s.get("content_hash")
    idx = row.get("index")
    return SpineSegment(
        id=row["id"], index=int(idx) if idx is not None else -1,
        text=row.get("text") or "", start_time=row.get("start_time"),
        end_time=row.get("end_time"),
        source_locator=(str(locator_from_dict(audio["locator"])) if audio and audio.get("locator") else None),
        content_hash=auth_hash, text_from=text_from, text_slices=slices,
    )


async def source_audio_segment_ids(
    queue: JobQueue,  # Started job queue
    graph_id: str,    # Graph-storage capability id
    source_id: str,   # Source node id
) -> List[str]:  # Ordered AudioSegment node ids under the Source
    """The Source's coarse spine (one small typed read; ordered by index)."""
    q = NodeQuery(label=TranscriptGraphLabels.AUDIO_SEGMENT,
                  related=RelationPredicate(SpineRelations.PART_OF, node_id=source_id),
                  order_by=OrderBy(prop="index"), project=["index"])
    res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
    return [r["id"] for r in (res.rows or [])]


def _rendition_label(r: Dict[str, Any]) -> str:  # Human-readable rendition label
    """A rendition's selector/display label ("raw" or its preprocessing chain)."""
    if r.get("is_raw"):
        return "raw"
    return str(r.get("preprocessing") or "|".join(r.get("chain") or []) or "raw")


async def resolve_source_renditions(
    queue: JobQueue,                   # Started job queue
    graph_id: str,                     # Graph-storage capability id
    source_id: str,                    # Source node id
    selector: Optional[str] = None,    # Which rendition: "raw" | a preprocessing-descriptor substring | None = auto
) -> List[str]:  # The AudioRendition ids whose fine spine to operate on (one chain group)
    """Pick the AudioRendition set whose fine Segment spine correction operates on.

    The fine spine hangs under renditions (raw | vocals | ...) that COEXIST under
    one AudioSegment. With ONE decomposed rendition (the common case) it is
    selected automatically; multiple decomposed renditions REQUIRE an explicit
    `selector` ("raw", or a substring of the preprocessing descriptor) — the
    spines are never silently mixed. Returns the rendition ids of the chosen
    chain group (one per AudioSegment), or [] when nothing is decomposed yet.
    """
    aseg_ids = await source_audio_segment_ids(queue, graph_id, source_id)
    if not aseg_ids:
        return []
    rq = NodeQuery(label=TranscriptGraphLabels.AUDIO_RENDITION,
                   related=RelationPredicate(OverlayRelations.DERIVED_FROM, node_ids=aseg_ids),
                   project=["chain", "is_raw", "preprocessing"])
    rres = await graph_task(queue, graph_id, "query_nodes", query=rq.to_dict())
    rends = [{"id": r["id"], "chain": tuple(r.get("chain") or []),
              "is_raw": bool(r.get("is_raw")), "preprocessing": r.get("preprocessing")}
             for r in (rres.rows or [])]
    if not rends:
        return []
    by_chain: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rends:
        by_chain.setdefault(r["chain"], []).append(r)
    labels = sorted({_rendition_label(rs[0]) for rs in by_chain.values()})

    if selector is not None:
        sel = selector.strip().lower()
        for rs in by_chain.values():
            if (sel == "raw" and rs[0]["is_raw"]) or \
               (rs[0].get("preprocessing") and sel in str(rs[0]["preprocessing"]).lower()):
                return [r["id"] for r in rs]
        raise ValueError(f"--rendition {selector!r} matches no rendition for source "
                         f"{source_id} (available: {labels})")

    # Auto: the chain group that actually carries Segments.
    populated = await _populated_rendition_ids(queue, graph_id, [r["id"] for r in rends])
    populated_chains = {r["chain"] for r in rends if r["id"] in populated}
    if not populated_chains:
        return []
    if len(populated_chains) == 1:
        ck = next(iter(populated_chains))
        return [r["id"] for r in by_chain[ck]]
    pop_labels = sorted({_rendition_label(by_chain[c][0]) for c in populated_chains})
    raise ValueError(f"source {source_id} has multiple decomposed renditions ({pop_labels}); "
                     "pass --rendition to choose which spine to correct")


async def _populated_rendition_ids(
    queue: JobQueue,          # Started job queue
    graph_id: str,            # Graph-storage capability id
    rendition_ids: List[str], # Candidate rendition ids
) -> set:  # Subset that own >=1 fine Segment
    """Which renditions actually carry a fine Segment spine (one batched read)."""
    if not rendition_ids:
        return set()
    q = NodeQuery(label=TranscriptGraphLabels.SEGMENT,
                  related=RelationPredicate(SpineRelations.PART_OF, node_ids=list(rendition_ids)),
                  project=["rendition_id"])
    res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
    return {r.get("rendition_id") for r in (res.rows or []) if r.get("rendition_id")}


def _spine_query(
    rendition_ids: List[str],  # The AudioRendition ids the spine hangs under
    **overrides,               # NodeQuery field overrides (where / count / limit / ...)
) -> NodeQuery:  # The source-spine read
    """Segments PART_OF the chosen AudioRenditions (batched far-end), ordered by index."""
    base: Dict[str, Any] = dict(
        label=TranscriptGraphLabels.SEGMENT,
        related=RelationPredicate(SpineRelations.PART_OF, node_ids=list(rendition_ids)),
        order_by=OrderBy(prop="index"),
        project=list(_SPINE_PROJECTION),
    )
    base.update(overrides)
    return NodeQuery(**base)


async def load_source_segments(
    queue: JobQueue,                        # Started job queue
    graph_id: str,                          # Graph-storage capability id
    source_id: str,                         # Source node id
    limit: Optional[int] = None,            # Optional page size
    offset: Optional[int] = None,           # Optional page offset
    rendition_selector: Optional[str] = None,  # Which rendition spine (None = auto-select)
    skeleton_selector: Optional[str] = None,   # Which SKELETON spine ("legacy" | hash prefix); None = auto, refuses when >1 coexist
) -> List[SpineSegment]:  # Ordered spine segments (by index)
    """Load a Source's fine Segment spine under its chosen rendition + skeleton (typed query surface)."""
    rendition_ids = await resolve_source_renditions(queue, graph_id, source_id, rendition_selector)
    if not rendition_ids:
        return []
    where = await _resolve_spine_where(queue, graph_id, rendition_ids, skeleton_selector)
    q = _spine_query(rendition_ids, limit=limit, offset=int(offset or 0), where=where)
    res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
    return [_row_to_spine_segment(r) for r in (res.rows or [])]


async def load_empty_segments(
    queue: JobQueue,  # Started job queue
    graph_id: str,    # Graph-storage capability id
    source_id: str,   # Source node id
    rendition_selector: Optional[str] = None,  # Which rendition spine (None = auto-select)
    skeleton_selector: Optional[str] = None,   # Which SKELETON spine ("legacy" | hash prefix); None = auto, refuses when >1 coexist
) -> List[SpineSegment]:  # Only the empty-text segments (server-side filtered)
    """Load ONLY a Source's empty-text segments under its chosen rendition (D14 prune).

    The evidenced OR case (text = '' OR text IS NULL) = TWO server-side-filtered
    queries unioned client-side — compound boolean predicates stay deferred (P8);
    both halves materialize ~10% of the spine, never the whole source.
    """
    rendition_ids = await resolve_source_renditions(queue, graph_id, source_id, rendition_selector)
    if not rendition_ids:
        return []
    spine = await _resolve_spine_where(queue, graph_id, rendition_ids, skeleton_selector)
    r1 = await graph_task(queue, graph_id, "query_nodes", query=_spine_query(
        rendition_ids, where=spine + [PropertyPredicate("text", "eq", "")]).to_dict())
    r2 = await graph_task(queue, graph_id, "query_nodes", query=_spine_query(
        rendition_ids, where=spine + [PropertyPredicate("text", "is_null")]).to_dict())
    segs = [_row_to_spine_segment(r) for r in (r1.rows or []) + (r2.rows or [])]
    segs.sort(key=lambda s: s.index)
    return segs


async def count_source_segments(
    queue: JobQueue,  # Started job queue
    graph_id: str,    # Graph-storage capability id
    source_id: str,   # Source node id
    rendition_selector: Optional[str] = None,  # Which rendition spine (None = auto-select)
    skeleton_selector: Optional[str] = None,   # Which SKELETON spine ("legacy" | hash prefix); None = auto, refuses when >1 coexist
) -> int:  # Number of fine Segment nodes under the Source's chosen rendition + skeleton
    """Count a Source's segments server-side under its chosen rendition + skeleton (typed count mode)."""
    rendition_ids = await resolve_source_renditions(queue, graph_id, source_id, rendition_selector)
    if not rendition_ids:
        return 0
    where = await _resolve_spine_where(queue, graph_id, rendition_ids, skeleton_selector)
    q = _spine_query(rendition_ids, order_by=None, project=None, count=True, where=where)
    res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
    return int(res.count or 0)


def _edge(
    source_id: str,                            # Origin node id
    target_id: str,                            # Destination node id
    relation_type: str,                        # Edge relation type
    properties: Optional[Dict[str, Any]] = None,  # Edge properties
) -> Dict[str, Any]:  # Edge wire dict
    """Build an edge wire dict with a GENERATED id.

    Used for REVIEWED markers only: review decisions are EVENTS (a re-decision
    appends), so their edges keep generated ids. Structural overlay edges
    (CORRECTS / SUPERSEDES / DERIVED_FROM) use the layer's deterministic
    `make_edge` — unique per (source, target, relation) by construction.
    """
    return {"id": str(uuid4()), "source_id": source_id, "target_id": target_id,
            "relation_type": relation_type, "properties": properties or {}}


def build_correction_node(
    correction_type: str,                  # "text_content" | "punctuation" | "grouping" | "review" | "mark" | "timing" | "insertion"
    session_id: str,                       # Owning session id
    payload: Dict[str, Any],               # Type-specific payload
    actor: str = "human",                  # Actor
    status: str = "applied",               # Lifecycle status
    canonical_form: Optional[str] = None,  # Optional entity key
    rationale: Optional[str] = None,       # Optional note
) -> Correction:  # The Correction overlay node (not yet committed)
    """Construct a Correction overlay node (pure; commit happens separately)."""
    return Correction(
        correction_type=correction_type, session_id=session_id, payload=payload,
        actor=actor, status=status, canonical_form=canonical_form, rationale=rationale,
    )


def build_prune_correction(
    source_id: str,              # Source being corrected
    pruned: List[SpineSegment],  # Empty layer-0 segments to prune
    session_id: str,             # Owning session id
    actor: str = "human",        # Actor
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:  # (correction node dict, edge dicts)
    """Build one batch grouping Correction that prunes empty segments (D14).

    Non-destructive: layer-0 nodes are NOT deleted; the Correction records the
    pruned ids and DERIVED_FROM edges point at each pruned Segment. The effective
    spine drops them at projection time (reversible by superseding this node).
    The payload carries the layer's `prune` spine-edit op vocabulary.
    """
    payload = {
        "operation": "prune_empty",
        "source_id": source_id,
        "pruned_segment_ids": [s.id for s in pruned],
        "pruned_count": len(pruned),
    }
    node = build_correction_node("grouping", session_id, payload, actor=actor).to_graph_node()
    edges = [make_edge(node.id, s.id, CorrectionRelations.DERIVED_FROM) for s in pruned]
    return node.to_dict(), edges


async def commit_nodes_edges(
    queue: JobQueue,              # Started job queue
    graph_id: str,                # Graph-storage capability id
    nodes: List[Dict[str, Any]],  # Node wire dicts
    edges: List[Dict[str, Any]],  # Edge wire dicts
) -> Dict[str, int]:  # {"nodes": n, "edges": m} created counts
    """Commit overlay nodes/edges through the layer's idempotent extend_graph.

    Stage 5 (C5 plumbing migrated): the layer owns emit-if-absent +
    verify-if-present; overlay nodes have generated ids so they always add,
    but a replayed commit collides into a verified no-op instead of
    duplicating structural edges.
    """
    res = await extend_graph(queue, graph_id, nodes, edges)
    return {"nodes": res.nodes_added, "edges": res.edges_added}


async def start_session(
    queue: JobQueue,   # Started job queue
    graph_id: str,     # Graph-storage capability id
    scope: List[str],  # Source node ids in scope
    journal_path: Optional[str] = None,  # Sidecar journal — append the op on success (None = unjournaled)
) -> CorrectionSession:  # The committed CorrectionSession
    """Create + commit a new CorrectionSession node."""
    sess = CorrectionSession(scope=scope)
    node = sess.to_graph_node().to_dict()
    await commit_nodes_edges(queue, graph_id, [node], [])
    if journal_path:
        journal_correction_op(journal_path, "session-start", actor="human",
                              session_id=node["id"], args={"scope": scope},
                              nodes=[node], edges=[], op_id=node["id"])
    return sess


async def get_session(
    queue: JobQueue,  # Started job queue
    graph_id: str,    # Graph-storage capability id
    session_id: str,  # CorrectionSession node id
) -> Optional[Dict[str, Any]]:  # The session node dict, or None
    """Fetch a CorrectionSession node by id (resume/reopen) — typed get, dict shape preserved."""
    node = await graph_task(queue, graph_id, "get_node", node_id=session_id)
    if node is None:
        return None
    return node.to_dict() if isinstance(node, GraphNode) else node


async def set_session_status(
    queue: JobQueue,  # Started job queue
    graph_id: str,    # Graph-storage capability id
    session_id: str,  # CorrectionSession node id
    status: str,      # New status ("in_progress" | "completed" | "reopened")
    journal_path: Optional[str] = None,  # Sidecar journal — append the op on success (None = unjournaled)
) -> None:
    """Update a session's status + updated_at.

    The ONLY update_node use in this core: a CorrectionSession is OVERLAY metadata
    whose lifecycle is mutable. Layer-0 Segments + Corrections stay append-only.
    Its journal op replays via update_node (last-wins in append order), not wires.
    """
    ts = time.time()
    await graph_task(queue, graph_id, "update_node", node_id=session_id,
                     properties={"status": status, "updated_at": ts})
    if journal_path:
        journal_correction_op(journal_path, "session-status", actor="human",
                              session_id=session_id,
                              args={"session_id": session_id, "status": status,
                                    "updated_at": ts},
                              nodes=[], edges=[])


async def record_review_markers(
    queue: JobQueue,                   # Started job queue
    graph_id: str,                     # Graph-storage capability id
    session_id: str,                   # Owning session id
    decisions: List[Tuple[str, str]],  # (segment_id, decision) pairs
    journal_path: Optional[str] = None,  # Sidecar journal — append the op on success (None = unjournaled)
) -> int:  # Number of REVIEWED edges committed
    """Persist per-(session, segment) review markers as REVIEWED edges."""
    edges = [_edge(session_id, seg_id, CorrectionRelations.REVIEWED,
                   {"decision": dec, "ts": time.time()})
             for seg_id, dec in decisions]
    n = (await commit_nodes_edges(queue, graph_id, [], edges))["edges"]
    if journal_path:
        # No anchor: markers are derivative session state — the CORRECTION op carries
        # the anchor (DEC ccbab9f5 point 5 scopes anchors to correction ops), and the
        # extra segment reads priced the TUI keystroke (journal-lag finding).
        journal_correction_op(journal_path, "review-markers", actor="human",
                              session_id=session_id,
                              args={"decisions": [list(d) for d in decisions]},
                              nodes=[], edges=edges)
    return n


async def load_review_markers(
    queue: JobQueue,  # Started job queue
    graph_id: str,    # Graph-storage capability id
    session_id: str,  # Owning session id
) -> Dict[str, str]:  # segment_id -> decision for the session
    """Load a session's review markers (typed edge projection over REVIEWED edges).

    Markers are EVENTS (a re-decision appends — the _edge docstring's contract),
    so the read is LATEST-WINS by the marker's ts, and a latest decision of
    "unreviewed" means UNDECIDED: the segment drops out of the returned map
    entirely (the space-undo gesture; compute_worklist then re-surfaces it).
    Historical edges without ts sort first, so any re-decision beats them.
    """
    q = EdgeQuery(source_id=session_id, relation_type=CorrectionRelations.REVIEWED,
                  project=["decision", "ts"])
    res = await graph_task(queue, graph_id, "query_edges", query=q.to_dict())
    latest: Dict[str, str] = {}
    for r in sorted((res.rows or []), key=lambda r: float(r.get("ts") or 0.0)):
        latest[r["target_id"]] = r["decision"]
    return {sid: dec for sid, dec in latest.items() if dec != "unreviewed"}


def _node_to_correction_dict(
    node: Any,  # GraphNode (typed task result) or its wire dict
) -> Dict[str, Any]:  # Correction properties + "id" (the pre-stage-4 row shape)
    """Flatten a Correction node to its properties dict + id."""
    d = node.to_dict() if isinstance(node, GraphNode) else dict(node)
    props = dict(d.get("properties", {}) or {})
    props["id"] = d["id"]
    return props


async def find_corrections_for_session(
    queue: JobQueue,  # Started job queue
    graph_id: str,    # Graph-storage capability id
    session_id: str,  # Owning session id
) -> List[Dict[str, Any]]:  # Correction property dicts for the session
    """List corrections recorded in a session (typed property filter)."""
    q = NodeQuery(label="Correction",
                  where=[PropertyPredicate("session_id", "eq", session_id)])
    res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
    return [_node_to_correction_dict(n) for n in (res.nodes or [])]


async def find_prior_corrections_by_hash(
    queue: JobQueue,    # Started job queue
    graph_id: str,      # Graph-storage capability id
    content_hash: str,  # SourceRef content hash to look up
) -> List[Dict[str, Any]]:  # Corrections whose CORRECTS target carried this hash
    """Cross-transcript correction-cache lookup (targeted; the graph IS the lexicon).

    THE stage-4 promotion landing: the raw two-table JOIN this function carried
    (C-ledger site 6) became ONE typed far-end provenance constraint
    (`RelationPredicate.node_source`, content-hash-primary per CR-19) — the
    hottest review-path read is portable now.
    """
    q = NodeQuery(label="Correction",
                  related=RelationPredicate(
                      CorrectionRelations.CORRECTS,
                      node_source=SourcePredicate(content_hash=content_hash)))
    res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
    return [_node_to_correction_dict(n) for n in (res.nodes or [])]


def corrections_to_edits(
    corrections: List[Dict[str, Any]],  # ACTIVE correction property dicts
) -> List[SpineEdit]:  # The layer's neutral spine-edit operations
    """Map this core's Correction payloads onto the layer's spine-edit vocabulary.

    The DOMAIN interpretation (correction_type + payload shapes) stays here;
    the projection MECHANICS (ordering, latest-wins, prune/replace/boundary
    semantics) are the layer's (CR-18: C11 migrated).
    """
    edits: List[SpineEdit] = []
    for c in corrections:
        if c.get("status") in ("superseded", "proposed"):
            # superseded: legacy skip retained; proposed: awaiting a review
            # verdict — a proposal never enters the effective view (DEC 7861d27e).
            continue
        ctype = c.get("correction_type")
        payload = c.get("payload", {}) or {}
        created = float(c.get("created_at") or 0.0)
        if ctype == "grouping" and payload.get("operation") == "prune_empty":
            edits.append(SpineEdit(edit_id=c["id"], op="prune",
                                   targets=list(payload.get("pruned_segment_ids") or []),
                                   created_at=created))
        elif ctype == "grouping" and payload.get("operation") == "shift_boundary":
            left = payload.get("boundary_after")
            if left is not None:
                edits.append(SpineEdit(edit_id=c["id"], op="boundary_shift",
                                       targets=[t for t in (left, payload.get("right_segment_id")) if t],
                                       payload={"boundary_after": left,
                                                "text": payload.get("text", ""),
                                                "direction": payload.get("direction", "push")},
                                       created_at=created))
        elif ctype == "text_content" and payload.get("operation") == "replace_text":
            sid = payload.get("segment_id")
            if sid is not None:
                edits.append(SpineEdit(edit_id=c["id"], op="replace_text", targets=[sid],
                                       payload={"text": payload.get("new_text", "")},
                                       created_at=created))
    return edits


def project_effective_spine(
    segments: List[SpineSegment],       # Ordered layer-0 spine
    corrections: List[Dict[str, Any]],  # Applied correction property dicts
) -> List[SpineSegment]:  # The effective spine after applying corrections
    """Project the effective spine = layer-0 + applied corrections.

    Stage 5: the projection MECHANICS live in `cjm-context-graph-layer`
    (`project_effective_spine` over SpineUnits — prune, replace_text, and the
    reserved boundary_shift, with created_at ordering + latest-wins); this
    wrapper converts SpineSegments <-> SpineUnits and re-attaches the
    segment metadata by id.

    SPINE-SCOPED (DEC f1024568 migration v0 = none): corrections are loaded
    SOURCE-wide, but parallel skeletons share no segment ids — an edit anchored
    to another spine's ids is simply not part of THIS spine's overlay, so it is
    dropped here rather than crashing the layer's loud unknown-target
    validation (2026-07-22 drive: legacy boundary-shifts SpineEditError'd the
    sentence-split spine's open).
    """
    units = [SpineUnit(id=s.id, text=s.text) for s in segments]
    known = {u.id for u in units}
    edits = [e for e in corrections_to_edits(corrections)
             if all(t in known for t in e.targets)]
    out_units = layer_project_effective_spine(units, edits)
    by_id = {u.id: u.text for u in out_units}
    out: List[SpineSegment] = []
    for s in segments:
        if s.id not in by_id:
            continue
        if by_id[s.id] != s.text:
            s = SpineSegment(id=s.id, index=s.index, text=by_id[s.id],
                             start_time=s.start_time, end_time=s.end_time,
                             source_locator=s.source_locator, content_hash=s.content_hash,
                             text_from=s.text_from, text_slices=s.text_slices)
        out.append(s)
    # Structural + timing corrections compose AFTER the text projection
    # (core-side — the layer's edit vocabulary stays text-scoped): chunk
    # inserts synthesize first, so time nudges can grow a synthetic chunk's
    # edges (the zero-width insert+nudge isolation pattern, DEC 3d3fa2a8).
    return apply_time_nudges(apply_chunk_inserts(out, corrections), corrections)


def build_text_correction(
    source_id: str,                        # Source the segment belongs to
    segment_id: str,                       # Layer-0 Segment being corrected
    new_text: str,                         # Corrected text
    session_id: str,                       # Owning session id
    old_text: Optional[str] = None,        # Prior effective text (for the record)
    supersedes_id: Optional[str] = None,   # Prior Correction this one replaces (re-edit)
    actor: str = "human",                  # Actor
    canonical_form: Optional[str] = None,  # Optional entity key (cross-transcript matching)
    rationale: Optional[str] = None,       # Optional note
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:  # (correction node dict, edge dicts)
    """Build a text_content Correction + its CORRECTS (+ optional SUPERSEDES) edges.

    Non-destructive: the layer-0 Segment is unchanged; the correction carries the
    new text in its payload + a CORRECTS edge to the segment. A re-edit adds a
    SUPERSEDES edge (new -> prior) so supersession is graph-native + append-only
    (the prior Correction is never mutated; it is excluded from the effective view
    because it is a SUPERSEDES target — the C16 semantics, layer-resolved).
    """
    payload = {"operation": "replace_text", "source_id": source_id,
               "segment_id": segment_id, "new_text": new_text, "old_text": old_text}
    node = build_correction_node("text_content", session_id, payload, actor=actor,
                                 canonical_form=canonical_form, rationale=rationale).to_graph_node()
    edges = [make_edge(node.id, segment_id, CorrectionRelations.CORRECTS)]
    if supersedes_id:
        edges.append(make_edge(node.id, supersedes_id, CorrectionRelations.SUPERSEDES))
    return node.to_dict(), edges


async def commit_text_correction(
    queue: JobQueue,                       # Started job queue
    graph_id: str,                         # Graph-storage capability id
    source_id: str,                        # Source the segment belongs to
    segment_id: str,                       # Layer-0 Segment being corrected
    new_text: str,                         # Corrected text
    session_id: str,                       # Owning session id
    old_text: Optional[str] = None,        # Prior effective text
    supersedes_id: Optional[str] = None,   # Prior Correction to supersede (re-edit)
    actor: str = "human",                  # Actor
    canonical_form: Optional[str] = None,  # Optional entity key
    journal_path: Optional[str] = None,    # Sidecar journal — append the op on success (None = unjournaled)
) -> str:  # The new Correction node id
    """Commit a text_content correction (node + CORRECTS [+ SUPERSEDES]) + a REVIEWED marker."""
    node, edges = build_text_correction(
        source_id, segment_id, new_text, session_id, old_text=old_text,
        supersedes_id=supersedes_id, actor=actor, canonical_form=canonical_form)
    await commit_nodes_edges(queue, graph_id, [node], edges)
    if journal_path:
        journal_correction_op(journal_path, "text-correction", actor=actor,
                              session_id=session_id,
                              args={"source_id": source_id, "segment_id": segment_id,
                                    "new_text": new_text, "old_text": old_text,
                                    "supersedes_id": supersedes_id,
                                    "canonical_form": canonical_form},
                              anchor=await segment_anchor(queue, graph_id, [segment_id]),
                              nodes=[node], edges=edges, op_id=node["id"])
    await record_review_markers(queue, graph_id, session_id, [(segment_id, "corrected")],
                                journal_path=journal_path)
    return node["id"]


def active_corrections(
    corrections: List[Dict[str, Any]],  # Corrections (e.g. from load_source_corrections)
    superseded_ids: set,                # Ids that are SUPERSEDES targets
) -> List[Dict[str, Any]]:  # Only the effective (non-superseded) corrections
    """Filter to the effective correction set (the layer's resolve_active over a read superseded set)."""
    active_ids = resolve_active([c.get("id") for c in corrections],
                                [("", t) for t in superseded_ids])
    return [c for c in corrections if c.get("id") in active_ids]


async def _superseded_ids(
    queue: JobQueue,            # Started job queue
    graph_id: str,              # Graph-storage capability id
    correction_ids: List[str],  # Candidate correction ids
) -> set:  # Subset that are SUPERSEDES targets
    """Which of the given corrections are SUPERSEDES targets (typed id-list batch)."""
    if not correction_ids:
        return set()
    q = EdgeQuery(relation_type=CorrectionRelations.SUPERSEDES,
                  target_ids=list(correction_ids), project=[])
    res = await graph_task(queue, graph_id, "query_edges", query=q.to_dict())
    return {r["target_id"] for r in (res.rows or [])}


async def load_source_corrections(
    queue: JobQueue,  # Started job queue
    graph_id: str,    # Graph-storage capability id
    source_id: str,   # Source node id
) -> Tuple[List[Dict[str, Any]], set]:  # (all corrections for the source, superseded id set)
    """Load every Correction targeting a Source (across sessions) + the superseded-id set.

    Source-scoped (corrections carry payload.source_id — a dotted-path typed
    predicate) so the effective view is a property of the SOURCE, not one
    session — the persistence/resume/reopen requirement. Append-only:
    supersession is read from SUPERSEDES edges, never a status mutation.
    """
    q = NodeQuery(label="Correction",
                  where=[PropertyPredicate("payload.source_id", "eq", source_id)])
    res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
    corrections = [_node_to_correction_dict(n) for n in (res.nodes or [])]
    superseded = await _superseded_ids(queue, graph_id, [c["id"] for c in corrections])
    return corrections, superseded


async def find_active_text_corrections_batch(
    queue: JobQueue,         # Started job queue
    graph_id: str,           # Graph-storage capability id
    segment_ids: List[str],  # Segments to look up
) -> Dict[str, Dict[str, Any]]:  # segment_id -> active text correction
    """Active text corrections for MANY segments in TWO round-trips (C17).

    One far-end batch read (Corrections with CORRECTS edges into the id set;
    `RelationPredicate.node_ids`) + one superseded-set read — replacing the
    per-item lookup the review loop paid (a 1,275-item review would have been
    1,275 queries; now 2). Latest non-superseded correction per segment wins.
    """
    if not segment_ids:
        return {}
    q = NodeQuery(label="Correction",
                  where=[PropertyPredicate("correction_type", "eq", "text_content")],
                  related=RelationPredicate(CorrectionRelations.CORRECTS,
                                            node_ids=list(segment_ids)))
    res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
    cands = [_node_to_correction_dict(n) for n in (res.nodes or [])]
    superseded = await _superseded_ids(queue, graph_id, [c["id"] for c in cands])
    out: Dict[str, Dict[str, Any]] = {}
    for c in cands:
        if c["id"] in superseded:
            continue
        sid = (c.get("payload") or {}).get("segment_id")
        if sid is None:
            continue
        prev = out.get(sid)
        if prev is None or c.get("created_at", 0.0) > prev.get("created_at", 0.0):
            out[sid] = c
    return out


async def find_active_text_correction(
    queue: JobQueue,  # Started job queue
    graph_id: str,    # Graph-storage capability id
    segment_id: str,  # Segment to look up
) -> Optional[Dict[str, Any]]:  # The current non-superseded text correction, or None
    """Single-segment convenience over the batch read (cross-session; latest wins)."""
    found = await find_active_text_corrections_batch(queue, graph_id, [segment_id])
    return found.get(segment_id)


async def load_variant_texts(
    queue: JobQueue,               # Started job queue
    graph_id: str,                 # Graph-storage capability id
    segments: List[SpineSegment],  # Spine segments (with text_slices)
) -> Dict[str, Dict[str, str]]:  # segment_id -> {transcriber: chunk text}
    """Resolve per-transcriber chunk texts from the segments' CharSlice refs.

    Stage 5: the cross-transcriber diff is INTRA-GRAPH — text is stored once
    per transcriber at the coarse Transcript nodes; this fetches ALL referenced
    Transcript nodes in ONE batched typed read and slices their text
    client-side by each segment's char ranges. Replaces the retired
    second-decomp-manifest positional join (C4/C14's shared-skeleton model).
    """
    transcript_ids = sorted({ts["transcript"] for s in segments
                             for ts in s.text_slices if ts.get("transcript")})
    if not transcript_ids:
        return {}
    res = await graph_task(queue, graph_id, "query_nodes",
                           query=NodeQuery(ids=transcript_ids).to_dict())
    tnodes: Dict[str, Dict[str, Any]] = {}
    for n in (res.nodes or []):
        d = n.to_dict() if isinstance(n, GraphNode) else dict(n)
        tnodes[d["id"]] = d.get("properties", {}) or {}
    out: Dict[str, Dict[str, str]] = {}
    for s in segments:
        per_t: Dict[str, str] = {}
        for ts in s.text_slices:
            props = tnodes.get(ts.get("transcript"))
            if props is None or ts.get("start") is None or ts.get("end") is None:
                continue
            transcriber = str(props.get("transcriber") or ts["transcript"])
            per_t[transcriber] = str(props.get("text") or "")[int(ts["start"]):int(ts["end"])]
        if per_t:
            out[s.id] = per_t
    return out


def build_boundary_shift_correction(
    source_id: str,                        # Source the boundary belongs to
    left_segment_id: str,                  # Layer-0 Segment LEFT of the boundary (the layer's `boundary_after`)
    right_segment_id: str,                 # Layer-0 Segment RIGHT of the boundary (recorded; layer re-derives from order)
    text: str,                             # The exact text moved across the boundary (verbatim, whitespace included)
    direction: str,                        # "push" (left tail -> right head) | "pull" (the mirror)
    session_id: str,                       # Owning session id
    supersedes_id: Optional[str] = None,   # Prior Correction this one replaces (re-edit)
    actor: str = "human",                  # Actor
    rationale: Optional[str] = None,       # Optional note
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:  # (correction node dict, edge dicts)
    """Build a grouping Correction that moves text across one segment boundary.

    The FA-misassignment gesture (an ALIGNMENT error, not a transcription
    error): text sits on the wrong side of a boundary; 1:1 audio alignment is
    preserved (segment count and positions never change). The payload carries
    the layer's `boundary_shift` op vocabulary; the layer validates the moved
    text VERBATIM at projection time and fails loudly on mismatch. CORRECTS
    edges anchor BOTH boundary segments.
    """
    if direction not in ("push", "pull"):
        raise ValueError(f"boundary-shift direction must be 'push' or 'pull', got {direction!r}")
    payload = {"operation": "shift_boundary", "source_id": source_id,
               "boundary_after": left_segment_id, "right_segment_id": right_segment_id,
               "text": text, "direction": direction}
    node = build_correction_node("grouping", session_id, payload, actor=actor,
                                 rationale=rationale).to_graph_node()
    edges = [make_edge(node.id, left_segment_id, CorrectionRelations.CORRECTS),
             make_edge(node.id, right_segment_id, CorrectionRelations.CORRECTS)]
    if supersedes_id:
        edges.append(make_edge(node.id, supersedes_id, CorrectionRelations.SUPERSEDES))
    return node.to_dict(), edges


def build_reject_review(
    source_id: str,                   # Source the rejected correction belongs to (source-scoped loads)
    rejected_id: str,                 # The prior Correction (typically a proposal) being rejected
    session_id: str,                  # Owning session id
    actor: str = "human",             # Actor (the reviewer)
    rationale: Optional[str] = None,  # Optional note (why it was rejected)
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:  # (review node dict, edge dicts)
    """Build a review Correction that REJECTS a prior correction (reject-as-supersede).

    An append-only VERDICT: rejection is a SUPERSEDES edge from a small
    `review` node, never a status mutation — `resolve_active` then excludes
    the rejected correction with the machinery supersession already uses, and
    the verdict carries actor/rationale/timestamp. A review node maps to NO
    spine edit (`corrections_to_edits` has no arm for it) so it can never
    touch the effective view itself. The ACCEPT verdict is deferred until
    agents actually propose; it rides this same node shape.
    """
    payload = {"operation": "reject", "source_id": source_id, "rejected_id": rejected_id}
    node = build_correction_node("review", session_id, payload, actor=actor,
                                 rationale=rationale).to_graph_node()
    edges = [make_edge(node.id, rejected_id, CorrectionRelations.SUPERSEDES)]
    return node.to_dict(), edges


async def commit_boundary_shift_correction(
    queue: JobQueue,                       # Started job queue
    graph_id: str,                         # Graph-storage capability id
    source_id: str,                        # Source the boundary belongs to
    left_segment_id: str,                  # Segment LEFT of the boundary
    right_segment_id: str,                 # Segment RIGHT of the boundary
    text: str,                             # Verbatim text moved across the boundary
    direction: str,                        # "push" | "pull"
    session_id: str,                       # Owning session id
    supersedes_id: Optional[str] = None,   # Prior Correction to supersede (re-edit)
    actor: str = "human",                  # Actor
    journal_path: Optional[str] = None,    # Sidecar journal — append the op on success (None = unjournaled)
) -> str:  # The new Correction node id
    """Commit a boundary-shift correction (node + CORRECTS x2 [+ SUPERSEDES]) + REVIEWED markers on both segments."""
    node, edges = build_boundary_shift_correction(
        source_id, left_segment_id, right_segment_id, text, direction, session_id,
        supersedes_id=supersedes_id, actor=actor)
    await commit_nodes_edges(queue, graph_id, [node], edges)
    if journal_path:
        journal_correction_op(journal_path, "boundary-shift", actor=actor,
                              session_id=session_id,
                              args={"source_id": source_id, "left_segment_id": left_segment_id,
                                    "right_segment_id": right_segment_id, "text": text,
                                    "direction": direction, "supersedes_id": supersedes_id},
                              anchor=await segment_anchor(queue, graph_id,
                                                          [left_segment_id, right_segment_id]),
                              nodes=[node], edges=edges, op_id=node["id"])
    await record_review_markers(queue, graph_id, session_id,
                                [(left_segment_id, "corrected"), (right_segment_id, "corrected")],
                                journal_path=journal_path)
    return node["id"]


def build_prune_amendment(
    prior: Dict[str, Any],            # The ACTIVE prune Correction being amended (property dict + id)
    unprune_ids: List[str],           # Segment ids to REMOVE from the prune set (rescued positions)
    session_id: str,                  # Owning session id
    actor: str = "human",             # Actor
    rationale: Optional[str] = None,  # Optional note
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:  # (correction node dict, edge dicts)
    """Build a grouping Correction that supersedes a prune with a REDUCED set (unprune).

    The append-only inverse of build_prune_correction for single positions: a
    boundary shift that gives text to a pruned segment RESCUES it (the
    falsified-D14 class), and the segment must leave the prune set or the
    layer's projection drops it — WITH its new text — from the effective view.
    The amendment re-carries the prior payload minus the rescued ids plus a
    SUPERSEDES edge; DERIVED_FROM edges re-anchor the still-pruned ids.
    """
    prior_payload = dict(prior.get("payload") or {})
    remaining = [sid for sid in (prior_payload.get("pruned_segment_ids") or [])
                 if sid not in set(unprune_ids)]
    payload = {"operation": "prune_empty", "source_id": prior_payload.get("source_id"),
               "pruned_segment_ids": remaining, "pruned_count": len(remaining)}
    node = build_correction_node("grouping", session_id, payload, actor=actor,
                                 rationale=rationale).to_graph_node()
    edges = [make_edge(node.id, sid, CorrectionRelations.DERIVED_FROM) for sid in remaining]
    edges.append(make_edge(node.id, prior["id"], CorrectionRelations.SUPERSEDES))
    return node.to_dict(), edges


async def commit_prune_amendment(
    queue: JobQueue,            # Started job queue
    graph_id: str,              # Graph-storage capability id
    prior: Dict[str, Any],      # The ACTIVE prune Correction being amended
    unprune_ids: List[str],     # Segment ids rescued from the prune set
    session_id: str,            # Owning session id
    actor: str = "human",       # Actor
    journal_path: Optional[str] = None,  # Sidecar journal — append the op on success (None = unjournaled)
) -> Dict[str, Any]:  # The amended correction as a property dict + id (local-echo ready)
    """Commit an unprune amendment (node + DERIVED_FROM edges + SUPERSEDES)."""
    node, edges = build_prune_amendment(prior, unprune_ids, session_id, actor=actor)
    await commit_nodes_edges(queue, graph_id, [node], edges)
    if journal_path:
        journal_correction_op(journal_path, "prune-amendment", actor=actor,
                              session_id=session_id,
                              args={"prior_id": prior.get("id"), "unprune_ids": unprune_ids},
                              anchor=await segment_anchor(queue, graph_id, unprune_ids),
                              nodes=[node], edges=edges, op_id=node["id"])
    return _node_to_correction_dict(node)


def mark_anchor_segments(
    anchor: Dict[str, Any],  # Mark anchor: {"kind": "segment" | "boundary" | "span", ...}
) -> List[str]:  # The layer-0 Segment ids the anchor touches (CORRECTS targets)
    """Validate a mark anchor and list the Segment ids it touches.

    The three anchor shapes (DEC 2a231843): `segment` {segment_id} ·
    `boundary` {boundary_after, right_segment_id} (the boundary AFTER a
    segment — the shift gesture's coordinates) · `span` {segment_id,
    char_start, char_end, text_snapshot} (offsets into the segment's effective
    text AT MARK TIME; the snapshot is what re-anchoring trusts — see
    `reanchor_span` — the offsets are only its hint).
    """
    kind = anchor.get("kind")
    if kind == "segment":
        ids = [anchor.get("segment_id")]
    elif kind == "boundary":
        ids = [anchor.get("boundary_after"), anchor.get("right_segment_id")]
    elif kind == "span":
        ids = [anchor.get("segment_id")]
        if not anchor.get("text_snapshot") or anchor.get("char_start") is None \
                or anchor.get("char_end") is None:
            raise ValueError("span mark anchor needs char_start, char_end and a text_snapshot")
    else:
        raise ValueError(f"mark anchor kind must be 'segment' | 'boundary' | 'span', got {kind!r}")
    if any(not sid for sid in ids):
        raise ValueError(f"{kind} mark anchor is missing its segment id(s): {anchor!r}")
    return [str(sid) for sid in ids]


def build_mark_correction(
    source_id: str,                       # Source the marked segment(s) belong to
    anchor: Dict[str, Any],               # Anchor: segment | boundary | span (see mark_anchor_segments)
    mark_class: str,                      # Open-vocabulary class (RECOMMENDED_MARK_CLASSES is the slate)
    session_id: str,                      # Owning session id
    supersedes_id: Optional[str] = None,  # Prior mark this one replaces (re-mark)
    actor: str = "human",                 # Actor ("human" | "capability:<name>" for pass-2 assists)
    note: Optional[str] = None,           # Optional free-text note
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:  # (correction node dict, edge dicts)
    """Build a NON-MUTATING mark Correction (DEC 2a231843: routed attention).

    A mark is a placeholder correction: it records that a slot needs judgment
    (an omission-entangled boundary, a suspect entity, ...) WITHOUT resolving
    anything — `corrections_to_edits` has no arm for it, so it can never touch
    the effective view. It closes by SUPERSESSION: the real correction that
    discharges it passes `supersedes_id=<mark id>`, or a dismissal supersedes
    it (`commit_mark_dismissal`, riding the reject-review shape). CORRECTS
    edges anchor every touched segment so marks surface in segment-scoped
    reads.
    """
    mc = (mark_class or "").strip()
    if not mc:
        raise ValueError("mark_class must be a non-empty string")
    if not mc[:1].isalnum():
        raise ValueError(f"mark_class must start with a letter or digit, got {mark_class!r}"
                         " — punctuation-led tokens are reserved for gestures ('-' dismissal)")
    seg_ids = mark_anchor_segments(anchor)
    payload = {"operation": "mark", "source_id": source_id,
               "anchor": dict(anchor), "mark_class": mc}
    node = build_correction_node("mark", session_id, payload, actor=actor,
                                 rationale=note).to_graph_node()
    edges = [make_edge(node.id, sid, CorrectionRelations.CORRECTS) for sid in seg_ids]
    if supersedes_id:
        edges.append(make_edge(node.id, supersedes_id, CorrectionRelations.SUPERSEDES))
    return node.to_dict(), edges


async def commit_mark_correction(
    queue: JobQueue,                      # Started job queue
    graph_id: str,                        # Graph-storage capability id
    source_id: str,                       # Source the marked segment(s) belong to
    anchor: Dict[str, Any],               # Anchor: segment | boundary | span
    mark_class: str,                      # Open-vocabulary class
    session_id: str,                      # Owning session id
    supersedes_id: Optional[str] = None,  # Prior mark to supersede (re-mark)
    actor: str = "human",                 # Actor
    note: Optional[str] = None,           # Optional free-text note
    journal_path: Optional[str] = None,   # Sidecar journal — append the op on success (None = unjournaled)
) -> str:  # The new mark Correction node id
    """Commit a mark (node + CORRECTS per anchored segment [+ SUPERSEDES]).

    NO review marker: a mark is routed attention, not a review decision — the
    walked-past state stays exactly as the operator left it.
    """
    node, edges = build_mark_correction(source_id, anchor, mark_class, session_id,
                                        supersedes_id=supersedes_id, actor=actor, note=note)
    await commit_nodes_edges(queue, graph_id, [node], edges)
    if journal_path:
        journal_correction_op(journal_path, "mark", actor=actor, session_id=session_id,
                              args={"source_id": source_id, "anchor": dict(anchor),
                                    "mark_class": mark_class, "note": note,
                                    "supersedes_id": supersedes_id},
                              anchor=await segment_anchor(queue, graph_id,
                                                          mark_anchor_segments(anchor)),
                              nodes=[node], edges=edges, op_id=node["id"])
    return node["id"]


async def commit_mark_dismissal(
    queue: JobQueue,                     # Started job queue
    graph_id: str,                       # Graph-storage capability id
    source_id: str,                      # Source the mark belongs to
    mark_id: str,                        # The open mark being dismissed
    session_id: str,                     # Owning session id
    actor: str = "human",                # Actor (the dismisser)
    note: Optional[str] = None,          # Optional note (why dismissed)
    journal_path: Optional[str] = None,  # Sidecar journal — append the op on success (None = unjournaled)
) -> str:  # The review node id that superseded the mark
    """Dismiss an open mark WITHOUT a correction (reject-as-supersede).

    Rides the review verdict shape (`build_reject_review`): dismissal is a
    SUPERSEDES edge from a small review node — append-only, carrying
    actor/note/timestamp — and `open_marks` then excludes the mark. No anchor
    on the journal op: dismissal is derivative session state, like review
    markers (the ccbab9f5 anchor scope).
    """
    node, edges = build_reject_review(source_id, mark_id, session_id,
                                      actor=actor, rationale=note)
    await commit_nodes_edges(queue, graph_id, [node], edges)
    if journal_path:
        journal_correction_op(journal_path, "mark-dismiss", actor=actor,
                              session_id=session_id,
                              args={"source_id": source_id, "mark_id": mark_id, "note": note},
                              nodes=[node], edges=edges, op_id=node["id"])
    return node["id"]


def open_marks(
    corrections: List[Dict[str, Any]],  # Corrections (e.g. from load_source_corrections)
    superseded_ids: set,                # Ids that are SUPERSEDES targets
) -> List[Dict[str, Any]]:  # The OPEN marks (not yet discharged/dismissed), oldest first
    """Filter to the OPEN marks — the pass-2 worklist ('query open marks, walk them').

    A mark leaves this set the moment anything supersedes it (the discharging
    correction or a dismissal review) — resolution is read from SUPERSEDES
    edges, never a status mutation.
    """
    marks = [c for c in corrections
             if c.get("correction_type") == "mark"
             and (c.get("payload") or {}).get("operation") == "mark"
             and c.get("id") not in superseded_ids]
    marks.sort(key=lambda c: c.get("created_at") or 0.0)
    return marks


def reanchor_span(
    anchor: Dict[str, Any],  # A span mark anchor (char_start / char_end / text_snapshot)
    effective_text: str,     # The segment's CURRENT effective text
) -> Optional[Tuple[int, int]]:  # Current (char_start, char_end); None = degrade to segment level
    """Re-locate a span anchor in text that may have been edited since mark time.

    The SNAPSHOT is the anchor's truth; the recorded offsets are only its hint
    (DEC 2a231843): exact offsets verified first, then the snapshot occurrence
    NEAREST the recorded start, else None — the caller degrades the mark to
    segment level (which usually means the marked text was addressed).
    """
    snap = anchor.get("text_snapshot")
    if not snap:
        return None
    try:
        cs, ce = int(anchor.get("char_start")), int(anchor.get("char_end"))
    except (TypeError, ValueError):
        cs, ce = -1, -1
    if 0 <= cs <= ce <= len(effective_text) and effective_text[cs:ce] == snap:
        return cs, ce
    hits: List[int] = []
    at = effective_text.find(snap)
    while at != -1:
        hits.append(at)
        at = effective_text.find(snap, at + 1)
    if not hits:
        return None
    best = min(hits, key=lambda h: abs(h - cs) if cs >= 0 else h)
    return best, best + len(snap)


async def _list_spines(
    queue: JobQueue,           # Started job queue
    graph_id: str,             # Graph-storage capability id
    rendition_ids: List[str],  # The chosen rendition chain group
) -> List[Dict[str, Any]]:  # [{"skeleton_hash", "split_policy", "segments"}], legacy first
    """Group the fine Segments under a rendition set by SKELETON (parallel spines).

    One bounded projection (two props over the PART_OF far-end batch), grouped
    client-side — group-by aggregates stay deferred in the typed query surface.
    A None skeleton_hash = the pre-split LEGACY spine (nodes committed before
    the prop existed); it sorts first so pickers show it as the incumbent.
    """
    q = _spine_query(rendition_ids, order_by=None,
                     project=["skeleton_hash", "split_policy"])
    res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
    groups: Dict[Optional[str], Dict[str, Any]] = {}
    for r in (res.rows or []):
        key = r.get("skeleton_hash")
        g = groups.setdefault(key, {"skeleton_hash": key, "split_policy": None,
                                    "segments": 0})
        g["segments"] += 1
        if r.get("split_policy"):
            g["split_policy"] = r["split_policy"]
    return sorted(groups.values(),
                  key=lambda g: (g["skeleton_hash"] is not None,
                                 g["skeleton_hash"] or ""))


async def list_source_spines(
    queue: JobQueue,  # Started job queue
    graph_id: str,    # Graph-storage capability id
    source_id: str,   # Source node id
    rendition_selector: Optional[str] = None,  # Which rendition chain (None = auto)
) -> List[Dict[str, Any]]:  # [{"skeleton_hash", "split_policy", "segments"}], legacy first
    """The SPINES coexisting under a Source's chosen rendition (DEC f1024568).

    The picker/discovery surface: each row is one skeleton — the legacy
    (pre-split) spine has skeleton_hash None; sentence-split spines carry their
    composite hash + policy tag. Empty when nothing is decomposed yet.
    """
    rendition_ids = await resolve_source_renditions(queue, graph_id, source_id,
                                                    rendition_selector)
    if not rendition_ids:
        return []
    return await _list_spines(queue, graph_id, rendition_ids)


async def _resolve_spine_where(
    queue: JobQueue,                 # Started job queue
    graph_id: str,                   # Graph-storage capability id
    rendition_ids: List[str],        # The chosen rendition chain group
    selector: Optional[str] = None,  # "legacy" | a skeleton-hash (prefix ok) | None = auto
) -> List[PropertyPredicate]:  # where-predicates scoping the chosen spine ([] = sole spine, no filter)
    """Resolve which SKELETON spine reads operate on (the rendition pattern, one
    level down).

    Reads the observed spine set, then delegates the selector semantics to the
    pure `spine_where_for` (auto refuses when several coexist; "legacy" = the
    pre-split prop-absent spine; otherwise a unique hash/hex-tail prefix).
    """
    return spine_where_for(await _list_spines(queue, graph_id, rendition_ids), selector)


def spine_where_for(
    spines: List[Dict[str, Any]],    # _list_spines rows ({"skeleton_hash", "split_policy", "segments"})
    selector: Optional[str] = None,  # "legacy" | a skeleton-hash (prefix ok) | None = auto
) -> List[PropertyPredicate]:  # where-predicates scoping the chosen spine ([] = sole spine, no filter)
    """Resolve a skeleton selector against the observed spine set (pure).

    Auto (None): a sole spine needs no filter; MULTIPLE coexisting spines
    refuse loudly (unfiltered reads would MIX them — the f1024568 hazard) and
    name the choices. Explicit: "legacy" scopes to pre-split nodes (prop
    absent); anything else matches one skeleton hash (full value, or a
    case-insensitive prefix of the hash / its hex tail).
    """
    def _label(s: Dict[str, Any]) -> str:
        h = s["skeleton_hash"]
        tag = s.get("split_policy") or ("vad-only" if h else LEGACY_SKELETON)
        return f"{tag}:{h.split(':')[-1][:8]}" if h else tag

    if selector is None:
        if len(spines) <= 1:
            return []
        raise ValueError(
            f"{len(spines)} spines coexist under this rendition "
            f"({', '.join(_label(s) for s in spines)}); pass --skeleton "
            f"('{LEGACY_SKELETON}' or a hash prefix) to choose one")
    sel = selector.strip().lower()
    if sel == LEGACY_SKELETON:
        if not any(s["skeleton_hash"] is None for s in spines):
            raise ValueError(f"no {LEGACY_SKELETON} spine here "
                             f"(available: {[_label(s) for s in spines]})")
        return [PropertyPredicate("skeleton_hash", "is_null")]
    hits = [s for s in spines if s["skeleton_hash"] and
            (s["skeleton_hash"].lower().startswith(sel)
             or s["skeleton_hash"].split(":")[-1].lower().startswith(sel))]
    if len(hits) != 1:
        raise ValueError(f"--skeleton {selector!r} matches {len(hits)} spine(s) "
                         f"(available: {[_label(s) for s in spines]})")
    return [PropertyPredicate("skeleton_hash", "eq", hits[0]["skeleton_hash"])]


def build_time_nudge_correction(
    source_id: str,                        # Source the nudged boundary belongs to
    edits: List[Dict[str, Any]],           # 1-2 edge edits: {"segment_id", "edge": "start"|"end", "old_time", "new_time"}
    session_id: str,                       # Owning session id
    boundary_words: Optional[Dict[str, Any]] = None,  # {"left": str|None, "right": str|None} — the words at the boundary (flywheel context)
    step_s: Optional[float] = None,        # The press's signed step (seconds; the granularity record)
    actor: str = "human",                  # Actor
    rationale: Optional[str] = None,       # Optional note
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:  # (correction node dict, edge dicts)
    """Build a timing Correction that nudges segment boundary TIMES (node + CORRECTS edges).

    The 3f9948d6 surface: FA boundary imprecision (2e42a737 — mid-word cuts,
    numeral slivers) is corrected by ear, one edge at a time; a WELDED point
    cut moves both edges in ONE correction (atomic — the cut point is one
    decision). Payload keeps OLD and NEW absolute times per edge plus the
    boundary words, so (prediction, correction, context) finetuning pairs for
    VAD+FA derive straight from the journal (the hitl-correction-flywheel
    extension from VAD labels to FA word timings). Non-destructive: layer-0
    Segment times are untouched; the effective view applies nudges via
    apply_time_nudges (latest-wins per edge, created_at order).
    """
    payload = {"operation": "time_nudge", "source_id": source_id,
               "edits": [dict(e) for e in edits]}
    if boundary_words:
        payload["boundary_words"] = dict(boundary_words)
    if step_s is not None:
        payload["step_s"] = float(step_s)
    node = build_correction_node("timing", session_id, payload, actor=actor,
                                 rationale=rationale).to_graph_node()
    edges = [make_edge(node.id, str(e["segment_id"]), CorrectionRelations.CORRECTS)
             for e in edits]
    return node.to_dict(), edges


async def commit_time_nudge_correction(
    queue: JobQueue,                       # Started job queue
    graph_id: str,                         # Graph-storage capability id
    source_id: str,                        # Source the nudged boundary belongs to
    edits: List[Dict[str, Any]],           # 1-2 edge edits: {"segment_id", "edge", "old_time", "new_time"}
    session_id: str,                       # Owning session id
    boundary_words: Optional[Dict[str, Any]] = None,  # Words at the boundary (flywheel context)
    step_s: Optional[float] = None,        # The press's signed step (seconds)
    actor: str = "human",                  # Actor
    journal_path: Optional[str] = None,    # Sidecar journal — append the op on success (None = unjournaled)
) -> str:  # The new Correction node id
    """Commit a time-nudge correction (node + CORRECTS per touched segment).

    No review marker: a nudge is a boundary-time decision made mid-walk, not a
    verdict on the segment's text — the walk's ✎/✓ bookkeeping stays with text
    ops. The journal op carries the full payload (old/new per edge + boundary
    words + step), the flywheel's training-pair record."""
    node, edges = build_time_nudge_correction(
        source_id, edits, session_id, boundary_words=boundary_words,
        step_s=step_s, actor=actor)
    await commit_nodes_edges(queue, graph_id, [node], edges)
    if journal_path:
        journal_correction_op(journal_path, "time-nudge", actor=actor,
                              session_id=session_id,
                              args={"source_id": source_id, "edits": [dict(e) for e in edits],
                                    "boundary_words": dict(boundary_words or {}),
                                    "step_s": step_s},
                              anchor=await segment_anchor(
                                  queue, graph_id,
                                  [str(e["segment_id"]) for e in edits]),
                              nodes=[node], edges=edges, op_id=node["id"])
    return node["id"]


def apply_time_nudges(
    segments: List[SpineSegment],       # The (text-)projected effective spine
    corrections: List[Dict[str, Any]],  # ACTIVE correction property dicts
) -> List[SpineSegment]:  # Segments with nudged start/end times applied
    """Apply timing corrections onto segment times (latest-wins per edge).

    Time edits stay CORE-SIDE: the layer's SpineEdit vocabulary is text-scoped
    (prune / replace_text / boundary_shift), and a second consumer has not yet
    demanded a layer-level time op — so the wrapper composes this after the
    text projection. Nudges apply in created_at order and each edge keeps the
    LAST new_time (repeated presses chain; every press journals old/new, so
    the training-pair record is the CHAIN, the projection only its endpoint).
    Spine-scoped like text edits: edits anchored to another skeleton's ids are
    dropped, not errors."""
    nudges = [c for c in corrections
              if c.get("correction_type") == "timing"
              and (c.get("payload") or {}).get("operation") == "time_nudge"]
    if not nudges:
        return segments
    known = {s.id for s in segments}
    times: Dict[Tuple[str, str], float] = {}
    for c in sorted(nudges, key=lambda c: float(c.get("created_at") or 0.0)):
        for e in (c.get("payload") or {}).get("edits") or []:
            sid, edge = e.get("segment_id"), e.get("edge")
            if sid in known and edge in ("start", "end") and e.get("new_time") is not None:
                times[(sid, edge)] = float(e["new_time"])
    if not times:
        return segments
    out: List[SpineSegment] = []
    for s in segments:
        ns = times.get((s.id, "start"))
        ne = times.get((s.id, "end"))
        if ns is None and ne is None:
            out.append(s)
            continue
        out.append(SpineSegment(
            id=s.id, index=s.index, text=s.text,
            start_time=ns if ns is not None else s.start_time,
            end_time=ne if ne is not None else s.end_time,
            source_locator=s.source_locator, content_hash=s.content_hash,
            text_from=s.text_from, text_slices=s.text_slices))
    return out


def build_chunk_insert_correction(
    source_id: str,                       # Source the inserted chunk belongs to
    after_segment_id: str,                # Layer-0 Segment the insertion follows (placement anchor)
    start_time: float,                    # Inserted span start (source-coordinate seconds)
    end_time: float,                      # Inserted span end (== start_time = zero-width, grown by nudges)
    session_id: str,                      # Owning session id
    before_segment_id: Optional[str] = None,  # Layer-0 Segment right of the gap (None at spine tail)
    label: Optional[str] = None,          # Optional annotation class (open mark-class vocabulary)
    text: str = "",                       # Born-empty text (missed speech arrives by e-edit)
    actor: str = "human",                 # Actor
    rationale: Optional[str] = None,      # Optional note
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:  # (correction node dict, edge dicts)
    """Build an insertion Correction that adds a chunk the skeleton never cut (DEC 3d3fa2a8).

    OVERLAY-PROJECTED: layer-0 stays a pure function of config over audio — the
    chunk exists only as this Correction, synthesized into the effective spine
    by apply_chunk_inserts, so re-decomposition never collides with it. The
    Correction's OWN NODE ID doubles as the synthetic segment's id (the 1:1
    mapping): later corrections that target the inserted chunk (time nudges
    growing a zero-width insert, replace_text carrying missed speech) anchor
    THIS node — real edges to a real node. Born empty with an optional
    annotation label (inhale/um/throat-clear — the isolation pattern wants
    labels, not transcripts); undo = supersession (commit_chunk_insert_removal).
    CORRECTS edges anchor the flanking layer-0 segments.
    """
    if not after_segment_id:
        raise ValueError("chunk insert needs after_segment_id (the gap follows a segment)")
    if float(end_time) < float(start_time):
        raise ValueError(f"chunk insert span is negative ({start_time}..{end_time})")
    payload = {"operation": "chunk_insert", "source_id": source_id,
               "after_segment_id": after_segment_id,
               "before_segment_id": before_segment_id,
               "start_time": float(start_time), "end_time": float(end_time),
               "text": text}
    if label:
        payload["label"] = str(label)
    node = build_correction_node("insertion", session_id, payload, actor=actor,
                                 rationale=rationale).to_graph_node()
    edges = [make_edge(node.id, after_segment_id, CorrectionRelations.CORRECTS)]
    if before_segment_id:
        edges.append(make_edge(node.id, before_segment_id, CorrectionRelations.CORRECTS))
    return node.to_dict(), edges


async def commit_chunk_insert_correction(
    queue: JobQueue,                      # Started job queue
    graph_id: str,                        # Graph-storage capability id
    source_id: str,                       # Source the inserted chunk belongs to
    after_segment_id: str,                # Segment the insertion follows
    start_time: float,                    # Span start (source seconds)
    end_time: float,                      # Span end (== start = zero-width)
    session_id: str,                      # Owning session id
    before_segment_id: Optional[str] = None,  # Right flank (None at spine tail)
    label: Optional[str] = None,          # Optional annotation class
    actor: str = "human",                 # Actor
    journal_path: Optional[str] = None,   # Sidecar journal — append the op on success (None = unjournaled)
) -> str:  # The new insertion Correction node id (= the synthetic segment id)
    """Commit a chunk insertion (node + CORRECTS per flank).

    No review marker: the inserted chunk is born UNREVIEWED — judging its audio
    (and typing its text) is exactly the work that follows the insert. The
    journal op anchors the flanking layer-0 segments: run-independent
    coordinates for the gap the human filled — with the label, the flywheel's
    labeled-VAD-gold record (a span the skeleton missed, classified by ear).
    """
    node, edges = build_chunk_insert_correction(
        source_id, after_segment_id, start_time, end_time, session_id,
        before_segment_id=before_segment_id, label=label, actor=actor)
    await commit_nodes_edges(queue, graph_id, [node], edges)
    if journal_path:
        journal_correction_op(journal_path, "chunk-insert", actor=actor,
                              session_id=session_id,
                              args={"source_id": source_id,
                                    "after_segment_id": after_segment_id,
                                    "before_segment_id": before_segment_id,
                                    "start_time": float(start_time),
                                    "end_time": float(end_time), "label": label},
                              anchor=await segment_anchor(
                                  queue, graph_id,
                                  [sid for sid in (after_segment_id, before_segment_id) if sid]),
                              nodes=[node], edges=edges, op_id=node["id"])
    return node["id"]


async def commit_chunk_insert_removal(
    queue: JobQueue,                     # Started job queue
    graph_id: str,                       # Graph-storage capability id
    source_id: str,                      # Source the insertion belongs to
    insert_id: str,                      # The chunk_insert Correction being removed
    session_id: str,                     # Owning session id
    actor: str = "human",                # Actor
    note: Optional[str] = None,          # Optional note (why removed)
    journal_path: Optional[str] = None,  # Sidecar journal — append the op on success (None = unjournaled)
) -> str:  # The review node id that superseded the insertion
    """Remove an inserted chunk WITHOUT touching layer-0 (reject-as-supersede).

    Rides the review verdict shape (build_reject_review), like mark dismissal:
    a SUPERSEDES edge from a small review node excludes the insertion from the
    active set, and the synthetic segment leaves the effective view. Later
    corrections anchored to the synthetic id (nudges, text) become foreign-id
    edits — dropped by the spine-scoped projection, never errors. No anchor on
    the journal op: removal is a verdict on an overlay node, not a layer-0
    touch (the ccbab9f5 anchor scope).
    """
    node, edges = build_reject_review(source_id, insert_id, session_id,
                                      actor=actor, rationale=note)
    await commit_nodes_edges(queue, graph_id, [node], edges)
    if journal_path:
        journal_correction_op(journal_path, "chunk-insert-remove", actor=actor,
                              session_id=session_id,
                              args={"source_id": source_id, "insert_id": insert_id,
                                    "note": note},
                              nodes=[node], edges=edges, op_id=node["id"])
    return node["id"]


def apply_chunk_inserts(
    segments: List[SpineSegment],       # The (text-)projected effective spine
    corrections: List[Dict[str, Any]],  # ACTIVE correction property dicts
) -> List[SpineSegment]:  # Spine with synthetic inserted segments spliced in
    """Synthesize inserted chunks into the effective spine (DEC 3d3fa2a8).

    Insertion stays CORE-SIDE like time nudges — the layer's edit vocabulary is
    text-scoped and never sees synthetic ids. Each active chunk_insert becomes
    a SpineSegment whose id IS the Correction's node id (the 1:1 mapping),
    spliced after its after_segment_id anchor; when the left flank vanished
    from this projection (a pruned empty chunk), the before_segment_id flank
    places it instead. Several inserts in one gap order by (start_time,
    created_at). Text is latest-wins across the insert payload and any
    replace_text corrections targeting the synthetic id — the e-edit lane,
    invisible to the layer projection, which only knows layer-0 ids. Synthetic
    index = the flank's index: indexes are LAYER-0 coordinates and stay honest;
    the walk distinguishes inserts by id, not index. Spine-scoped: inserts
    anchored to another skeleton's ids drop, not error. Composes BEFORE
    apply_time_nudges so nudges can grow a zero-width insert's edges.
    """
    inserts = [c for c in corrections
               if c.get("correction_type") == "insertion"
               and c.get("status") not in ("superseded", "proposed")
               and (c.get("payload") or {}).get("operation") == "chunk_insert"]
    if not inserts:
        return segments
    insert_ids = {c["id"] for c in inserts}
    texts: Dict[str, Tuple[float, str]] = {}
    for c in corrections:
        p = c.get("payload") or {}
        if c.get("correction_type") == "text_content" \
                and p.get("operation") == "replace_text" \
                and p.get("segment_id") in insert_ids:
            created = float(c.get("created_at") or 0.0)
            prev = texts.get(p["segment_id"])
            if prev is None or created > prev[0]:
                texts[p["segment_id"]] = (created, p.get("new_text", ""))
    known = {s.id for s in segments}
    after_groups: Dict[str, List[Dict[str, Any]]] = {}
    before_groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in inserts:
        p = c.get("payload") or {}
        if p.get("after_segment_id") in known:
            after_groups.setdefault(p["after_segment_id"], []).append(c)
        elif p.get("before_segment_id") in known:
            before_groups.setdefault(p["before_segment_id"], []).append(c)
    if not after_groups and not before_groups:
        return segments

    def _key(c: Dict[str, Any]) -> Tuple[float, float]:
        p = c.get("payload") or {}
        return (float(p.get("start_time") or 0.0), float(c.get("created_at") or 0.0))

    def _synth(c: Dict[str, Any], index: int) -> SpineSegment:
        p = c.get("payload") or {}
        override = texts.get(c["id"])
        return SpineSegment(
            id=c["id"], index=index,
            text=override[1] if override else str(p.get("text") or ""),
            start_time=float(p["start_time"]) if p.get("start_time") is not None else None,
            end_time=float(p["end_time"]) if p.get("end_time") is not None else None)

    out: List[SpineSegment] = []
    for s in segments:
        for c in sorted(before_groups.get(s.id, ()), key=_key):
            out.append(_synth(c, s.index))
        out.append(s)
        for c in sorted(after_groups.get(s.id, ()), key=_key):
            out.append(_synth(c, s.index))
    return out
