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
) -> List[SpineSegment]:  # Ordered spine segments (by index)
    """Load a Source's fine Segment spine under its chosen rendition (typed query surface)."""
    rendition_ids = await resolve_source_renditions(queue, graph_id, source_id, rendition_selector)
    if not rendition_ids:
        return []
    q = _spine_query(rendition_ids, limit=limit, offset=int(offset or 0))
    res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
    return [_row_to_spine_segment(r) for r in (res.rows or [])]


async def load_empty_segments(
    queue: JobQueue,  # Started job queue
    graph_id: str,    # Graph-storage capability id
    source_id: str,   # Source node id
    rendition_selector: Optional[str] = None,  # Which rendition spine (None = auto-select)
) -> List[SpineSegment]:  # Only the empty-text segments (server-side filtered)
    """Load ONLY a Source's empty-text segments under its chosen rendition (D14 prune).

    The evidenced OR case (text = '' OR text IS NULL) = TWO server-side-filtered
    queries unioned client-side — compound boolean predicates stay deferred (P8);
    both halves materialize ~10% of the spine, never the whole source.
    """
    rendition_ids = await resolve_source_renditions(queue, graph_id, source_id, rendition_selector)
    if not rendition_ids:
        return []
    r1 = await graph_task(queue, graph_id, "query_nodes", query=_spine_query(
        rendition_ids, where=[PropertyPredicate("text", "eq", "")]).to_dict())
    r2 = await graph_task(queue, graph_id, "query_nodes", query=_spine_query(
        rendition_ids, where=[PropertyPredicate("text", "is_null")]).to_dict())
    segs = [_row_to_spine_segment(r) for r in (r1.rows or []) + (r2.rows or [])]
    segs.sort(key=lambda s: s.index)
    return segs


async def count_source_segments(
    queue: JobQueue,  # Started job queue
    graph_id: str,    # Graph-storage capability id
    source_id: str,   # Source node id
    rendition_selector: Optional[str] = None,  # Which rendition spine (None = auto-select)
) -> int:  # Number of fine Segment nodes under the Source's chosen rendition
    """Count a Source's segments server-side under its chosen rendition (typed count mode)."""
    rendition_ids = await resolve_source_renditions(queue, graph_id, source_id, rendition_selector)
    if not rendition_ids:
        return 0
    q = _spine_query(rendition_ids, order_by=None, project=None, count=True)
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
    correction_type: str,                  # "text_content" | "punctuation" | "grouping"
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
) -> CorrectionSession:  # The committed CorrectionSession
    """Create + commit a new CorrectionSession node."""
    sess = CorrectionSession(scope=scope)
    await commit_nodes_edges(queue, graph_id, [sess.to_graph_node().to_dict()], [])
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
) -> None:
    """Update a session's status + updated_at.

    The ONLY update_node use in this core: a CorrectionSession is OVERLAY metadata
    whose lifecycle is mutable. Layer-0 Segments + Corrections stay append-only.
    """
    await graph_task(queue, graph_id, "update_node", node_id=session_id,
                     properties={"status": status, "updated_at": time.time()})


async def record_review_markers(
    queue: JobQueue,                   # Started job queue
    graph_id: str,                     # Graph-storage capability id
    session_id: str,                   # Owning session id
    decisions: List[Tuple[str, str]],  # (segment_id, decision) pairs
) -> int:  # Number of REVIEWED edges committed
    """Persist per-(session, segment) review markers as REVIEWED edges."""
    edges = [_edge(session_id, seg_id, CorrectionRelations.REVIEWED, {"decision": dec})
             for seg_id, dec in decisions]
    return (await commit_nodes_edges(queue, graph_id, [], edges))["edges"]


async def load_review_markers(
    queue: JobQueue,  # Started job queue
    graph_id: str,    # Graph-storage capability id
    session_id: str,  # Owning session id
) -> Dict[str, str]:  # segment_id -> decision for the session
    """Load a session's review markers (typed edge projection over REVIEWED edges)."""
    q = EdgeQuery(source_id=session_id, relation_type=CorrectionRelations.REVIEWED,
                  project=["decision"])
    res = await graph_task(queue, graph_id, "query_edges", query=q.to_dict())
    return {r["target_id"]: r["decision"] for r in (res.rows or [])}


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
        if c.get("status") == "superseded":  # legacy status skip retained
            continue
        ctype = c.get("correction_type")
        payload = c.get("payload", {}) or {}
        created = float(c.get("created_at") or 0.0)
        if ctype == "grouping" and payload.get("operation") == "prune_empty":
            edits.append(SpineEdit(edit_id=c["id"], op="prune",
                                   targets=list(payload.get("pruned_segment_ids") or []),
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
    """
    units = [SpineUnit(id=s.id, text=s.text) for s in segments]
    out_units = layer_project_effective_spine(units, corrections_to_edits(corrections))
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
    return out


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
) -> str:  # The new Correction node id
    """Commit a text_content correction (node + CORRECTS [+ SUPERSEDES]) + a REVIEWED marker."""
    node, edges = build_text_correction(
        source_id, segment_id, new_text, session_id, old_text=old_text,
        supersedes_id=supersedes_id, actor=actor, canonical_form=canonical_form)
    await commit_nodes_edges(queue, graph_id, [node], edges)
    await record_review_markers(queue, graph_id, session_id, [(segment_id, "corrected")])
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
