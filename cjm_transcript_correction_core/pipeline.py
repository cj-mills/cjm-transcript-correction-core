"""The headless correction workflow: load a decomp run manifest, resolve the shared graph DB, start/resume/reopen a CorrectionSession, recompute the worklist from deterministic signals + persisted review state, run the D14 empty-segment prune (first operation), and record a chainable correction run manifest — with a cheapest-form HITL approval seam."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cjm_substrate.core.journal_store import JournalEvent, SubstrateEventType
from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate.core.queue import JobQueue
from cjm_substrate.core.workspace import resolve_recorded_tree
from cjm_transcript_correction_core.graph import (active_corrections, build_prune_correction,
                                                  commit_nodes_edges, commit_text_correction,
                                                  count_source_segments,
                                                  find_active_text_corrections_batch,
                                                  find_corrections_for_session,
                                                  find_prior_corrections_by_hash, get_session,
                                                  load_empty_segments, load_review_markers,
                                                  load_source_corrections, load_source_segments,
                                                  load_variant_texts, project_effective_spine,
                                                  record_review_markers, set_session_status,
                                                  start_session)
from cjm_transcript_correction_core.models import (CorrectionConfig, CorrectionManifest, new_run_id,
                                                   SpineSegment, WorklistItem)
from cjm_transcript_correction_core.signals import (compute_signal_flags, detect_empty_segments,
                                                    variant_divergence)

logger = logging.getLogger(__name__)


def load_decomp_manifest(
    path: str,  # Path to a decomp-core run manifest JSON
) -> Dict[str, Any]:  # Parsed manifest dict
    """Load + lightly validate a decomp-core run manifest (untyped JSON; CR-20 interchange).

    ${WS}/ recorded paths (5daadfc4 rung f) resolve to absolute here, anchored
    at the manifest's own location (active workspace as fallback)."""
    data = resolve_recorded_tree(json.loads(Path(path).read_text()), Path(path))
    fmt = data.get("format", "")
    if "decomp-core" not in fmt:
        logger.warning(f"unexpected decomp manifest format: {fmt!r} (continuing)")
    return data


def resolve_graph_db_path(
    manifest: Dict[str, Any],        # Decomp manifest dict
    graph_capability: str,               # Graph-storage capability id
    override: Optional[str] = None,  # Explicit override (wins)
) -> Optional[str]:  # Absolute DB path the spine lives in
    """Resolve the graph DB path: explicit override > the decomp manifest's recorded db_path.

    The spine lives in the decomp-core graph DB; this core writes its overlay into
    the SAME DB (a shared substrate resource owned by no single core -- pass-2
    graph-DB-ownership evidence).
    """
    if override:
        return override
    rec = (manifest.get("capabilities", {}) or {}).get(graph_capability, {}) or {}
    return rec.get("db_path")


def compute_worklist(
    segments: List[SpineSegment],                       # Ordered layer-0 spine
    review_markers: Dict[str, str],                     # segment_id -> decision (persisted)
    variants: Optional[Dict[str, Dict[str, str]]] = None,  # segment_id -> {transcriber: text} (intra-graph)
) -> List[WorklistItem]:  # Items still needing review, flagged
    """Recompute the worklist from layer-0 + signals + review state (only decisions persist).

    Segments already decided in this session (reviewed/corrected/skipped) drop out;
    everything flagged by a deterministic signal and not yet decided surfaces.
    """
    flags = compute_signal_flags(segments, variants=variants)
    items: List[WorklistItem] = []
    for i, s in enumerate(segments):
        if review_markers.get(s.id):  # already decided this session
            continue
        fl = flags.get(i, [])
        if fl:
            items.append(WorklistItem(segment=s, flags=fl))
    return items


def confirm_seam(
    seam: str,                 # Seam label
    summary_lines: List[str],  # What the operator is accepting
    warnings: List[str],       # Tier-1 warnings
    assume_yes: bool = False,  # Headless: accept without prompting
) -> bool:  # True = proceed, False = aborted
    """HITL approval seam in its cheapest viable form (log + optional CLI prompt).

    Per-seam HITL-assist annotation (5 fields):
      1. signal: per-document summary + Tier-1 worklist flags
      2. deterministic pre-filter: compute_signal_flags (no AI)
      3. modality-bridge candidate: spectrogram / audio review (future Tier 2)
      4. authoritative verifier: re-transcribe-and-compare (future Tier 3; Gemini)
      5. flywheel capture: decisions persist as graph nodes/edges (DURABLE, unlike
         decomp's log-only seam -- the correction overlay IS the captured decision)
    input() blocks the loop; safe between stages with no jobs in flight (pass-2).
    """
    for line in summary_lines:
        logger.info(f"[{seam}] {line}")
    for w in warnings:
        logger.warning(f"[{seam}] {w}")
    if assume_yes:
        logger.info(f"[{seam}] auto-accepted (assume_yes)")
        return True
    reply = input(f"[{seam}] proceed? [Y/n] ").strip().lower()
    accepted = reply in ("", "y", "yes")
    logger.info(f"[{seam}] {'accepted' if accepted else 'ABORTED'} by operator")
    return accepted


async def prune_empty_segments(
    queue: JobQueue,        # Started job queue
    cfg: CorrectionConfig,  # Run configuration
    graph_id: str,          # Graph-storage capability id
    source_id: str,         # Source being corrected
    total_count: int,       # Total segment count (for the summary)
    session_id: str,        # Owning session id
) -> Dict[str, Any]:  # {"pruned": n, "correction_id": id|None}
    """First operation: prune empty (silence) segments as one grouping correction (D14).

    Deterministic, no-human restructure proof: loads ONLY the empty segments
    (server-side filter -- scale-shaped) under the chosen rendition spine, builds
    a batch grouping Correction + DERIVED_FROM edges, commits via the queue, and
    records REVIEWED markers (decision=corrected). Layer-0 untouched; reversible
    by superseding.
    """
    empties = await load_empty_segments(queue, graph_id, source_id,
                                        rendition_selector=cfg.rendition_selector)
    if not empties:
        logger.info(f"[prune] {source_id}: no empty segments")
        return {"pruned": 0, "correction_id": None}
    if not confirm_seam("prune-review",
                        [f"{source_id}: prune {len(empties)}/{total_count} empty segment(s)"],
                        [], assume_yes=cfg.assume_yes):
        logger.warning(f"[prune] {source_id}: declined by operator")
        return {"pruned": 0, "correction_id": None}
    node, edges = build_prune_correction(source_id, empties, session_id, actor=cfg.actor)
    await commit_nodes_edges(queue, graph_id, [node], edges)
    await record_review_markers(queue, graph_id, session_id,
                                [(s.id, "corrected") for s in empties])
    logger.info(f"[prune] {source_id}: grouping correction {node['id']} pruned {len(empties)} segment(s)")
    return {"pruned": len(empties), "correction_id": node["id"]}


def collect_capability_info(
    manager: CapabilityManager,   # Manager holding the loaded capabilities
    instance_ids: List[str],  # Instance ids to record
) -> Dict[str, Dict[str, Any]]:  # instance_id -> {name, version, db_path}
    """Record capability identity + data-DB pointers for the run manifest (provenance)."""
    info: Dict[str, Dict[str, Any]] = {}
    for iid in instance_ids:
        meta = (getattr(manager, "capabilities", {}) or {}).get(iid)
        if meta is None:
            continue
        manifest = getattr(meta, "manifest", {}) or {}
        info[iid] = {"name": meta.name, "version": getattr(meta, "version", None),
                     "db_path": manifest.get("db_path")}
    return info


def _journal_run_event(
    manager: CapabilityManager,  # Manager owning the journal store
    event_type: str,         # SubstrateEventType value (run_started / run_finished)
    run_id: str,             # This run's manifest id
    actor: Optional[str],    # Who/what initiated the run (cfg.actor)
    payload: Dict[str, Any], # Run-level structured detail
) -> None:
    """Append a host-tier run event to the journal (CR-14 follow-up).

    The cores are the trusted host writer class: RUN_STARTED/RUN_FINISHED
    bracket the run so the run manifest (same run_id) links to every job row
    the run produced. No-op when the manager has no journal store; append
    failures stay LOUD (journal contract)."""
    journal = getattr(manager, "journal_store", None)
    if journal is None:
        return
    journal.append(JournalEvent(
        event_type=event_type, run_id=run_id, actor=actor, payload=payload))


async def run_correction(
    manager: CapabilityManager,            # Manager with the graph capability loaded
    queue: JobQueue,                   # Started job queue
    cfg: CorrectionConfig,             # Run configuration
    decomp_manifest_path: str,         # Decomp run manifest to correct
    graph_db_path: str,                # Resolved graph DB path (shared with decomp)
    run_id: Optional[str] = None,      # Override run id
    session_id: Optional[str] = None,  # Resume/reopen an existing session
    reopen: bool = False,              # Reopen a completed session
) -> CorrectionManifest:  # Manifest of the correction run
    """Correct every source in a decomp run manifest (prune + worklist surfacing).

    Per source: load spine (with variant slices) -> recompute worklist -> prune
    empty segments [prune-review seam] -> project effective spine -> record
    outcome. The cross-transcriber diff is INTRA-GRAPH (stage 5): variant texts
    come from the segments' own Transcript slice refs — no second manifest. The
    fine spine is scoped to the chosen AudioRendition (cfg.rendition_selector;
    auto-selected when a source has one decomposed rendition). Resumable: a prior
    session's review markers drop decided segments.
    """
    run_id = run_id or new_run_id()
    # CR-14 follow-up: queue-scoped run context — every job submitted in this
    # run carries run_id/actor into its journal rows + worker diagnostics
    # (run-manifest <-> journal linkage); the run itself is bracketed by
    # RUN_STARTED/RUN_FINISHED host-tier rows. Actor = cfg.actor (the same
    # attribution recorded on Corrections in the graph).
    queue.set_run_context(run_id=run_id, actor=cfg.actor)
    _journal_run_event(manager, SubstrateEventType.RUN_STARTED.value, run_id, cfg.actor, {
        "core": "cjm-transcript-correction-core", "mode": "correction",
        "decomp_manifest": str(decomp_manifest_path),
        "graph_capability": cfg.graph_capability,
    })
    try:
        decomp = load_decomp_manifest(decomp_manifest_path)
        source_ids = [s.get("source_node_id") for s in (decomp.get("sources", []) or [])
                      if s.get("source_node_id")]
        if not source_ids:
            raise SystemExit("decomp manifest lists no sources (pre-0.2.0 manifest? re-run decomp)")

        # Session: start fresh, or resume/reopen an existing one.
        if session_id:
            if await get_session(queue, cfg.graph_capability, session_id) is None:
                raise SystemExit(f"session {session_id} not found in graph")
            if reopen:
                await set_session_status(queue, cfg.graph_capability, session_id, "reopened")
            sess_id = session_id
            logger.info(f"resumed session {sess_id}")
        else:
            sess = await start_session(queue, cfg.graph_capability, source_ids)
            sess_id = sess.id
            logger.info(f"started session {sess_id} over {len(source_ids)} source(s)")

        manifest = CorrectionManifest(
            run_id=run_id, created_at=time.time(), config=cfg.to_dict(),
            decomp_manifest=str(decomp_manifest_path),
            graph_db_path=graph_db_path, session_id=sess_id,
            source_format=decomp.get("format", ""), source_version=decomp.get("version", ""),
            signals_used=["empty-text", "missing-timing", "boundary-missing-terminal",
                          "boundary-terminal-then-lowercase", "transcriber-divergence"],
        )

        for sid in source_ids:
            n = await count_source_segments(queue, cfg.graph_capability, sid,
                                            rendition_selector=cfg.rendition_selector)
            segments = await load_source_segments(queue, cfg.graph_capability, sid,
                                                  rendition_selector=cfg.rendition_selector)
            variants = await load_variant_texts(queue, cfg.graph_capability, segments)
            markers = await load_review_markers(queue, cfg.graph_capability, sess_id)
            worklist = compute_worklist(segments, markers, variants=variants)
            divergences = sum(1 for it in worklist if "transcriber-divergence" in it.flags)
            logger.info(f"[src {sid[:8]}] {n} segment(s); worklist {len(worklist)} flagged; "
                        f"{len(detect_empty_segments(segments))} empty; "
                        f"{divergences} transcriber-divergence (intra-graph)")

            prune = {"pruned": 0, "correction_id": None}
            if cfg.prune_empty:
                prune = await prune_empty_segments(queue, cfg, cfg.graph_capability, sid, n, sess_id)

            corrections = await find_corrections_for_session(queue, cfg.graph_capability, sess_id)
            effective = project_effective_spine(segments, corrections)
            manifest.sources.append({
                "source_node_id": sid,
                "segment_count": n,
                "worklist_flagged": len(worklist),
                "empty_segments": len(detect_empty_segments(segments)),
                "transcriber_divergences": divergences,
                "pruned": prune["pruned"],
                "prune_correction_id": prune["correction_id"],
                "effective_segment_count": len(effective),
            })

        await set_session_status(queue, cfg.graph_capability, sess_id, "completed")
    except BaseException as e:
        # The journal exists for exactly this row: a run that DIED records
        # that it was attempted (failures stop being the unattributed case).
        _journal_run_event(manager, SubstrateEventType.RUN_FINISHED.value, run_id, cfg.actor, {
            "core": "cjm-transcript-correction-core", "mode": "correction",
            "status": "failed", "error": repr(e),
        })
        raise
    _journal_run_event(manager, SubstrateEventType.RUN_FINISHED.value, run_id, cfg.actor, {
        "core": "cjm-transcript-correction-core", "mode": "correction",
        "status": "completed", "session_id": sess_id,
        "sources": len(manifest.sources),
    })
    return manifest


def _format_worklist_item(
    item: WorklistItem,                     # The item to present
    effective_text: str,                    # Current effective text (layer-0 or latest correction)
    variant_text: Optional[str] = None,     # Divergent variant text (another transcriber's reading)
    prior_correction: Optional[str] = None, # Suggested text from the cross-transcript cache
) -> str:  # Multi-line presentation block
    """Render a worklist item for the CLI review seam (text + timing + flags + hints)."""
    s = item.segment
    t = f"[{s.start_time:.1f}-{s.end_time:.1f}s]" if s.start_time is not None else "[--]"
    lines = [f"  #{s.index} {t} flags={','.join(item.flags)}",
             f"    text:    {effective_text!r}"]
    if variant_text is not None:
        lines.append(f"    variant: {variant_text!r}")
    if prior_correction is not None:
        lines.append(f"    cache-hit: {prior_correction!r}")
    return "\n".join(lines)


async def review_worklist(
    queue: JobQueue,                                    # Started job queue
    cfg: CorrectionConfig,                              # Run configuration
    source_id: str,                                     # Source under review
    worklist: List[WorklistItem],                       # Flagged, undecided items
    session_id: str,                                    # Owning session id
    variant_by_segment: Optional[Dict[str, str]] = None,  # segment_id -> divergent variant text
    max_items: int = 0,                                 # Cap (0 = all)
) -> Dict[str, int]:  # {"corrected": n, "skipped": n, "reviewed": n}
    """Interactive review loop -> text_content corrections (cheapest HITL seam).

    Per item: present text + timing + flags (+ the divergent variant text +
    cross-transcript cache hit), then read a decision from stdin:
      [a]ccept (mark reviewed) / [e]dit (commit a text_content correction;
      auto-supersedes any prior correction on the segment) / [s]kip / [q]uit.
    On 'e', the next stdin line is the new text (blank -> adopt the variant) —
    adopting a variant IS the "extract the superior lightweight reading" move,
    and its provenance is the variant slice already on the segment.
    Drivable headless via a stdin pipe (E9-companion); cfg.assume_yes marks every
    item reviewed with no edits.
    """
    variant_by_segment = variant_by_segment or {}
    counts = {"corrected": 0, "skipped": 0, "reviewed": 0}
    items = worklist[:max_items] if max_items > 0 else worklist
    # C17 (stage 4): ONE batched far-end read replaces the per-item lookup —
    # 2 graph round-trips for the whole worklist instead of 1 per item. (The
    # per-item hash-cache lookup below stays lazy/per-item: batching it needs
    # a hash-LIST far-end constraint — recorded promotion candidate.)
    active_by_segment = await find_active_text_corrections_batch(
        queue, cfg.graph_capability, [it.segment.id for it in items])
    for item in items:
        seg = item.segment
        active = active_by_segment.get(seg.id)
        effective_text = (active.get("payload", {}) or {}).get("new_text", seg.text) if active else seg.text
        var = variant_by_segment.get(seg.id)
        prior = None
        if seg.content_hash:
            hits = await find_prior_corrections_by_hash(queue, cfg.graph_capability, seg.content_hash)
            hits = [h for h in hits if (h.get("payload") or {}).get("segment_id") != seg.id]
            if hits:
                prior = (hits[0].get("payload") or {}).get("new_text")
        logger.info("review item:\n" + _format_worklist_item(item, effective_text, var, prior))
        if cfg.assume_yes:
            await record_review_markers(queue, cfg.graph_capability, session_id, [(seg.id, "reviewed")])
            counts["reviewed"] += 1
            continue
        try:
            decision = input(f"  #{seg.index} [a]ccept/[e]dit/[s]kip/[q]uit: ").strip().lower()
        except EOFError:
            break
        if decision in ("q", "quit"):
            break
        if decision in ("e", "edit"):
            try:
                new_text = input("    new text (blank = adopt variant): ").rstrip("\n")
            except EOFError:
                new_text = ""
            if not new_text and var:
                new_text = var
            if not new_text:
                await record_review_markers(queue, cfg.graph_capability, session_id, [(seg.id, "skipped")])
                counts["skipped"] += 1
                continue
            cid = await commit_text_correction(
                queue, cfg.graph_capability, source_id, seg.id, new_text, session_id,
                old_text=effective_text, supersedes_id=(active.get("id") if active else None),
                actor=cfg.actor)
            logger.info(f"  #{seg.index}: text correction {cid}"
                        + (f" supersedes {active['id']}" if active else ""))
            counts["corrected"] += 1
        elif decision in ("s", "skip"):
            await record_review_markers(queue, cfg.graph_capability, session_id, [(seg.id, "skipped")])
            counts["skipped"] += 1
        else:
            await record_review_markers(queue, cfg.graph_capability, session_id, [(seg.id, "reviewed")])
            counts["reviewed"] += 1
    return counts


async def run_review(
    manager: CapabilityManager,            # Manager with the graph capability loaded
    queue: JobQueue,                   # Started job queue
    cfg: CorrectionConfig,             # Run configuration
    decomp_manifest_path: str,         # Decomp run manifest to review
    graph_db_path: str,                # Resolved graph DB path (shared with decomp)
    run_id: Optional[str] = None,      # Override run id
    session_id: Optional[str] = None,  # Resume/reopen an existing session
    reopen: bool = False,              # Reopen a completed session
    max_items: int = 0,                # Max worklist items to review per source (0 = all)
) -> CorrectionManifest:  # Manifest of the review run
    """Interactive review pass over a decomp manifest's flagged worklist (text corrections).

    Like run_correction but enters the text-correction review loop instead of the
    prune. Empty segments are excluded from the review worklist (they belong to the
    prune); the effective spine is projected from the SOURCE's corrections (across
    sessions), so a resumed/reopened session sees prior prune + text corrections.
    The variant hints come from the segments' own slice refs (intra-graph). The
    fine spine is scoped to the chosen AudioRendition (cfg.rendition_selector).
    """
    run_id = run_id or new_run_id()
    # CR-14 follow-up: queue-scoped run context — every job submitted in this
    # run carries run_id/actor into its journal rows + worker diagnostics
    # (run-manifest <-> journal linkage); the run itself is bracketed by
    # RUN_STARTED/RUN_FINISHED host-tier rows. Actor = cfg.actor (the same
    # attribution recorded on Corrections in the graph).
    queue.set_run_context(run_id=run_id, actor=cfg.actor)
    _journal_run_event(manager, SubstrateEventType.RUN_STARTED.value, run_id, cfg.actor, {
        "core": "cjm-transcript-correction-core", "mode": "review",
        "decomp_manifest": str(decomp_manifest_path),
        "graph_capability": cfg.graph_capability,
    })
    try:
        decomp = load_decomp_manifest(decomp_manifest_path)
        source_ids = [s.get("source_node_id") for s in (decomp.get("sources", []) or [])
                      if s.get("source_node_id")]
        if not source_ids:
            raise SystemExit("decomp manifest lists no sources (pre-0.2.0 manifest? re-run decomp)")

        if session_id:
            if await get_session(queue, cfg.graph_capability, session_id) is None:
                raise SystemExit(f"session {session_id} not found in graph")
            if reopen:
                await set_session_status(queue, cfg.graph_capability, session_id, "reopened")
            sess_id = session_id
            logger.info(f"resumed session {sess_id}")
        else:
            sess = await start_session(queue, cfg.graph_capability, source_ids)
            sess_id = sess.id
            logger.info(f"started review session {sess_id} over {len(source_ids)} source(s)")

        manifest = CorrectionManifest(
            run_id=run_id, created_at=time.time(), config=cfg.to_dict(),
            decomp_manifest=str(decomp_manifest_path),
            graph_db_path=graph_db_path, session_id=sess_id,
            source_format=decomp.get("format", ""), source_version=decomp.get("version", ""),
            signals_used=["boundary-missing-terminal", "boundary-terminal-then-lowercase",
                          "transcriber-divergence"],
        )

        for sid in source_ids:
            n = await count_source_segments(queue, cfg.graph_capability, sid,
                                            rendition_selector=cfg.rendition_selector)
            segments = await load_source_segments(queue, cfg.graph_capability, sid,
                                                  rendition_selector=cfg.rendition_selector)
            variants = await load_variant_texts(queue, cfg.graph_capability, segments)
            variant_by_segment: Dict[str, str] = {}
            for i, (auth, var) in variant_divergence(segments, variants).items():
                variant_by_segment[segments[i].id] = var
            markers = await load_review_markers(queue, cfg.graph_capability, sess_id)
            worklist = [it for it in compute_worklist(segments, markers, variants=variants)
                        if not it.segment.is_empty]
            logger.info(f"[src {sid[:8]}] {n} segment(s); review worklist {len(worklist)} "
                        f"(reviewing up to {max_items or len(worklist)})")
            counts = await review_worklist(queue, cfg, sid, worklist, sess_id,
                                           variant_by_segment=variant_by_segment, max_items=max_items)

            corrections, superseded = await load_source_corrections(queue, cfg.graph_capability, sid)
            active = active_corrections(corrections, superseded)
            effective = project_effective_spine(segments, active)
            manifest.sources.append({
                "source_node_id": sid, "segment_count": n, "review_worklist": len(worklist),
                "corrected": counts["corrected"], "skipped": counts["skipped"],
                "reviewed": counts["reviewed"], "active_corrections": len(active),
                "superseded_corrections": len(superseded), "effective_segment_count": len(effective),
            })
            logger.info(f"[src {sid[:8]}] corrected={counts['corrected']} skipped={counts['skipped']} "
                        f"reviewed={counts['reviewed']}; active corrections={len(active)}; "
                        f"effective spine={len(effective)}")

        await set_session_status(queue, cfg.graph_capability, sess_id, "completed")
    except BaseException as e:
        # The journal exists for exactly this row: a run that DIED records
        # that it was attempted (failures stop being the unattributed case).
        _journal_run_event(manager, SubstrateEventType.RUN_FINISHED.value, run_id, cfg.actor, {
            "core": "cjm-transcript-correction-core", "mode": "review",
            "status": "failed", "error": repr(e),
        })
        raise
    _journal_run_event(manager, SubstrateEventType.RUN_FINISHED.value, run_id, cfg.actor, {
        "core": "cjm-transcript-correction-core", "mode": "review",
        "status": "completed", "session_id": sess_id,
        "sources": len(manifest.sources),
    })
    return manifest
