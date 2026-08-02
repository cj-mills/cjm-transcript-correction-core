"""The CLI driver — the correction core's first (and currently only) frontend. run <decomp-manifest> corrects the committed spine in the decomp graph DB, pointing the graph worker at that shared DB via load-time config, with optional session resume/reopen; review runs the interactive text-correction loop (the cross-transcriber diff is intra-graph since stage 5)."""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from cjm_context_graph_layer.journal import sidecar_journal_path
from cjm_context_graph_layer.ops import graph_task
from cjm_context_graph_primitives.query import NodeQuery
from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate.core.queue import JobQueue
from cjm_substrate.core.workspace import relativize_recorded, resolve_workspace
from cjm_transcript_correction_core.graph import (bench_event_proposals,
                                                  commit_chunk_insert_correction,
                                                  commit_extraction_gate, correction_stats,
                                                  extract_spine_dataset, labeled_insert_spans,
                                                  list_source_spines, load_extraction_gates,
                                                  load_source_corrections, load_source_segments,
                                                  project_effective_spine, set_session_status,
                                                  skeleton_hash_for, start_session)
from cjm_transcript_correction_core.models import CorrectionConfig, DatasetManifest, new_dataset_id
from cjm_transcript_correction_core.pipeline import (load_decomp_manifest, resolve_graph_db_path,
                                                     run_correction, run_review)
from cjm_transcript_correction_core.signals import (EVENT_PROPOSAL_SET_FORMAT,
                                                    load_event_proposal_set)

logger = logging.getLogger(__name__)


def _add_common_run_args(p: argparse.ArgumentParser) -> None:  # Shared run/review arguments
    """Attach the capability / session / output arguments shared by `run` and `review`."""
    p.add_argument("manifest", help="Decomp-core run manifest JSON (the committed spine)")
    p.add_argument("--manifests-dir", default=None,
                   help="Capability manifests directory (default: the workspace's .cjm/manifests "
                        "when one is active, else .cjm/manifests under the cwd)")
    p.add_argument("--workspace", default=None,
                   help="Workspace root (5daadfc4; default: CJM_WORKSPACE env, else upward walk "
                        "from cwd). Supplies manifests/output defaults and is exported so "
                        "capability workers resolve workspace-scoped paths")
    p.add_argument("--graph-capability", default="cjm-capability-graph-sqlite", help="Graph-storage capability name")
    p.add_argument("--graph-db-path", default=None,
                   help="Override graph DB path (default: the decomp manifest's recorded db_path)")
    p.add_argument("--rendition", default=None,
                   help="Which AudioRendition spine to correct when a source has more than one "
                        "(\"raw\" or a preprocessing substring e.g. \"demucs\"); default: auto-select the "
                        "decomposed one (errors if ambiguous)")
    p.add_argument("--skeleton", default=None,
                   help="Which SKELETON spine to correct when several coexist under one rendition "
                        "(sentence-split, DEC f1024568): \"legacy\" = the pre-split spine, or a "
                        "skeleton-hash prefix; default: auto (errors when >1 coexist)")
    p.add_argument("--sysmon-capability", default=None,
                   help="monitor for empirical attribution (CR-7); loaded first; default: none")
    p.add_argument("--session", default=None, help="Resume an existing CorrectionSession id")
    p.add_argument("--reopen", action="store_true", help="Reopen a completed session (with --session)")
    p.add_argument("--actor", default="human", help="Actor recorded on corrections + review markers")
    p.add_argument("--output", default=None,
                   help="Correction-manifest output path (default: <workspace>/runs/<run_id>.json "
                        "when a workspace is active, else runs/<run_id>.json under the cwd)")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")


def build_parser() -> argparse.ArgumentParser:  # Configured CLI parser
    """Build the CLI parser (subcommands: run, review).

    Stage 5: --secondary-manifest is RETIRED — the cross-transcriber diff is
    intra-graph now (variant slices on the shared-skeleton segments).
    """
    parser = argparse.ArgumentParser(
        prog="cjm-transcript-correction-core",
        description="Headless transcript correction: non-destructive overlay on a committed source spine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Prune empty segments + surface the worklist (deterministic)")
    _add_common_run_args(run)
    run.add_argument("--no-prune", action="store_true", help="Skip the D14 empty-segment prune")
    run.add_argument("-y", "--yes", action="store_true", help="Auto-accept HITL seams (headless mode)")

    review = sub.add_parser("review", help="Interactive text-correction review of the flagged worklist")
    _add_common_run_args(review)
    review.add_argument("--review-max", type=int, default=0, help="Max worklist items to review (0 = all)")
    review.add_argument("-y", "--yes", action="store_true", help="Auto-mark every reviewed item (no edits)")

    stats = sub.add_parser(
        "stats", help="Flywheel accounting: active labeled spans / open marks / op counts per source")
    stats.add_argument("--manifests-dir", default=None,
                       help="Capability manifests directory (default: the workspace's .cjm/manifests "
                            "when one is active, else .cjm/manifests under the cwd)")
    stats.add_argument("--workspace", default=None,
                       help="Workspace root (default: CJM_WORKSPACE env, else upward walk from cwd)")
    stats.add_argument("--graph-capability", default="cjm-capability-graph-sqlite",
                       help="Graph-storage capability name")
    stats.add_argument("--graph-db-path", default=None,
                       help="Graph db path (default: the capability's persisted workspace config)")
    stats.add_argument("--source", default=None,
                       help="Source node id or title substring (default: every Source)")
    stats.add_argument("--label", default=None,
                       help="Spotlight one label/class (e.g. inhale): prints its insert+mark grand total")
    stats.add_argument("--genuine-only", action="store_true",
                       help="Count only corrections from GENUINE sessions (purpose unset/genuine — "
                            "feature-test noise excluded, DEC c86714a4); supersession stays global")
    stats.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")

    gate = sub.add_parser(
        "gate", help="Per-spine extraction gate: show/assert status + annotated_through "
                     "watermark (DEC 8e05b87b; flywheel build leg 1)")
    gate.add_argument("--manifests-dir", default=None,
                      help="Capability manifests directory (default: the workspace's .cjm/manifests "
                           "when one is active, else .cjm/manifests under the cwd)")
    gate.add_argument("--workspace", default=None,
                      help="Workspace root (default: CJM_WORKSPACE env, else upward walk from cwd)")
    gate.add_argument("--graph-capability", default="cjm-capability-graph-sqlite",
                      help="Graph-storage capability name")
    gate.add_argument("--graph-db-path", default=None,
                      help="Graph db path (default: the capability's persisted workspace config)")
    gate.add_argument("--source", default=None,
                      help="Source node id or title substring (default: every Source; "
                           "REQUIRED to assert)")
    gate.add_argument("--rendition", default=None,
                      help="Which AudioRendition spine when a source has more than one")
    gate.add_argument("--skeleton", default=None,
                      help="Which SKELETON spine (\"legacy\" or a hash prefix); default: auto "
                           "(errors when several coexist)")
    gate.add_argument("--status", default=None, choices=["in_progress", "signed_off", "excluded"],
                      help="Assert this extraction_status (with --source); omit = show gates")
    gate.add_argument("--annotated-through", default=None,
                      help="Watermark in source seconds, or \"end\" = the spine's last segment end "
                           "(default when asserting: keep the current watermark)")
    gate.add_argument("--actor", default="human", help="Actor recorded on the assertion")
    gate.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")

    extract = sub.add_parser(
        "extract", help="Fold the gated overlay into a manifested dataset "
                        "(flywheel build leg 2; lands workspace-local under datasets/)")
    extract.add_argument("--manifests-dir", default=None,
                         help="Capability manifests directory (default: the workspace's "
                              ".cjm/manifests when one is active, else .cjm/manifests "
                              "under the cwd)")
    extract.add_argument("--workspace", default=None,
                         help="Workspace root (default: CJM_WORKSPACE env, else upward walk from cwd)")
    extract.add_argument("--graph-capability", default="cjm-capability-graph-sqlite",
                         help="Graph-storage capability name")
    extract.add_argument("--graph-db-path", default=None,
                         help="Graph db path (default: the capability's persisted workspace config)")
    extract.add_argument("--source", default=None,
                         help="Source node id or title substring (default: every Source)")
    extract.add_argument("--rendition", default=None,
                         help="Which AudioRendition spine when a source has more than one")
    extract.add_argument("--include-purpose", action="append", default=None,
                         help="Session purposes whose spans qualify as EXAMPLES (repeatable; "
                              "default: genuine only — the load-bearing 493b8b9e cut; an "
                              "unset purpose counts as genuine)")
    extract.add_argument("--output-dir", default=None,
                         help="Datasets root (default: <workspace>/datasets, else datasets/ "
                              "under the cwd); the dataset lands in <root>/<dataset_id>/")
    extract.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")

    bench = sub.add_parser(
        "bench", help="Reserved-tail bench: derive accept/edit/reject verdicts by joining "
                      "a proposal set against final spine state (leg 4, DECs 8e05b87b + "
                      "8cf12c22 — verdicts are never stored)")
    bench.add_argument("--manifests-dir", default=None,
                       help="Capability manifests directory (default: the workspace's "
                            ".cjm/manifests when one is active, else .cjm/manifests "
                            "under the cwd)")
    bench.add_argument("--workspace", default=None,
                       help="Workspace root (default: CJM_WORKSPACE env, else upward walk from cwd)")
    bench.add_argument("--graph-capability", default="cjm-capability-graph-sqlite",
                       help="Graph-storage capability name")
    bench.add_argument("--graph-db-path", default=None,
                       help="Graph db path (default: the capability's persisted workspace config)")
    bench.add_argument("--source", default=None,
                       help="Source node id or title substring (default: the proposal set's "
                            "recorded source_id)")
    bench.add_argument("--rendition", default=None,
                       help="Which AudioRendition spine when a source has more than one")
    bench.add_argument("--skeleton", default=None,
                       help="Skeleton-spine selector (full skeleton hash). Default: the "
                            "proposal set's recorded skeleton_hash; a respine propset "
                            "predates its spine and records none, so the CONSUMING decomp "
                            "manifest resolves it (event_propset_id chain, DEC 6cc10fb7); "
                            "legacy spine as the last resort")
    bench.add_argument("--proposals", default=None,
                       help="Proposal-set directory or manifest.json (default: the latest set "
                            "under <workspace>/proposals/ matching the source)")
    bench.add_argument("--include-purpose", action="append", default=None,
                       help="Session purposes whose inserts count as human verdicts "
                            "(repeatable; default: genuine only)")
    bench.add_argument("--tolerance", type=float, default=0.15,
                       help="Boundary tolerance in seconds separating accepted from edited")
    bench.add_argument("--output", default=None,
                       help="Also write the full verdict report as JSON to this path")
    bench.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")

    transfer = sub.add_parser(
        "transfer-wordless",
        help="Replay one spine's ACTIVE wordless event inserts onto a sibling spine "
             "anchored by source time (dea104ba: events = source layer, words = spine "
             "layer — boundary shifts, text edits, and word-bearing inserts stay behind)")
    transfer.add_argument("--manifests-dir", default=None,
                          help="Capability manifests directory (default: the workspace's "
                               ".cjm/manifests when one is active, else .cjm/manifests "
                               "under the cwd)")
    transfer.add_argument("--workspace", default=None,
                          help="Workspace root (default: CJM_WORKSPACE env, else upward walk from cwd)")
    transfer.add_argument("--graph-capability", default="cjm-capability-graph-sqlite",
                          help="Graph-storage capability name")
    transfer.add_argument("--graph-db-path", default=None,
                          help="Graph db path (REQUIRED to commit — the journal sidecar "
                               "hangs off it; --dry-run works without)")
    transfer.add_argument("--source", required=True,
                          help="Source node id or title substring (exactly one match)")
    transfer.add_argument("--rendition", default=None,
                          help="Which AudioRendition spine when a source has more than one")
    transfer.add_argument("--from-skeleton", required=True,
                          help="Donor spine (\"legacy\" or a skeleton-hash prefix)")
    transfer.add_argument("--to-skeleton", required=True,
                          help="Destination spine (\"legacy\" or a skeleton-hash prefix)")
    transfer.add_argument("--labels", action="append", default=None,
                          help="Restrict to these insert labels (repeatable; default: every "
                               "labeled wordless insert)")
    transfer.add_argument("--tolerance", type=float, default=0.05,
                          help="Duplicate-guard window in seconds: a same-label destination "
                               "insert this close counts as already transferred")
    transfer.add_argument("--dry-run", action="store_true",
                          help="Print the transfer plan; commit nothing")
    transfer.add_argument("--actor", default="human", help="Actor recorded on the transferred inserts")
    transfer.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")
    return parser


def load_capabilities(
    manager: CapabilityManager,                      # Freshly constructed manager
    instance_ids: List[str],                     # Capability names to load, in order
    configs: Optional[Dict[str, Dict]] = None,   # Per-capability load-time config (e.g. graph db_path)
) -> None:
    """Discover manifests + load each capability, passing per-capability config (CR-2 caller-wins)."""
    configs = configs or {}
    manager.discover_manifests()
    discovered = {m.name: m for m in manager.discovered}
    for iid in instance_ids:
        meta = discovered.get(iid)
        if meta is None:
            raise SystemExit(
                f"capability {iid!r} not found in manifests "
                f"(discovered: {sorted(discovered)}) -- run cjm-ctl install-all first"
            )
        if not manager.load_capability(meta, config=configs.get(iid)):
            raise SystemExit(f"failed to load capability {iid!r}")
        logger.info(f"loaded {iid}" + (f" (db_path override)" if iid in configs else ""))


async def run_command(
    args: argparse.Namespace,  # Parsed args for the `run` subcommand
) -> int:  # Process exit code
    """Execute the `run` subcommand: correct a decomp manifest's committed spine."""
    # 5daadfc4 workspace: resolve BEFORE any substrate config loads; export so
    # the process tree (substrate config, capability workers) is workspace-scoped.
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    if args.manifests_dir is None:
        args.manifests_dir = (str(ws.substrate_data_dir / "manifests")
                              if ws is not None else ".cjm/manifests")
    manifest_path = str(Path(args.manifest).resolve())
    if not Path(manifest_path).exists():
        raise SystemExit(f"decomp manifest not found: {manifest_path}")

    decomp = load_decomp_manifest(manifest_path)
    graph_db_path = resolve_graph_db_path(decomp, args.graph_capability, override=args.graph_db_path)
    if not graph_db_path:
        raise SystemExit("could not resolve graph DB path from manifest; pass --graph-db-path explicitly")

    cfg = CorrectionConfig(
        graph_capability=args.graph_capability, graph_db_path=graph_db_path,
        actor=args.actor, assume_yes=args.yes, prune_empty=not args.no_prune,
        rendition_selector=args.rendition, skeleton_selector=args.skeleton,
    )

    manager = CapabilityManager(
        search_paths=[Path(args.manifests_dir)],
        sysmon_capability_name=args.sysmon_capability,
    )
    load_order = ([args.sysmon_capability] if args.sysmon_capability else []) + [cfg.graph_capability]
    # Point the graph worker at the decomp graph DB (the shared spine) via load-time config.
    load_capabilities(manager, load_order, configs={cfg.graph_capability: {"db_path": graph_db_path}})

    queue = JobQueue(deps=manager, sysmon_capability_name=args.sysmon_capability)
    await queue.start()
    try:
        manifest = await run_correction(
            manager, queue, cfg, manifest_path, graph_db_path,
            session_id=args.session, reopen=args.reopen,
        )
    finally:
        await queue.stop()
        for iid in reversed(load_order):
            try:
                manager.unload_capability(iid)
            except Exception as e:  # Best-effort teardown; never mask the run's outcome
                logger.warning(f"unload {iid} failed: {e}")

    out = (Path(args.output) if args.output
           else (ws.runs_dir if ws is not None else Path("runs")) / f"{manifest.run_id}.json")
    manifest.save(out, workspace=ws)
    n_sources = len(manifest.sources)
    n_pruned = sum(s.get("pruned", 0) for s in manifest.sources)
    n_flagged = sum(s.get("worklist_flagged", 0) for s in manifest.sources)
    print(f"correction manifest: {out}")
    print(f"sources: {n_sources}  worklist flagged: {n_flagged}  pruned: {n_pruned}")
    print(f"session: {manifest.session_id}")
    return 0


async def review_command(
    args: argparse.Namespace,  # Parsed args for the `review` subcommand
) -> int:  # Process exit code
    """Execute the `review` subcommand: interactive text corrections over the flagged worklist."""
    # 5daadfc4 workspace: same early resolution + export as run_command.
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    if args.manifests_dir is None:
        args.manifests_dir = (str(ws.substrate_data_dir / "manifests")
                              if ws is not None else ".cjm/manifests")
    manifest_path = str(Path(args.manifest).resolve())
    if not Path(manifest_path).exists():
        raise SystemExit(f"decomp manifest not found: {manifest_path}")
    decomp = load_decomp_manifest(manifest_path)
    graph_db_path = resolve_graph_db_path(decomp, args.graph_capability, override=args.graph_db_path)
    if not graph_db_path:
        raise SystemExit("could not resolve graph DB path from manifest; pass --graph-db-path explicitly")

    cfg = CorrectionConfig(graph_capability=args.graph_capability, graph_db_path=graph_db_path,
                           actor=args.actor, assume_yes=args.yes, prune_empty=False,
                           rendition_selector=args.rendition, skeleton_selector=args.skeleton)
    manager = CapabilityManager(search_paths=[Path(args.manifests_dir)], sysmon_capability_name=args.sysmon_capability)
    load_order = ([args.sysmon_capability] if args.sysmon_capability else []) + [cfg.graph_capability]
    load_capabilities(manager, load_order, configs={cfg.graph_capability: {"db_path": graph_db_path}})

    queue = JobQueue(deps=manager, sysmon_capability_name=args.sysmon_capability)
    await queue.start()
    try:
        manifest = await run_review(
            manager, queue, cfg, manifest_path, graph_db_path,
            session_id=args.session, reopen=args.reopen, max_items=args.review_max,
        )
    finally:
        await queue.stop()
        for iid in reversed(load_order):
            try:
                manager.unload_capability(iid)
            except Exception as e:  # Best-effort teardown; never mask the run's outcome
                logger.warning(f"unload {iid} failed: {e}")

    out = (Path(args.output) if args.output
           else (ws.runs_dir if ws is not None else Path("runs")) / f"{manifest.run_id}.json")
    manifest.save(out, workspace=ws)
    n_corr = sum(s.get("corrected", 0) for s in manifest.sources)
    n_active = sum(s.get("active_corrections", 0) for s in manifest.sources)
    print(f"correction manifest: {out}")
    print(f"sources: {len(manifest.sources)}  corrected: {n_corr}  active corrections: {n_active}")
    print(f"session: {manifest.session_id}")
    return 0


def main(
    argv: Optional[List[str]] = None,  # Argument list override (None = sys.argv)
) -> int:  # Process exit code
    """CLI entry point (console script: `cjm-transcript-correction-core`)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    )
    if args.command == "run":
        return asyncio.run(run_command(args))
    if args.command == "review":
        return asyncio.run(review_command(args))
    if args.command == "stats":
        return asyncio.run(stats_command(args))
    if args.command == "gate":
        return asyncio.run(gate_command(args))
    if args.command == "extract":
        return asyncio.run(extract_command(args))
    if args.command == "bench":
        return asyncio.run(bench_command(args))
    if args.command == "transfer-wordless":
        return asyncio.run(transfer_command(args))
    raise SystemExit(f"unknown command: {args.command}")


async def stats_command(
    args: argparse.Namespace,  # Parsed args for the `stats` subcommand
) -> int:  # Process exit code
    """Execute the `stats` subcommand: flywheel accounting over the shared graph.

    The manual-tally retirement (drive ask 2026-07-27): per Source, the ACTIVE
    overlay folds into counts — labeled insert spans, split boundaries, open
    marks by class, op totals — plus the session purpose mix so dev-noise is
    visible; --genuine-only cuts the counts to genuine-session corrections
    (the DEC c86714a4 tag), --label spotlights one class corpus-wide (the
    "how many active inhale spans?" one-liner)."""
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    if args.manifests_dir is None:
        args.manifests_dir = (str(ws.substrate_data_dir / "manifests")
                              if ws is not None else ".cjm/manifests")
    manager = CapabilityManager(search_paths=[Path(args.manifests_dir)])
    configs = ({args.graph_capability: {"db_path": args.graph_db_path}}
               if args.graph_db_path else None)
    load_capabilities(manager, [args.graph_capability], configs=configs)
    queue = JobQueue(deps=manager)
    await queue.start()
    try:
        res = await graph_task(queue, args.graph_capability, "query_nodes",
                               query=NodeQuery(label="Source", project=["title"]).to_dict())
        sources = [(r["id"], str(r.get("title") or "")) for r in (res.rows or [])]
        if args.source:
            needle = args.source.lower()
            sources = [(i, t) for i, t in sources
                       if i == args.source or needle in t.lower()]
        if not sources:
            print("no matching Source nodes")
            return 1
        sres = await graph_task(queue, args.graph_capability, "query_nodes",
                                query=NodeQuery(label="CorrectionSession").to_dict())
        genuine_ids = set()
        purpose_of: Dict[str, str] = {}
        for n in (sres.nodes or []):
            d = n.to_dict() if hasattr(n, "to_dict") else n
            purpose = str((d.get("properties") or {}).get("purpose") or "genuine")
            purpose_of[d["id"]] = purpose
            if purpose == "genuine":
                genuine_ids.add(d["id"])
        totals: Dict[str, Dict[str, int]] = {"insert_labels": {}, "mark_classes": {}}
        total_splits = 0
        for sid, title in sources:
            corrections, superseded = await load_source_corrections(
                queue, args.graph_capability, sid)
            counted = corrections
            if args.genuine_only:
                # Purpose scopes the COUNTS; the superseded set stays global —
                # supersession is spine state whoever committed it.
                counted = [c for c in corrections if c.get("session_id") in genuine_ids]
            st = correction_stats(counted, superseded)
            mix: Dict[str, int] = {}
            for c in corrections:
                p = purpose_of.get(str(c.get("session_id")), "genuine")
                mix[p] = mix.get(p, 0) + 1
            print(f"== {title or sid}  ({sid}) ==")
            print("  ops by session purpose: "
                  + (" ".join(f"{k}x{v}" for k, v in sorted(mix.items())) or "none"))
            print(f"  active corrections: {st['active']}  ("
                  + (" ".join(f"{k}={v}" for k, v in sorted(st["ops"].items())) or "none") + ")")
            print(f"  splits: {st['splits']}")
            print("  labeled inserts: "
                  + (" · ".join(f"{k}x{v}" for k, v in sorted(st["insert_labels"].items())) or "none"))
            print(f"  open marks ({st['open_marks']}): "
                  + (" · ".join(f"{k}x{v}" for k, v in sorted(st["mark_classes"].items())) or "none"))
            total_splits += st["splits"]
            for k, v in st["insert_labels"].items():
                totals["insert_labels"][k] = totals["insert_labels"].get(k, 0) + v
            for k, v in st["mark_classes"].items():
                totals["mark_classes"][k] = totals["mark_classes"].get(k, 0) + v
        print(f"== TOTALS ({len(sources)} source(s)) ==")
        print(f"  splits: {total_splits}")
        print("  labeled inserts: "
              + (" · ".join(f"{k}x{v}" for k, v in sorted(totals["insert_labels"].items())) or "none"))
        print("  open marks: "
              + (" · ".join(f"{k}x{v}" for k, v in sorted(totals["mark_classes"].items())) or "none"))
        if args.label:
            ins = totals["insert_labels"].get(args.label, 0)
            mk = totals["mark_classes"].get(args.label, 0)
            print(f"  active {args.label!r} spans: {ins + mk} (inserts {ins} + marks {mk})")
    finally:
        await queue.stop()
        try:
            manager.unload_capability(args.graph_capability)
        except Exception as e:  # Best-effort teardown; never mask the stats outcome
            logger.warning(f"unload {args.graph_capability} failed: {e}")
    return 0


async def gate_command(
    args: argparse.Namespace,  # Parsed args for the `gate` subcommand
) -> int:  # Process exit code
    """Execute the `gate` subcommand: show or assert per-spine extraction gates.

    The CLI-first flywheel surface (DEC a5aa43b9) for DEC 8e05b87b: without
    --status it prints every matched Source's live gate per coexisting spine
    (absent assertion = the in_progress default, no watermark); with --status
    (+ optional --annotated-through, "end" = the spine's last segment end) it
    commits ONE journaled assertion on ONE spine — latest-wins, full history
    kept. The watermark is asserted EXPLICITLY, never derived from op
    positions (DEC 8e05b87b)."""
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    if args.manifests_dir is None:
        args.manifests_dir = (str(ws.substrate_data_dir / "manifests")
                              if ws is not None else ".cjm/manifests")
    manager = CapabilityManager(search_paths=[Path(args.manifests_dir)])
    configs = ({args.graph_capability: {"db_path": args.graph_db_path}}
               if args.graph_db_path else None)
    load_capabilities(manager, [args.graph_capability], configs=configs)
    queue = JobQueue(deps=manager)
    await queue.start()
    try:
        res = await graph_task(queue, args.graph_capability, "query_nodes",
                               query=NodeQuery(label="Source", project=["title"]).to_dict())
        sources = [(r["id"], str(r.get("title") or "")) for r in (res.rows or [])]
        if args.source:
            needle = args.source.lower()
            sources = [(i, t) for i, t in sources
                       if i == args.source or needle in t.lower()]
        if not sources:
            print("no matching Source nodes")
            return 1

        if args.status is None:
            for sid, title in sources:
                spines = await list_source_spines(queue, args.graph_capability, sid,
                                                  rendition_selector=args.rendition)
                gates = await load_extraction_gates(queue, args.graph_capability, sid)
                print(f"== {title or sid}  ({sid}) ==")
                if not spines:
                    print("  (no decomposed spine)")
                for sp in spines:
                    h = sp.get("skeleton_hash")
                    print(f"  spine {_spine_tag(h)} ({sp.get('segments', 0)} segs): "
                          + _gate_line(gates.get(h)))
                for h, g in gates.items():
                    if not any(sp.get("skeleton_hash") == h for sp in spines):
                        print(f"  spine {_spine_tag(h)} (gone?): " + _gate_line(g))
            return 0

        if len(sources) != 1:
            print(f"--status needs exactly ONE source (matched {len(sources)}) — "
                  "narrow --source")
            return 1
        sid, title = sources[0]
        spines = await list_source_spines(queue, args.graph_capability, sid,
                                          rendition_selector=args.rendition)
        skeleton_hash = skeleton_hash_for(spines, args.skeleton)
        gates = await load_extraction_gates(queue, args.graph_capability, sid)
        current = gates.get(skeleton_hash)
        wm_arg = args.annotated_through
        if wm_arg is None:
            watermark = (current or {}).get("annotated_through")
        elif str(wm_arg).strip().lower() == "end":
            segs = await load_source_segments(queue, args.graph_capability, sid,
                                              rendition_selector=args.rendition,
                                              skeleton_selector=args.skeleton)
            ends = [float(s.end_time) for s in segs if s.end_time is not None]
            if not ends:
                print("--annotated-through end: the spine has no timed segments")
                return 1
            watermark = max(ends)
        else:
            watermark = float(wm_arg)
        db = args.graph_db_path or (
            (manager.instances[args.graph_capability].config or {}).get("db_path"))
        gate_id = await commit_extraction_gate(
            queue, args.graph_capability, sid, skeleton_hash, args.status, watermark,
            actor=args.actor,
            journal_path=(sidecar_journal_path(db) if db else None))
        wm_txt = f"{float(watermark):.1f}s" if watermark is not None else "none"
        print(f"gate asserted on {title or sid} · spine {_spine_tag(skeleton_hash)}: "
              f"{args.status} · annotated_through {wm_txt}  ({gate_id})")
    finally:
        await queue.stop()
        try:
            manager.unload_capability(args.graph_capability)
        except Exception as e:  # Best-effort teardown; never mask the gate outcome
            logger.warning(f"unload {args.graph_capability} failed: {e}")
    return 0


def _spine_tag(
    skeleton_hash: Optional[str],  # A spine identity (None = the legacy pre-split spine)
) -> str:  # Display handle for the spine
    """Short display handle for a spine identity (the spine_label naming)."""
    return "legacy" if skeleton_hash is None else str(skeleton_hash).split(":")[-1][:8]


def _gate_line(
    gate: Optional[Dict],  # The live gate assertion property dict, or None = never asserted
) -> str:  # One-line gate render
    """Render one spine's live gate state (absent assertion = the in_progress
    default with no watermark — DEC 8e05b87b's default, never a stored one)."""
    if gate is None:
        return "in_progress (default) · annotated_through: none"
    wm = gate.get("annotated_through")
    return (f"{gate.get('extraction_status')}"
            f" · annotated_through: {f'{float(wm):.1f}s' if wm is not None else 'none'}"
            f" · asserted {time.strftime('%Y-%m-%d %H:%M', time.localtime(float(gate.get('created_at') or 0.0)))}"
            f" by {gate.get('actor')}")


async def extract_command(
    args: argparse.Namespace,  # Parsed args for the `extract` subcommand
) -> int:  # Process exit code
    """Execute the `extract` subcommand: fold the gated overlay into a manifested dataset.

    Flywheel build leg 2 (DECs d02a38d4 + 16159e09 + a5883992 + 8e05b87b): a
    deterministic core-side fold, sibling of stats — per gated spine, labeled
    insert spans become examples (genuine-purpose sessions by default), speech
    + negative regions derive below the annotated_through watermark, and the
    dataset lands WORKSPACE-LOCAL under datasets/<dataset_id>/ (a regenerable
    projection of the correction journal) with a DatasetManifest recording the
    config, the consumed journal family, split/augmentation policy as DATA,
    the observed open class vocabulary, and every spine's gate state at
    extraction time."""
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    if args.manifests_dir is None:
        args.manifests_dir = (str(ws.substrate_data_dir / "manifests")
                              if ws is not None else ".cjm/manifests")
    manager = CapabilityManager(search_paths=[Path(args.manifests_dir)])
    configs = ({args.graph_capability: {"db_path": args.graph_db_path}}
               if args.graph_db_path else None)
    load_capabilities(manager, [args.graph_capability], configs=configs)
    queue = JobQueue(deps=manager)
    await queue.start()
    try:
        res = await graph_task(queue, args.graph_capability, "query_nodes",
                               query=NodeQuery(label="Source").to_dict())
        sources = []
        for n in (res.nodes or []):
            d = n.to_dict() if hasattr(n, "to_dict") else n
            props = d.get("properties") or {}
            hashes = [r.get("content_hash") for r in (d.get("sources") or [])
                      if isinstance(r, dict) and r.get("content_hash")]
            sources.append({"id": d["id"], "title": str(props.get("title") or ""),
                            "path": props.get("path"),
                            "content_hash": (hashes[0] if hashes else None)})
        if args.source:
            needle = args.source.lower()
            sources = [s for s in sources
                       if s["id"] == args.source or needle in s["title"].lower()]
        if not sources:
            print("no matching Source nodes")
            return 1
        purposes = list(args.include_purpose or ["genuine"])
        sres = await graph_task(queue, args.graph_capability, "query_nodes",
                                query=NodeQuery(label="CorrectionSession").to_dict())
        include_ids = set()
        for n in (sres.nodes or []):
            d = n.to_dict() if hasattr(n, "to_dict") else n
            purpose = str((d.get("properties") or {}).get("purpose") or "genuine")
            if purpose in purposes:
                include_ids.add(d["id"])

        dataset_id = new_dataset_id()
        created_at = time.time()
        out_root = (Path(args.output_dir) if args.output_dir
                    else (ws.root / "datasets" if ws is not None else Path("datasets")))
        ddir = out_root / dataset_id
        events: List[Dict] = []
        regions: List[Dict] = []
        spine_records: List[Dict] = []
        vocab: Dict[str, int] = {}
        skipped_totals: Dict[str, int] = {}
        for src in sources:
            corrections, superseded = await load_source_corrections(
                queue, args.graph_capability, src["id"])
            spines = await list_source_spines(queue, args.graph_capability, src["id"],
                                              rendition_selector=args.rendition)
            gates = await load_extraction_gates(queue, args.graph_capability, src["id"])
            print(f"== {src['title'] or src['id']}  ({src['id']}) ==")
            if not spines:
                print("  (no decomposed spine)")
            for sp in spines:
                h = sp.get("skeleton_hash")
                gate = gates.get(h)
                status = str((gate or {}).get("extraction_status") or "in_progress")
                extractable = (status != "excluded"
                               and (gate or {}).get("annotated_through") is not None)
                # Gate first: an ineligible spine never pays the full-spine read.
                segs = (await load_source_segments(
                    queue, args.graph_capability, src["id"],
                    rendition_selector=args.rendition,
                    skeleton_selector=("legacy" if h is None else h))
                    if extractable else [])
                r = extract_spine_dataset(segs, corrections, superseded, gate,
                                          include_session_ids=include_ids)
                wm_txt = (f"{r['watermark']:.1f}s" if r["watermark"] is not None else "none")
                rec = {"source_id": src["id"], "source_title": src["title"],
                       "skeleton_hash": h, "extraction_status": r["status"],
                       "annotated_through": r["watermark"], "eligible": r["eligible"],
                       "examples": len(r["examples"]), "speech_regions": len(r["speech"]),
                       "negative_regions": len(r["negatives"]), "skipped": r["skipped"]}
                spine_records.append(rec)
                if not r["eligible"]:
                    print(f"  spine {_spine_tag(h)}: {r['status']} · annotated_through "
                          f"{wm_txt} — not extractable")
                    continue
                print(f"  spine {_spine_tag(h)}: {r['status']} @ {wm_txt} — "
                      f"examples {len(r['examples'])} · speech {len(r['speech'])} · "
                      f"negatives {len(r['negatives'])}"
                      + (f" · skipped {r['skipped']}" if r["skipped"] else ""))
                for e in r["examples"]:
                    vocab[e["label"]] = vocab.get(e["label"], 0) + 1
                    events.append({
                        "kind": "labeled_span", "source_id": src["id"],
                        "source_title": src["title"], "source_path": src["path"],
                        "source_content_hash": src["content_hash"],
                        "skeleton_hash": h, "insert_id": e["insert_id"],
                        "label": e["label"], "text": e["text"], "speech": e["speech"],
                        "start_time": e["start_time"], "end_time": e["end_time"],
                        "split": "train",
                        "provenance": {"tag": "real", "sessions": e["session_ids"],
                                       "op_ids": e["op_ids"]}})
                for s in r["speech"]:
                    regions.append({"kind": "speech", "source_id": src["id"],
                                    "skeleton_hash": h, **s})
                for g in r["negatives"]:
                    regions.append({"kind": "negative", "source_id": src["id"],
                                    "skeleton_hash": h, **g})
                for k, v in r["skipped"].items():
                    skipped_totals[k] = skipped_totals.get(k, 0) + v

        ddir.mkdir(parents=True, exist_ok=True)
        (ddir / "events.jsonl").write_text(
            "".join(json.dumps(relativize_recorded(row, ws)) + "\n" for row in events))
        (ddir / "regions.jsonl").write_text(
            "".join(json.dumps(relativize_recorded(row, ws)) + "\n" for row in regions))
        db = args.graph_db_path or (
            (manager.instances[args.graph_capability].config or {}).get("db_path"))
        stem = sidecar_journal_path(db)[: -len(".jsonl")] if db else None
        journals = (sorted(str(p) for p in Path(db).parent.glob(Path(stem).name + "*.jsonl"))
                    if db else [])
        manifest = DatasetManifest(
            dataset_id=dataset_id, created_at=created_at,
            config={"graph_capability": args.graph_capability,
                    "source": args.source, "rendition": args.rendition,
                    "include_purposes": purposes},
            graph_db_path=str(db or ""), journals=journals,
            session_purpose_policy={"include": purposes, "unset_means": "genuine"},
            split_policy={"policy": "tail-reservation",
                          "train": "annotated head — spans starting below each spine's "
                                   "annotated_through watermark",
                          "bench": "reserved tail above the watermark — NEVER extracted; "
                                   "the live-bench half of DEC 8cf12c22"},
            augmentation_policy={"policy": "none",
                                 "provenance_tags": ["real", "augmented", "spliced",
                                                     "synthetic"],
                                 "note": "rungs unpulled unless a finetune is "
                                         "data-gated (DEC 03d207cf)"},
            class_vocabulary=vocab, spines=spine_records,
            files={"events": "events.jsonl", "regions": "regions.jsonl"},
            counts={"examples": len(events),
                    "speech_regions": sum(1 for r in regions if r["kind"] == "speech"),
                    "negative_regions": sum(1 for r in regions if r["kind"] == "negative"),
                    **{f"skipped_{k}": v for k, v in sorted(skipped_totals.items())}})
        out = manifest.save(ddir / "manifest.json", workspace=ws)
        print(f"== DATASET {dataset_id} ==")
        print("  classes: "
              + (" · ".join(f"{k}x{v}" for k, v in sorted(vocab.items())) or "none"))
        print(f"  examples: {len(events)}  regions: {len(regions)}  "
              f"spines: {sum(1 for r in spine_records if r['eligible'])} extractable "
              f"of {len(spine_records)}")
        print(f"  manifest: {out}")
    finally:
        await queue.stop()
        try:
            manager.unload_capability(args.graph_capability)
        except Exception as e:  # Best-effort teardown; never mask the extract outcome
            logger.warning(f"unload {args.graph_capability} failed: {e}")
    return 0


async def bench_command(
    args: argparse.Namespace,  # Parsed args for the `bench` subcommand
) -> int:  # Process exit code
    """Execute the `bench` subcommand: the reserved-tail verdict join.

    Leg 4's measurement half (DECs 8e05b87b + 8cf12c22): verdicts are NEVER
    stored — this derives them fresh by joining the proposal set (durable
    inference-run output) against final spine state: materialized within
    tolerance = accepted, moved = edited, absent = rejected, and active
    inserts no proposal covered = the model's misses. Wall-clock derives from
    the sidecar journal's op timestamps for the sessions that touched the
    window."""
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    if args.manifests_dir is None:
        args.manifests_dir = (str(ws.substrate_data_dir / "manifests")
                              if ws is not None else ".cjm/manifests")

    # Resolve the proposal set FIRST — it names the source/spine to bench.
    pset = None
    if args.proposals:
        mp = Path(args.proposals)
        if mp.is_dir():
            mp = mp / "manifest.json"
        m = json.loads(mp.read_text())
        if m.get("format") != EVENT_PROPOSAL_SET_FORMAT:
            print(f"not a proposal-set manifest: {mp} (format {m.get('format')!r})")
            return 1
        data_file = mp.parent / str((m.get("files") or {}).get("proposals") or "proposals.jsonl")
        proposals = [json.loads(line) for line in data_file.read_text().splitlines() if line.strip()]
        pset = {"manifest": m, "proposals": proposals}
    elif ws is not None and args.source:
        pset = load_event_proposal_set(str(ws.root), source_id=args.source)
    if pset is None:
        print("no proposal set: pass --proposals, or --source with a set under "
              "<workspace>/proposals/ recording that source_id")
        return 1
    manifest = pset["manifest"]
    window = (float((manifest.get("window") or {}).get("start") or 0.0),
              (manifest.get("window") or {}).get("end"))
    source_id = args.source or (manifest.get("source") or {}).get("source_id")
    skeleton = getattr(args, "skeleton", None) or (manifest.get("source") or {}).get("skeleton_hash")
    if skeleton is None and ws is not None:
        # Respine chain (DEC 6cc10fb7): a propset consumed as the CUT AUTHORITY
        # predates its spine, so it cannot record a skeleton_hash — but the
        # CONSUMING decomp manifest names this set (event_propset_id) and its
        # skeleton_config_hash IS the spine the drive walked. Latest consumer
        # wins (run ids sort by time). Without this, the join silently lands
        # on the LEGACY spine and every proposal derives REJECTED.
        set_id = manifest.get("proposal_set_id")
        for rm in sorted((ws.root / "runs").glob("*.json")):
            try:
                d = json.loads(rm.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if d.get("format") == "cjm-transcript-decomp-core/run-manifest" \
                    and d.get("event_propset_id") == set_id \
                    and d.get("skeleton_config_hash"):
                skeleton = d["skeleton_config_hash"]
        if skeleton:
            print(f"  spine: {skeleton[:24]}… (resolved via the consuming decomp manifest)")
    if not source_id:
        print("proposal set records no source_id — pass --source")
        return 1

    manager = CapabilityManager(search_paths=[Path(args.manifests_dir)])
    configs = ({args.graph_capability: {"db_path": args.graph_db_path}}
               if args.graph_db_path else None)
    load_capabilities(manager, [args.graph_capability], configs=configs)
    queue = JobQueue(deps=manager)
    await queue.start()
    try:
        # --source may be a substring — resolve against Source nodes.
        res = await graph_task(queue, args.graph_capability, "query_nodes",
                               query=NodeQuery(label="Source", project=["title"]).to_dict())
        sources = [(r["id"], str(r.get("title") or "")) for r in (res.rows or [])]
        needle = source_id.lower()
        picked = [(i, t) for i, t in sources if i == source_id or needle in t.lower()]
        if len(picked) != 1:
            print(f"need exactly one Source (matched {len(picked)}) for {source_id!r}")
            return 1
        source_id, title = picked[0]

        purposes = list(args.include_purpose or ["genuine"])
        sres = await graph_task(queue, args.graph_capability, "query_nodes",
                                query=NodeQuery(label="CorrectionSession").to_dict())
        include_ids = set()
        for n in (sres.nodes or []):
            d = n.to_dict() if hasattr(n, "to_dict") else n
            purpose = str((d.get("properties") or {}).get("purpose") or "genuine")
            if purpose in purposes:
                include_ids.add(d["id"])

        corrections, superseded = await load_source_corrections(
            queue, args.graph_capability, source_id)
        segments = await load_source_segments(
            queue, args.graph_capability, source_id,
            rendition_selector=args.rendition,
            skeleton_selector=("legacy" if skeleton is None else skeleton))
        spans = [r for r in labeled_insert_spans(segments, corrections, superseded)
                 if r.get("session_id") in include_ids]
        report = bench_event_proposals(pset["proposals"], spans, window,
                                       tolerance=args.tolerance)

        # Wall-clock: sidecar-journal op timestamps for the sessions whose
        # inserts the join matched (the bench pass's real duration).
        db = args.graph_db_path or (
            (manager.instances[args.graph_capability].config or {}).get("db_path"))
        bench_sessions = {r.get("session_id")
                          for r in spans if r.get("session_id")} & include_ids
        wall: Dict[str, Dict[str, float]] = {}
        jp = Path(sidecar_journal_path(db)) if db else None
        if jp is not None and jp.is_file():
            for line in jp.read_text().splitlines():
                try:
                    op = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = str(op.get("session_id") or "")
                ts = op.get("ts")
                if sid in bench_sessions and ts is not None:
                    w = wall.setdefault(sid, {"first": float(ts), "last": float(ts), "ops": 0})
                    w["first"] = min(w["first"], float(ts))
                    w["last"] = max(w["last"], float(ts))
                    w["ops"] += 1

        c = report["counts"]
        w_end = f"{window[1]:.1f}" if window[1] is not None else "end"
        print(f"== BENCH {manifest.get('proposal_set_id')} · {title or source_id} ==")
        print(f"  model: {manifest.get('training_run_id')} · window [{window[0]:.1f}, {w_end}]s "
              f"· tolerance {args.tolerance}s")
        print(f"  proposals {c['proposals']}: accepted {c['accepted']} · edited {c['edited']} "
              f"· rejected {c['rejected']}  ·  missed (manual inserts) {c['missed']}")
        if report["rates"]:
            print("  rates: " + " · ".join(f"{k} {v:.1%}" for k, v in report["rates"].items()))
        for sid, w in sorted(wall.items()):
            print(f"  session {sid[:8]}: {w['ops']} ops over {w['last'] - w['first']:.0f}s wall-clock")
        if not wall:
            print("  wall-clock: no journaled bench-session ops yet")
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(
                {"proposal_set_id": manifest.get("proposal_set_id"),
                 "training_run_id": manifest.get("training_run_id"),
                 "source_id": source_id, "window": {"start": window[0], "end": window[1]},
                 "tolerance": args.tolerance, "report": report,
                 "wall_clock": wall}, indent=2))
            print(f"  report: {out}")
    finally:
        await queue.stop()
        try:
            manager.unload_capability(args.graph_capability)
        except Exception as e:  # Best-effort teardown; never mask the bench outcome
            logger.warning(f"unload {args.graph_capability} failed: {e}")
    return 0


async def transfer_command(
    args: argparse.Namespace,  # Parsed args for the `transfer-wordless` subcommand
) -> int:  # Process exit code
    """Execute `transfer-wordless`: replay wordless event inserts across sibling spines.

    dea104ba — the layering principle's first mechanical expression: accepted
    wordless inserts are SOURCE-truth ((span, label) pairs with zero text
    dependency), so a respine (re-transcription, FA/VAD upgrade) must not lose
    the human event layer. Donor spans come from the FROM spine's EFFECTIVE
    projection (time nudges applied — the heard span, not the born one) and
    keep their walked rank; placement on the destination spine anchors by
    source time between its layer-0 segments (the projection re-sorts by
    effective times, so a span the new carve disagrees with stays honest).
    What does NOT transfer, by design: boundary shifts, text edits, and any
    insert that GAINED text (word-bearing = spine-truth — new transcript, new
    FA). Idempotent: a destination insert with the same label within
    --tolerance of a donor span counts as already present and is skipped.
    Commits require --graph-db-path (the flywheel journal rides the db
    sidecar — journaling stays default-on); --dry-run needs neither."""
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    if args.manifests_dir is None:
        args.manifests_dir = (str(ws.substrate_data_dir / "manifests")
                              if ws is not None else ".cjm/manifests")
    if not args.dry_run and not args.graph_db_path:
        raise SystemExit("committing needs --graph-db-path (the journal sidecar hangs off it); "
                         "use --dry-run to plan without one")
    manager = CapabilityManager(search_paths=[Path(args.manifests_dir)])
    configs = ({args.graph_capability: {"db_path": args.graph_db_path}}
               if args.graph_db_path else None)
    load_capabilities(manager, [args.graph_capability], configs=configs)
    queue = JobQueue(deps=manager)
    await queue.start()
    try:
        res = await graph_task(queue, args.graph_capability, "query_nodes",
                               query=NodeQuery(label="Source", project=["title"]).to_dict())
        needle = args.source.lower()
        matches = [(r["id"], str(r.get("title") or "")) for r in (res.rows or [])
                   if r["id"] == args.source or needle in str(r.get("title") or "").lower()]
        if len(matches) != 1:
            raise SystemExit(f"--source matched {len(matches)} Source nodes "
                             f"({[t for _, t in matches]}); need exactly one")
        sid, title = matches[0]
        print(f"source: {title or sid}  ({sid})")

        segs_from = await load_source_segments(
            queue, args.graph_capability, sid,
            rendition_selector=args.rendition, skeleton_selector=args.from_skeleton)
        segs_to = await load_source_segments(
            queue, args.graph_capability, sid,
            rendition_selector=args.rendition, skeleton_selector=args.to_skeleton)
        if not segs_from or not segs_to:
            raise SystemExit(f"empty spine (from={len(segs_from)} to={len(segs_to)} segments)")
        if {s.id for s in segs_from} == {s.id for s in segs_to}:
            raise SystemExit("--from-skeleton and --to-skeleton resolved to the SAME spine")
        print(f"spines: from {len(segs_from)} segs -> to {len(segs_to)} segs")

        corrections, superseded = await load_source_corrections(queue, args.graph_capability, sid)
        active = [c for c in corrections
                  if c["id"] not in superseded and c.get("status") != "proposed"]
        insert_meta = {c["id"]: (c.get("payload") or {}) for c in active
                       if c.get("correction_type") == "insertion"
                       and (c.get("payload") or {}).get("operation") == "chunk_insert"}

        # Donors: labeled + effectively wordless units of the FROM projection.
        eff_from = project_effective_spine(segs_from, active)
        donors, word_bearing = [], 0
        for u in eff_from:
            meta = insert_meta.get(u.id)
            if meta is None:
                continue
            label = meta.get("label")
            if not label or (args.labels and label not in args.labels):
                continue
            if (u.text or "").strip():
                word_bearing += 1
                continue
            if u.start_time is None or u.end_time is None:
                continue
            donors.append({"start": float(u.start_time), "end": float(u.end_time),
                           "label": str(label), "rank": float(meta.get("rank") or 0.0)})

        # Existing destination events (idempotency guard) from the TO projection.
        eff_to = project_effective_spine(segs_to, active)
        existing = [(str(insert_meta[u.id].get("label")), float(u.start_time))
                    for u in eff_to
                    if u.id in insert_meta and insert_meta[u.id].get("label")
                    and u.start_time is not None]

        to_l0 = sorted((s for s in segs_to if s.start_time is not None),
                       key=lambda s: s.index)
        plan, dups, unanchored = [], 0, 0
        for d in donors:
            if any(lb == d["label"] and abs(st - d["start"]) <= args.tolerance
                   for lb, st in existing):
                dups += 1
                continue
            after = None
            pos = -1
            for i, s in enumerate(to_l0):
                if float(s.start_time) <= d["start"]:
                    after, pos = s, i
                else:
                    break
            if after is None:
                unanchored += 1
                continue
            before = to_l0[pos + 1] if pos + 1 < len(to_l0) else None
            plan.append({**d, "after_id": after.id,
                         "before_id": before.id if before else None})

        by_label: Dict[str, int] = {}
        for p in plan:
            by_label[p["label"]] = by_label.get(p["label"], 0) + 1
        print(f"donors: {len(donors)}  ->  transfer {len(plan)}  "
              f"(dup-skip {dups} · word-bearing-skip {word_bearing} · unanchored {unanchored})")
        print("by label: " + (" · ".join(f"{k}x{v}" for k, v in sorted(by_label.items())) or "none"))
        if args.dry_run:
            for p in plan[:10]:
                print(f"  {p['label']:>16} {p['start']:9.3f}-{p['end']:9.3f}s "
                      f"after {p['after_id'][:8]}")
            if len(plan) > 10:
                print(f"  … {len(plan) - 10} more")
            print("dry run — nothing committed")
            return 0

        jp = sidecar_journal_path(args.graph_db_path)
        sess = await start_session(queue, args.graph_capability, [sid],
                                   journal_path=jp, purpose="wordless-transfer")
        for p in plan:
            await commit_chunk_insert_correction(
                queue, args.graph_capability, sid, p["after_id"],
                p["start"], p["end"], sess.id,
                before_segment_id=p["before_id"], label=p["label"], rank=p["rank"],
                actor=args.actor, journal_path=jp)
        await set_session_status(queue, args.graph_capability, sess.id,
                                 "completed", journal_path=jp)
        print(f"transferred {len(plan)} event inserts (session {sess.id})")
    finally:
        await queue.stop()
        try:
            manager.unload_capability(args.graph_capability)
        except Exception as e:  # Best-effort teardown; never mask the transfer outcome
            logger.warning(f"unload {args.graph_capability} failed: {e}")
    return 0
