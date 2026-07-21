"""The CLI driver — the correction core's first (and currently only) frontend. run <decomp-manifest> corrects the committed spine in the decomp graph DB, pointing the graph worker at that shared DB via load-time config, with optional session resume/reopen; review runs the interactive text-correction loop (the cross-transcriber diff is intra-graph since stage 5)."""

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate.core.queue import JobQueue
from cjm_substrate.core.workspace import resolve_workspace
from cjm_transcript_correction_core.models import CorrectionConfig
from cjm_transcript_correction_core.pipeline import (load_decomp_manifest, resolve_graph_db_path,
                                                     run_correction, run_review)

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
        rendition_selector=args.rendition,
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
                           rendition_selector=args.rendition)
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
    raise SystemExit(f"unknown command: {args.command}")
