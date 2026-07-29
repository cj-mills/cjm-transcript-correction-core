"""The CLI driver — the correction core's first (and currently only) frontend. run <decomp-manifest> corrects the committed spine in the decomp graph DB, pointing the graph worker at that shared DB via load-time config, with optional session resume/reopen; review runs the interactive text-correction loop (the cross-transcriber diff is intra-graph since stage 5)."""

import argparse
import asyncio
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
from cjm_substrate.core.workspace import resolve_workspace
from cjm_transcript_correction_core.graph import (commit_extraction_gate, correction_stats,
                                                  list_source_spines, load_extraction_gates,
                                                  load_source_corrections, load_source_segments,
                                                  skeleton_hash_for)
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
