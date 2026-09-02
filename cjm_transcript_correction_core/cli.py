"""The CLI driver — the correction core's first (and currently only) frontend. run <decomp-manifest> corrects the committed spine in the decomp graph DB, pointing the graph worker at that shared DB via load-time config, with optional session resume/reopen; review runs the interactive text-correction loop (the cross-transcriber diff is intra-graph since stage 5)."""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cjm_context_graph_layer.journal import sidecar_journal_path
from cjm_context_graph_layer.ops import graph_task
from cjm_context_graph_primitives.query import NodeQuery
from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate.core.queue import JobQueue
from cjm_substrate.core.workspace import relativize_recorded, resolve_workspace
from cjm_transcript_correction_core.graph import (bench_event_proposals,
                                                  commit_chunk_insert_correction,
                                                  commit_chunk_split_correction,
                                                  commit_extraction_gate, commit_mark_correction,
                                                  commit_stratum_correction,
                                                  commit_stratum_retraction, correction_stats,
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
from cjm_transcript_correction_core.strata import (active_strata, bench_filter_proposals,
                                                   build_filter_pack, FILTER_LANE,
                                                   FILTER_PACK_FORMAT, load_filter_proposal_sets,
                                                   materialized_mark_ids,
                                                   pending_filter_proposals, proposals_from_rows,
                                                   render_filter_pack,
                                                   render_filter_propset_markdown,
                                                   validate_proposal_rows, write_filter_propset)

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
    transfer.add_argument("--no-splits", action="store_true",
                          help="Leave speaker-split boundaries behind (default: a split's "
                               "cut time transfers and the destination re-cuts its own "
                               "words there — 54aac7d3)")
    transfer.add_argument("--dry-run", action="store_true",
                          help="Print the transfer plan; commit nothing")
    transfer.add_argument("--actor", default="human", help="Actor recorded on the transferred inserts")
    transfer.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")

    export = sub.add_parser(
        "export-wordless-propset",
        help="Export one spine's effective wordless layer (accepted + nudged + "
             "manual — exactly the transfer-wordless donor set) as a proposal "
             "set a respine consumes as carve authority (f5d080b9 direction a)")
    export.add_argument("--manifests-dir", default=None,
                        help="Capability manifests directory (default: the workspace's "
                             ".cjm/manifests when one is active, else .cjm/manifests "
                             "under the cwd)")
    export.add_argument("--workspace", default=None,
                        help="Workspace root (default: CJM_WORKSPACE env, else upward walk from cwd)")
    export.add_argument("--graph-capability", default="cjm-capability-graph-sqlite",
                        help="Graph-storage capability name")
    export.add_argument("--graph-db-path", default=None,
                        help="Graph db path (reads only — default: the workspace "
                             "capability config)")
    export.add_argument("--source", required=True,
                        help="Source node id or title substring (exactly one match)")
    export.add_argument("--rendition", default=None,
                        help="Which AudioRendition spine when a source has more than one")
    export.add_argument("--from-skeleton", required=True,
                        help="The walked spine whose effective wordless layer exports "
                             "(\"legacy\" or a skeleton-hash prefix)")
    export.add_argument("--labels", action="append", default=None,
                        help="Restrict to these insert labels (repeatable; default: every "
                             "labeled wordless insert)")
    export.add_argument("--out-dir", default=None,
                        help="Proposal-set root directory (default: <workspace>/proposals)")
    export.add_argument("--dry-run", action="store_true",
                        help="Print the export plan; write nothing")
    export.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")

    scan = sub.add_parser(
        "scan-mishomed",
        help="Flag authoritative FA words stranded outside every chunk of a spine "
             "(96edc646: mis-homed text — carve-sliver + VAD-gap detector; the "
             "standing spine QA gate, reads only)")
    scan.add_argument("--manifests-dir", default=None,
                      help="Capability manifests directory (default: the workspace's "
                           ".cjm/manifests when one is active, else .cjm/manifests "
                           "under the cwd)")
    scan.add_argument("--workspace", default=None,
                      help="Workspace root (default: CJM_WORKSPACE env, else upward walk from cwd)")
    scan.add_argument("--graph-capability", default="cjm-capability-graph-sqlite",
                      help="Graph-storage capability name")
    scan.add_argument("--graph-db-path", default=None,
                      help="Graph db path (reads only — default: the workspace "
                           "capability config)")
    scan.add_argument("--source", required=True,
                      help="Source node id or title substring (exactly one match)")
    scan.add_argument("--rendition", default=None,
                      help="Which AudioRendition spine when a source has more than one")
    scan.add_argument("--skeleton", required=True,
                      help="Which spine to scan (\"legacy\" or a skeleton-hash prefix)")
    scan.add_argument("--fa-cache-db", default=None,
                      help="Forced-alignment cache db (default: the workspace's "
                           "qwen3-forced-aligner data dir)")
    scan.add_argument("--min-overlap", type=float, default=0.03,
                      help="Seconds of word/assigned-chunk overlap below which the word "
                           "counts as mis-homed (the fold's assignment is replicated; a "
                           "boundary-clipped word with real overlap is precision, not "
                           "mis-homing)")
    scan.add_argument("--strict", action="store_true",
                      help="Exit nonzero when any mis-homed word is found (CI/QA gate mode)")
    scan.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")

    # ---- the filtering lane (DECs 304fd984 + 9d4c0a38; pass 1 runs HEADLESS per
    # bc8dbbdd: pack -> proposer -> ingest -> confirm, journal-first) ----------
    fpack = sub.add_parser(
        "filter-pack",
        help="Write one spine window's effective text as a proposer PACK (json + the "
             "markdown brief any proposer reads — sub-agent, API, local model)")
    _add_graph_read_args(fpack)
    fpack.add_argument("--source", required=True,
                       help="Source node id or title substring (exactly one match)")
    fpack.add_argument("--rendition", default=None,
                       help="Which AudioRendition spine when a source has more than one")
    fpack.add_argument("--skeleton", default=None,
                       help="Which SKELETON spine (\"legacy\" or a hash prefix); default: auto "
                            "(errors when several coexist)")
    fpack.add_argument("--window", nargs=2, type=float, default=None, metavar=("START", "END"),
                       help="Source-seconds window to pack (default: the whole spine) — the "
                            "partitioning seam for long sources")
    fpack.add_argument("--out-dir", default=None,
                       help="Pack directory (default: <workspace>/packs)")
    fpack.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")

    fingest = sub.add_parser(
        "filter-ingest",
        help="Validate a proposer's JSONL rows against their pack and mint a filtering "
             "PROPOSAL SET (durable inference output; no graph writes)")
    fingest.add_argument("--pack", required=True, help="The pack json the proposer read")
    fingest.add_argument("--rows", required=True,
                         help="The proposer's output: JSONL rows (or a JSON array); code "
                              "fences and blank lines are tolerated")
    fingest.add_argument("--proposer", required=True,
                         help="Proposer name recorded as provenance (e.g. the sub-agent "
                              "brief name, an API model id)")
    fingest.add_argument("--proposer-kind", default="claude-code-subagent",
                         help="Proposer kind (claude-code-subagent | api | local-model | human)")
    fingest.add_argument("--model", default=None,
                         help="Model identity when known (recorded beside the kind)")
    fingest.add_argument("--workspace", default=None,
                         help="Workspace root (default: CJM_WORKSPACE env, else upward walk from cwd)")
    fingest.add_argument("--out-dir", default=None,
                         help="Proposal-set root directory (default: <workspace>/proposals)")
    fingest.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")

    fconfirm = sub.add_parser(
        "filter-confirm",
        help="The headless HITL worklist over a filtering proposal set: list pending "
             "proposals + derived verdicts; --accept/--accept-tier1/--accept-all commit "
             "strata (accept IS the domain op), --retract supersedes a live stratum, "
             "--watermark asserts the lane's annotated_through — all journaled")
    _add_graph_read_args(fconfirm)
    fconfirm.add_argument("--source", required=True,
                          help="Source node id or title substring (exactly one match)")
    fconfirm.add_argument("--rendition", default=None,
                          help="Which AudioRendition spine when a source has more than one")
    fconfirm.add_argument("--skeleton", default=None,
                          help="Which SKELETON spine (\"legacy\" or a hash prefix); default: auto")
    fconfirm.add_argument("--set", default=None,
                          help="Proposal-set id or prefix (default: the newest set bound to "
                               "this source + spine)")
    fconfirm.add_argument("--accept", action="append", default=None, metavar="PROPOSAL",
                          help="Accept one pending proposal by id prefix (repeatable)")
    fconfirm.add_argument("--accept-tier1", action="store_true",
                          help="Accept EVERY pending tier-1 proposal (the explicit human "
                               "batch-accept gesture — never automatic, DEC 304fd984)")
    fconfirm.add_argument("--accept-all", action="store_true",
                          help="Accept every pending proposal, audition tier included")
    fconfirm.add_argument("--relabel", action="append", default=None, metavar="PROPOSAL:CLASS",
                          help="Accept one pending proposal under a DIFFERENT category "
                               "(repeatable; derives as 'relabeled')")
    fconfirm.add_argument("--accept-as-mark", action="append", default=None,
                          metavar="PROPOSAL[:CLASS]",
                          help="Accept one pending proposal AS A MARK (class-family routing: "
                               "ASR errors, suspect nouns and other correction-pass attention "
                               "are marks, never strata); CLASS defaults to the proposal's "
                               "category; multi-segment runs anchor on each segment")
    fconfirm.add_argument("--retract", action="append", default=None, metavar="STRATUM",
                          help="Retract a live stratum by id prefix (repeatable)")
    fconfirm.add_argument("--watermark", default=None,
                          help="Assert the filtering lane's annotated_through (source seconds, "
                               "\"end\", or \"none\" = nothing visited) — set EXPLICITLY on "
                               "pause; absence below it derives rejects")
    fconfirm.add_argument("--tier2", action="store_true",
                          help="Show the audition tier in the pending list")
    fconfirm.add_argument("--markdown", action="store_true",
                          help="(Re)write <set-dir>/proposals.md — the human view of the set "
                               "(full rationale + quote + the verbatim segment run with spine "
                               "indices, joined against the pack) — and print its path")
    fconfirm.add_argument("--actor", default="human", help="Actor recorded on the ops")
    fconfirm.add_argument("--note", default=None, help="Note recorded on retractions")
    fconfirm.add_argument("--purpose", default=None,
                          help="CorrectionSession purpose (None = genuine; \"feature-test\" "
                               "= excludable from flywheel datasets)")
    fconfirm.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")
    return parser


def _add_graph_read_args(p: argparse.ArgumentParser) -> None:  # Shared workspace/graph plumbing
    """Attach the workspace + graph-capability arguments the spine verbs share."""
    p.add_argument("--manifests-dir", default=None,
                   help="Capability manifests directory (default: the workspace's .cjm/manifests "
                        "when one is active, else .cjm/manifests under the cwd)")
    p.add_argument("--workspace", default=None,
                   help="Workspace root (default: CJM_WORKSPACE env, else upward walk from cwd)")
    p.add_argument("--graph-capability", default="cjm-capability-graph-sqlite",
                   help="Graph-storage capability name")
    p.add_argument("--graph-db-path", default=None,
                   help="Graph db path (default: the workspace capability config)")


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
    if args.command == "export-wordless-propset":
        return asyncio.run(export_command(args))
    if args.command == "scan-mishomed":
        return asyncio.run(scan_command(args))
    if args.command == "filter-pack":
        return asyncio.run(filter_pack_command(args))
    if args.command == "filter-ingest":
        return filter_ingest_command(args)
    if args.command == "filter-confirm":
        return asyncio.run(filter_confirm_command(args))
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
        db = _resolve_graph_db(args, manager, args.graph_capability, ws)
        gate_id = await commit_extraction_gate(
            queue, args.graph_capability, sid, skeleton_hash, args.status, watermark,
            actor=args.actor, journal_path=sidecar_journal_path(db))
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


def overlay_event_rows(
    src: Dict[str, Any],                 # Source record ({"id","title","path","content_hash"})
    skeleton_hash: Optional[str],        # The emitting spine's skeleton hash
    overlays: List[Dict[str, Any]],      # speech_overlay_spans records (source-scoped)
    spine_segment_ids: set,              # This spine's segment-id set (the anchoring cut)
) -> Tuple[List[Dict[str, Any]], int]:   # (event rows, foreign-spine skip count)
    """Overlay span records -> dataset event rows for ONE spine (pure).

    The second sample source's WRITER half (check fc42614d, DEC 4e05a066 —
    the fold computed overlays but extract never emitted them; caught at the
    first real overlay extraction 2026-08-04). Overlays are SOURCE-scoped in
    the fold, so the anchoring-spine cut here is what prevents double
    emission when a source carries more than one eligible gated spine: an
    overlay emits only under the spine whose segment set contains its anchor.
    `snap` rides along — human-refined (nudged) vs machine (fa-word/
    fa-partial) vs estimated is provenance the bench splits on."""
    rows: List[Dict[str, Any]] = []
    foreign = 0
    for o in overlays:
        if str(o.get("segment_id")) not in spine_segment_ids:
            foreign += 1
            continue
        rows.append({
            "kind": "speech_overlay", "source_id": src["id"],
            "source_title": src["title"], "source_path": src["path"],
            "source_content_hash": src["content_hash"],
            "skeleton_hash": skeleton_hash, "overlay_id": o["overlay_id"],
            "segment_id": o.get("segment_id"),
            "label": o["label"], "text": o["text"],
            "start_time": o["start_time"], "end_time": o["end_time"],
            "snap": o.get("snap"), "words": list(o.get("words") or []),
            "split": "train",
            "provenance": {"tag": "real", "sessions": [o["session_id"]],
                           "op_ids": list(o.get("op_ids") or [])}})
    return rows, foreign


async def extract_command(
    args: argparse.Namespace,  # Parsed args for the `extract` subcommand
) -> int:  # Process exit code
    """Execute the `extract` subcommand: fold the gated overlay into a manifested dataset.

    Thin stack wrapper since DEC 82c463fe: resolves the workspace, opens the
    graph seat, and hands off to `run_extract` — the fold itself is seat-
    agnostic so the correction TUI's flywheel page runs it on the stack it
    already holds."""
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
        manifest = await run_extract(
            queue, args.graph_capability, ws=ws, manager=manager,
            source=args.source, rendition=args.rendition,
            include_purposes=args.include_purpose,
            output_dir=args.output_dir, graph_db_path=args.graph_db_path)
        if manifest is None:
            return 1
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
            if d.get("format") != "cjm-transcript-decomp-core/run-manifest":
                continue
            if d.get("event_propset_id") == set_id and d.get("skeleton_config_hash"):
                skeleton = d["skeleton_config_hash"]
            # Multi-source runs (decomp manifest 0.2.6) carry the consumed
            # set + skeleton PER SOURCE record; the run-level fields are
            # empty there, so the join reads the record that names this set.
            for rec in d.get("sources") or []:
                if rec.get("event_propset_id") == set_id and rec.get("skeleton_config_hash"):
                    skeleton = rec["skeleton_config_hash"]
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
    """Execute `transfer-wordless`: replay wordless event inserts (and speaker-
    split boundaries) across sibling spines.

    A thin driver over the shared engine (plan_wordless_transfer +
    commit_wordless_transfer — the correction shell's spine-picker verb
    drives the same two calls in-process with the plan rendered before
    commit, 9af9793a). dea104ba — the layering principle's first mechanical
    expression: accepted wordless inserts are SOURCE-truth ((span, label)
    pairs with zero text dependency), so a respine (re-transcription, FA/VAD
    upgrade) must not lose the human event layer. Speaker splits ride as
    boundaries at source time (54aac7d3; --no-splits leaves them). What does
    NOT transfer, by design: boundary shifts, text edits, and any labeled
    insert that GAINED text (word-bearing = spine-truth — new transcript,
    new FA). Idempotent: a destination insert with the same label within
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
        sid, title, _path = await resolve_source_node(
            queue, args.graph_capability, args.source)
        print(f"source: {title or sid}  ({sid})")
        plan = await plan_wordless_transfer(
            queue, args.graph_capability, sid,
            from_skeleton=args.from_skeleton, to_skeleton=args.to_skeleton,
            rendition=args.rendition, labels=args.labels,
            tolerance=args.tolerance, splits=not args.no_splits)
        print(f"spines: from {plan['from_segments']} segs -> to {plan['to_segments']} segs")
        rows = plan["plan"]
        print(f"donors: {plan['donors']}  ->  transfer {len(rows)}  "
              f"(dup-skip {plan['dups']} · word-bearing-skip {plan['word_bearing']} · "
              f"unanchored {plan['unanchored']})")
        print("by label: " + (" · ".join(f"{k}x{v}" for k, v in sorted(plan["by_label"].items()))
                              or "none"))
        if plan["split_donors"]:
            print(f"speaker splits: {plan['split_donors']}  ->  transfer {len(plan['splits'])}  "
                  f"(dup-skip {plan['split_dups']} · unanchored {plan['split_unanchored']} · "
                  f"same-segment conflicts {plan['split_conflicts']})")
        if args.dry_run:
            for p in rows[:10]:
                print(f"  {p['label']:>16} {p['start']:9.3f}-{p['end']:9.3f}s "
                      f"after {p['after_id'][:8]}")
            if len(rows) > 10:
                print(f"  … {len(rows) - 10} more")
            for s in plan["splits"][:10]:
                print(f"  {'split':>16} {s['time']:9.3f}s  "
                      f"{s['left_text'][-24:]!r} | {s['right_text'][:24]!r}")
            if len(plan["splits"]) > 10:
                print(f"  … {len(plan['splits']) - 10} more splits")
            print("dry run — nothing committed")
            return 0
        res = await commit_wordless_transfer(
            queue, args.graph_capability, sid, plan,
            journal_path=sidecar_journal_path(args.graph_db_path), actor=args.actor)
        print(f"transferred {res['transferred']} event inserts + {res['splits']} "
              f"speaker splits (session {res['session_id']})")
    finally:
        await queue.stop()
        try:
            manager.unload_capability(args.graph_capability)
        except Exception as e:  # Best-effort teardown; never mask the transfer outcome
            logger.warning(f"unload {args.graph_capability} failed: {e}")
    return 0


def wordless_donors(
    effective_units: List,                    # project_effective_spine output for the donor spine
    insert_meta: Dict[str, Dict],             # chunk_insert payloads keyed by effective-unit id
    labels: Optional[List[str]] = None,       # Restrict to these labels (None = every labeled insert)
) -> Tuple[List[Dict], int]:  # (donor rows {start,end,label,rank}, word-bearing skip count)
    """The EFFECTIVE wordless layer of a spine: labeled, effectively wordless
    chunk-insert units of its projection (time nudges applied — the heard span,
    not the born one), rank preserved. ONE definition, two consumers
    (f5d080b9 direction a): `transfer-wordless` replays these donors onto a
    sibling spine; `export-wordless-propset` writes them out as a proposal set
    a respine consumes as carve authority — so the exported carve spans are the
    transfer donor set BY CONSTRUCTION and transferred events land in exact
    gaps. An insert that GAINED text stays behind (word-bearing = spine-truth)."""
    donors: List[Dict] = []
    word_bearing = 0
    for u in effective_units:
        meta = insert_meta.get(u.id)
        if meta is None:
            continue
        label = meta.get("label")
        if not label or (labels and label not in labels):
            continue
        if (u.text or "").strip():
            word_bearing += 1
            continue
        if u.start_time is None or u.end_time is None:
            continue
        donors.append({"start": float(u.start_time), "end": float(u.end_time),
                       "label": str(label), "rank": float(meta.get("rank") or 0.0)})
    return donors, word_bearing


async def resolve_source_node(
    queue: JobQueue,   # Started job queue
    graph_id: str,     # Graph-storage capability id
    source: str,       # Source node id, or a case-insensitive title substring
) -> Tuple[str, str, Optional[str]]:  # (source id, title, media path)
    """Resolve the --source selector the respine verbs share: exactly ONE
    Source node by id or title substring (loud on 0 or many — the contract
    every spine verb inherits). The media path rides along because the
    export's content-hash binding wants it and the shell holds only an id."""
    res = await graph_task(queue, graph_id, "query_nodes",
                           query=NodeQuery(label="Source",
                                           project=["title", "path"]).to_dict())
    needle = source.lower()
    matches = [(r["id"], str(r.get("title") or ""), r.get("path"))
               for r in (res.rows or [])
               if r["id"] == source or needle in str(r.get("title") or "").lower()]
    if len(matches) != 1:
        raise SystemExit(f"--source matched {len(matches)} Source nodes "
                         f"({[t for _, t, _ in matches]}); need exactly one")
    return matches[0]


def plan_transfer_rows(
    donors: List[Dict],                  # wordless_donors rows for the FROM spine ({start,end,label,rank})
    existing: List[Tuple[str, float]],   # (label, start) of the TO spine's ACTIVE labeled inserts (dup guard)
    to_l0: List,                         # The TO spine's LAYER-0 segments in index order (start_time set)
    tolerance: float = 0.05,             # Dup window (s): a same-label insert this close is already there
) -> Tuple[List[Dict], int, int]:  # (placed rows = donor + {after_id, before_id}, dup-skips, unanchored)
    """Place wordless donors on the destination (pure) — the event half of
    the transfer plan. Source-time anchoring: each donor lands after the
    last layer-0 segment starting at or before its start (the flank pair the
    chunk_insert projection needs); a donor before the first segment is
    `unanchored`; a same-label destination insert within `tolerance` of the
    donor's start is a dup (the idempotency guard — a rerun transfers 0)."""
    plan: List[Dict] = []
    dups = unanchored = 0
    for d in donors:
        if any(lb == d["label"] and abs(st - d["start"]) <= tolerance
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
    return plan, dups, unanchored


async def plan_wordless_transfer(
    queue: JobQueue,                      # Started job queue (an open stack — CLI or shell)
    graph_id: str,                        # Graph-storage capability id
    source_id: str,                       # The resolved Source node id
    *,
    from_skeleton: str,                   # Donor spine selector ("legacy" or a skeleton-hash prefix / full hash)
    to_skeleton: str,                     # Destination spine selector
    rendition: Optional[str] = None,      # Which AudioRendition spine (None = auto)
    labels: Optional[List[str]] = None,   # Restrict to these insert labels (None = every labeled wordless insert)
    tolerance: float = 0.05,              # Dup-guard window (s)
    splits: bool = True,                  # Also plan speaker-split boundaries (54aac7d3 rider)
) -> Dict[str, Any]:  # The plan: counts + placed rows (see below); reads only
    """PLAN a wordless transfer (reads only) — the `transfer-wordless`
    engine's first half, shared by the CLI verb and the correction shell's
    spine-picker verb (9af9793a: the dry-run plan renders in-app before
    commit, so one function must produce what both surfaces show).

    dea104ba's layering: accepted wordless inserts are SOURCE-truth ((span,
    label) pairs, zero text dependency), so a respine must not lose the
    human event layer. Donors come from the FROM spine's EFFECTIVE
    projection (time nudges applied) and keep their walked rank; placement
    anchors by source time between the TO spine's layer-0 segments
    (plan_transfer_rows). Speaker splits ride as BOUNDARIES (split_donors +
    plan_split_rows — the destination re-cuts its own words at the donor's
    time). What stays behind by design: boundary shifts, text edits, and
    labeled inserts that GAINED text (word-bearing = spine-truth).
    Idempotent by construction: dup-skips on both halves.

    Returns {"from_segments","to_segments","donors","plan","by_label","dups",
    "word_bearing","unanchored","split_donors","splits","split_dups",
    "split_unanchored","split_conflicts"}; refusals (empty spine, the same
    spine twice) raise SystemExit with the CLI's wording."""
    segs_from = await load_source_segments(
        queue, graph_id, source_id,
        rendition_selector=rendition, skeleton_selector=from_skeleton)
    segs_to = await load_source_segments(
        queue, graph_id, source_id,
        rendition_selector=rendition, skeleton_selector=to_skeleton)
    if not segs_from or not segs_to:
        raise SystemExit(f"empty spine (from={len(segs_from)} to={len(segs_to)} segments)")
    if {s.id for s in segs_from} == {s.id for s in segs_to}:
        raise SystemExit("--from-skeleton and --to-skeleton resolved to the SAME spine")

    corrections, superseded = await load_source_corrections(queue, graph_id, source_id)
    active = [c for c in corrections
              if c["id"] not in superseded and c.get("status") != "proposed"]
    insert_meta = {c["id"]: (c.get("payload") or {}) for c in active
                   if c.get("correction_type") == "insertion"
                   and (c.get("payload") or {}).get("operation") == "chunk_insert"}
    # Donors: labeled + effectively wordless units of the FROM projection
    # (the shared wordless_donors definition — export-wordless-propset
    # writes this exact set, f5d080b9 direction a).
    eff_from = project_effective_spine(segs_from, active)
    donors, word_bearing = wordless_donors(eff_from, insert_meta, labels)
    # Existing destination events (idempotency guard) from the TO projection.
    eff_to = project_effective_spine(segs_to, active)
    existing = [(str(insert_meta[u.id].get("label")), float(u.start_time))
                for u in eff_to
                if u.id in insert_meta and insert_meta[u.id].get("label")
                and u.start_time is not None]
    to_l0 = sorted((s for s in segs_to if s.start_time is not None),
                   key=lambda s: s.index)
    plan, dups, unanchored = plan_transfer_rows(donors, existing, to_l0, tolerance)
    by_label: Dict[str, int] = {}
    for p in plan:
        by_label[p["label"]] = by_label.get(p["label"], 0) + 1
    out: Dict[str, Any] = {
        "from_segments": len(segs_from), "to_segments": len(segs_to),
        "donors": len(donors), "plan": plan, "by_label": by_label,
        "dups": dups, "word_bearing": word_bearing, "unanchored": unanchored,
        "split_donors": 0, "splits": [], "split_dups": 0,
        "split_unanchored": 0, "split_conflicts": 0}
    if splits:
        # Local import: spine.py imports this module's load_capabilities at
        # its top, so the planners cannot be a top-level import here.
        from cjm_transcript_correction_core.spine import plan_split_rows, split_donors
        split_ids = {c["id"] for c in active
                     if c["id"] in insert_meta and c.get("rationale") == "chunk-split"}
        sd = split_donors(eff_from, insert_meta, split_ids)
        existing_splits = [float(u.start_time) for u in eff_to
                           if u.id in split_ids and u.start_time is not None]
        synthetic = {u.id for u in eff_to if u.id in insert_meta}
        srows, sdups, sun, sconf = plan_split_rows(sd, eff_to, synthetic,
                                                   existing_splits, tolerance)
        out.update(split_donors=len(sd), splits=srows, split_dups=sdups,
                   split_unanchored=sun, split_conflicts=sconf)
    return out


async def commit_wordless_transfer(
    queue: JobQueue,                      # Started job queue (an open stack — CLI or shell)
    graph_id: str,                        # Graph-storage capability id
    source_id: str,                       # The resolved Source node id
    plan: Dict[str, Any],                 # A plan_wordless_transfer result
    *,
    journal_path: Optional[str],          # Sidecar journal (the flywheel journal rides the db sidecar)
    actor: str = "human",                 # Actor recorded on the transferred corrections
    purpose: str = "wordless-transfer",   # CorrectionSession purpose tag
) -> Dict[str, Any]:  # {"session_id","transferred","splits"}
    """COMMIT a planned transfer — the engine's second half. One
    CorrectionSession stamps everything: the event rows land as labeled
    chunk inserts (rank preserved, layer-0 flank anchors from the plan), the
    split rows as composed chunk splits (insert + left text + end nudge —
    build_chunk_split_corrections, the donor's cut time), and the session
    completes. Every commit journals through the sidecar exactly as the
    walk's own gestures do — a replay reproduces the transfer."""
    sess = await start_session(queue, graph_id, [source_id],
                               journal_path=journal_path, purpose=purpose, actor=actor)
    for p in plan.get("plan") or []:
        await commit_chunk_insert_correction(
            queue, graph_id, source_id, p["after_id"],
            p["start"], p["end"], sess.id,
            before_segment_id=p["before_id"], label=p["label"], rank=p["rank"],
            actor=actor, journal_path=journal_path)
    for s in plan.get("splits") or []:
        await commit_chunk_split_correction(
            queue, graph_id, source_id, s["segment_id"], s["time"],
            s["left_text"], s["right_text"], s["end_s"], sess.id, s["after_id"],
            before_segment_id=s["before_id"], old_text=s["old_text"],
            boundary_words=s["boundary_words"], actor=actor,
            journal_path=journal_path)
    await set_session_status(queue, graph_id, sess.id, "completed",
                             journal_path=journal_path, actor=actor)
    return {"session_id": sess.id, "transferred": len(plan.get("plan") or []),
            "splits": len(plan.get("splits") or [])}


async def plan_wordless_export(
    queue: JobQueue,                      # Started job queue (an open stack — CLI or shell)
    graph_id: str,                        # Graph-storage capability id
    source_id: str,                       # The resolved Source node id
    *,
    from_skeleton: str,                   # The walked spine whose effective wordless layer exports
    rendition: Optional[str] = None,      # Which AudioRendition spine (None = auto)
    labels: Optional[List[str]] = None,   # Restrict to these insert labels (None = every labeled wordless insert)
) -> Dict[str, Any]:  # {"donors","word_bearing","counts","from_hash","window_end"}; reads only
    """PLAN an export (reads only) — the `export-wordless-propset` engine's
    read half, shared by the CLI verb and the shell's spine-picker verb
    (9af9793a). The donor set is exactly `transfer-wordless`'s (shared
    wordless_donors: accepted + nudged + manual, word-bearing stays behind),
    source-time ordered, with the per-label census and the donor spine's
    skeleton hash the manifest's provenance records. Refusals (empty spine,
    no donors) raise SystemExit with the CLI's wording."""
    segs = await load_source_segments(
        queue, graph_id, source_id,
        rendition_selector=rendition, skeleton_selector=from_skeleton)
    if not segs:
        raise SystemExit("empty spine (0 segments)")
    spines = await list_source_spines(queue, graph_id, source_id,
                                      rendition_selector=rendition)
    from_hash = skeleton_hash_for(spines, from_skeleton)
    corrections, superseded = await load_source_corrections(queue, graph_id, source_id)
    active = [c for c in corrections
              if c["id"] not in superseded and c.get("status") != "proposed"]
    insert_meta = {c["id"]: (c.get("payload") or {}) for c in active
                   if c.get("correction_type") == "insertion"
                   and (c.get("payload") or {}).get("operation") == "chunk_insert"}
    eff = project_effective_spine(segs, active)
    donors, word_bearing = wordless_donors(eff, insert_meta, labels)
    if not donors:
        raise SystemExit("no wordless donors on this spine — nothing to export")
    donors.sort(key=lambda d: d["start"])
    counts: Dict[str, int] = {}
    for d in donors:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    window_end = max((float(s.end_time) for s in segs
                      if s.end_time is not None), default=donors[-1]["end"])
    return {"donors": donors, "word_bearing": word_bearing, "counts": counts,
            "from_hash": from_hash, "window_end": window_end}


def write_wordless_propset(
    plan: Dict[str, Any],                 # A plan_wordless_export result
    *,
    out_root: Path,                       # Proposal-set root (<workspace>/proposals)
    source_id: str,                       # The Source node id
    media_path: Optional[str],            # The source media (content-hash binding; None = path-only binding)
    from_skeleton: str,                   # The selector the export was asked for (config provenance)
    rendition: Optional[str] = None,      # Rendition selector (config provenance)
    labels: Optional[List[str]] = None,   # Label restriction (config provenance)
    ws: Any = None,                       # Resolved workspace (relativize_recorded) or None
) -> Dict[str, Any]:  # {"set_id","set_dir","manifest_path","classes","counts"}
    """WRITE a planned export as a proposal set — the engine's write half.

    Serialized in the proposal-set-manifest format the decomp carve and the
    propset pickers already consume (EVENT_PROPOSAL_SET_FORMAT; every row
    tier 1 — human-verified spans ARE the operating point), so the decomp
    app's Shift+E ring lists it beside model sets and a respine consuming it
    cuts at the refined spans (the straddle class 2ba9e368 dies at the
    root). Provenance: model.kind='human-effective-layer' + the donor
    spine's skeleton hash; the source binding is the media content hash
    (else path) — the join key PropsetIndex.for_source uses."""
    donors = plan["donors"]
    counts = plan["counts"]
    content_hash = None
    if media_path and Path(media_path).is_file():
        h = hashlib.sha256()
        with open(media_path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        content_hash = f"sha256:{h.hexdigest()}"
    started = time.time()
    set_id = (f"propset_{time.strftime('%Y%m%d_%H%M%S', time.localtime(started))}"
              f"_{uuid.uuid4().hex[:8]}")
    set_dir = Path(out_root) / set_id
    set_dir.mkdir(parents=True)
    with open(set_dir / "proposals.jsonl", "w") as f:
        for d in donors:
            f.write(json.dumps({
                "proposal_id": str(uuid.uuid4()),
                "label": d["label"],
                "start_time": round(d["start"], 4),
                "end_time": round(d["end"], 4),
                "score": round(d["rank"], 4),
                "tier": 1,
            }) + "\n")
    from_hash = plan.get("from_hash")
    manifest = {
        "format": EVENT_PROPOSAL_SET_FORMAT,
        "version": "0.2.0",
        "proposal_set_id": set_id,
        "created_at": started,
        "config": {
            "exporter": "cjm-transcript-correction-core/export-wordless-propset",
            "from_skeleton": from_skeleton,
            "rendition": rendition,
            "labels": sorted(labels) if labels else None,
        },
        "training_run_manifest": "",
        "training_run_id": "",
        "model": {"kind": "human-effective-layer",
                  "from_skeleton_hash": from_hash},
        "source": {"path": media_path, "content_hash": content_hash,
                   "source_id": source_id,
                   **({"skeleton_hash": from_hash} if from_hash else {})},
        "window": {"start": 0.0, "end": plan["window_end"]},
        "classes": sorted(counts),
        "files": {"proposals": "proposals.jsonl"},
        "counts": counts,
    }
    manifest_path = set_dir / "manifest.json"
    manifest_path.write_text(json.dumps(relativize_recorded(manifest, ws), indent=2))
    return {"set_id": set_id, "set_dir": str(set_dir),
            "manifest_path": str(manifest_path),
            "classes": sorted(counts), "counts": dict(counts)}


async def export_command(
    args: argparse.Namespace,  # Parsed args for the `export-wordless-propset` subcommand
) -> int:  # Process exit code
    """Execute `export-wordless-propset`: write one spine's effective wordless
    layer out as a proposal set (f5d080b9 direction a — effective-layer-as-
    carve-authority).

    A thin driver over the shared engine (plan_wordless_export +
    write_wordless_propset — the correction shell's spine-picker verb drives
    the same two calls in-process, 9af9793a): the exported spans are exactly
    the `transfer-wordless` donor set (shared `wordless_donors`: accepted +
    nudged + manual, word-bearing stays behind), serialized in the
    proposal-set-manifest format the decomp carve and the propset pickers
    already consume. A respine consuming the set cuts at the refined spans,
    so a subsequent transfer-wordless lands every event in an EXACT gap — the
    straddle class (2ba9e368) dies at the root. Reads only — no graph
    writes, no journal."""
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    if args.manifests_dir is None:
        args.manifests_dir = (str(ws.substrate_data_dir / "manifests")
                              if ws is not None else ".cjm/manifests")
    out_root = (Path(args.out_dir) if args.out_dir
                else (ws.root / "proposals" if ws is not None else None))
    if out_root is None and not args.dry_run:
        raise SystemExit("proposal sets land workspace-local — pass --out-dir "
                         "or run inside a workspace (CJM_WORKSPACE)")
    manager = CapabilityManager(search_paths=[Path(args.manifests_dir)])
    configs = ({args.graph_capability: {"db_path": args.graph_db_path}}
               if args.graph_db_path else None)
    load_capabilities(manager, [args.graph_capability], configs=configs)
    queue = JobQueue(deps=manager)
    await queue.start()
    try:
        sid, title, media_path = await resolve_source_node(
            queue, args.graph_capability, args.source)
        print(f"source: {title or sid}  ({sid})")
        plan = await plan_wordless_export(
            queue, args.graph_capability, sid,
            from_skeleton=args.from_skeleton, rendition=args.rendition,
            labels=args.labels)
        donors, counts = plan["donors"], plan["counts"]
        print(f"donors: {len(donors)}  (word-bearing-skip {plan['word_bearing']})")
        print("by label: " + " · ".join(f"{k}x{v}" for k, v in sorted(counts.items())))
        if args.dry_run:
            for d in donors[:10]:
                print(f"  {d['label']:>16} {d['start']:9.3f}-{d['end']:9.3f}s")
            if len(donors) > 10:
                print(f"  … {len(donors) - 10} more")
            print("dry run — nothing written")
            return 0
        res = write_wordless_propset(
            plan, out_root=out_root, source_id=sid, media_path=media_path,
            from_skeleton=args.from_skeleton, rendition=args.rendition,
            labels=args.labels, ws=ws)
        print(f"proposal set {res['set_id']} -> {res['set_dir']}")
        print("consume: decomp --respine --event-split "
              f"--event-propset {res['manifest_path']} "
              + " ".join(f"--event-classes {c}" for c in res["classes"]))
    finally:
        await queue.stop()
        try:
            manager.unload_capability(args.graph_capability)
        except Exception as e:  # Best-effort teardown; never mask the export outcome
            logger.warning(f"unload {args.graph_capability} failed: {e}")
    return 0


# ---- the filtering lane verbs (strata.py is the engine; these are thin drivers) ----

async def _open_graph_stack(
    args: argparse.Namespace,  # Parsed args carrying the _add_graph_read_args surface
) -> Tuple[Any, CapabilityManager, JobQueue]:  # (resolved workspace or None, manager, started queue)
    """Resolve the workspace, load the graph capability, start the queue — the
    plumbing every read-only spine verb repeats (gate/export/scan), shared by
    the filtering verbs."""
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
    return ws, manager, queue


def _resolve_graph_db(
    args: argparse.Namespace,    # Parsed args (graph_db_path may be explicit)
    manager: CapabilityManager,  # The loaded manager
    graph_capability: str,       # The graph capability name
    ws: Any,                     # Resolved workspace or None
) -> str:  # The effective graph db path — the sidecar journal derives from it
    """Name the graph db a WRITE verb journals against: explicit --graph-db-path
    > the loaded instance's config (persisted, workspace-scoped) > the
    workspace's conventional capability data path when it exists. No answer =
    LOUD refusal before anything is written — a write that cannot be journaled
    must not happen (journal-first; the silent `journal none` sighting
    2026-09-01 on the filtering lane's first accept)."""
    if args.graph_db_path:
        return str(args.graph_db_path)
    inst = manager.instances.get(graph_capability)
    cfg_db = ((inst.config if inst is not None else None) or {}).get("db_path")
    if cfg_db:
        return str(cfg_db)
    if ws is not None:
        conv = ws.substrate_data_dir / "data" / graph_capability / "context_graph.db"
        if conv.is_file():
            return str(conv)
    raise SystemExit(f"cannot name the graph db to journal against: pass --graph-db-path, "
                     f"or persist db_path on {graph_capability} in the workspace config store")


async def _close_graph_stack(
    manager: CapabilityManager,  # The manager _open_graph_stack built
    queue: JobQueue,             # Its started queue
    graph_capability: str,       # The loaded capability name
) -> None:
    """Best-effort teardown; never masks the verb's outcome."""
    await queue.stop()
    try:
        manager.unload_capability(graph_capability)
    except Exception as e:
        logger.warning(f"unload {graph_capability} failed: {e}")


async def _source_content_hash(
    queue: JobQueue,  # Started job queue
    graph_id: str,    # Graph-storage capability id
    source_id: str,   # The Source node id
) -> Optional[str]:  # The media content hash on the Source's provenance SourceRef (None = unhashed)
    """The Source's content hash lives on its provenance SourceRef (node identity
    input), not in properties — the run-independent binding a pack records."""
    node = await graph_task(queue, graph_id, "get_node", node_id=source_id)
    d = node.to_dict() if hasattr(node, "to_dict") else (node or {})
    for r in (d.get("sources") or []):
        h = r.get("content_hash") if isinstance(r, dict) else getattr(r, "content_hash", None)
        if h:
            return str(h)
    return None


def _short(value: Optional[str], n: int = 8) -> str:  # Display tail of an id
    return str(value or "")[-n:] if value else "-"


async def filter_pack_command(
    args: argparse.Namespace,  # Parsed args for the `filter-pack` subcommand
) -> int:  # Process exit code
    """Execute `filter-pack`: write one spine window's effective text-bearing
    segments as a proposer pack — `<out-dir>/<pack_id>.json` (what ingest
    resolves against) + `<pack_id>.md` (the brief a proposer reads). Reads
    only; the pack is the read-trace every proposal from it cites."""
    ws, manager, queue = await _open_graph_stack(args)
    cap = args.graph_capability
    try:
        sid, title, _media = await resolve_source_node(queue, cap, args.source)
        segs = await load_source_segments(queue, cap, sid, rendition_selector=args.rendition,
                                          skeleton_selector=args.skeleton)
        if not segs:
            raise SystemExit("empty spine (0 segments)")
        spines = await list_source_spines(queue, cap, sid, rendition_selector=args.rendition)
        skel = skeleton_hash_for(spines, args.skeleton)
        corrections, superseded = await load_source_corrections(queue, cap, sid)
        active = [c for c in corrections
                  if c["id"] not in superseded and c.get("status") != "proposed"]
        eff = project_effective_spine(segs, active)
        strata = active_strata(corrections, superseded)
        chash = await _source_content_hash(queue, cap, sid)
        window = tuple(args.window) if args.window else None
        pack = build_filter_pack(sid, title, skel, eff, content_hash=chash,
                                 window=window, strata=strata)
    finally:
        await _close_graph_stack(manager, queue, cap)
    out_dir = (Path(args.out_dir) if args.out_dir
               else (ws.root / "packs" if ws is not None else Path("packs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{pack['pack_id']}.json"
    md_path = out_dir / f"{pack['pack_id']}.md"
    json_path.write_text(json.dumps(pack, indent=2, ensure_ascii=False))
    md_path.write_text(render_filter_pack(pack))
    w = pack["window"]
    print(f"source: {title or sid}  ({sid}) · spine {_spine_tag(skel)}")
    print(f"pack {pack['pack_id']}: {len(pack['segments'])} text segments · window "
          f"{w['start']:.1f}-{(w['end'] if w['end'] is not None else 0.0):.1f}s · "
          f"{len(pack['existing_strata'])} strata already asserted")
    print(f"  json  {json_path}")
    print(f"  brief {md_path}")
    print(f"next: hand the brief to a proposer; then filter-ingest --pack {json_path} "
          "--rows <its jsonl> --proposer <name>")
    return 0


def _read_proposer_rows(path: Path) -> List[Dict[str, Any]]:  # Lenient JSONL / JSON-array reader
    """Read a proposer's output: JSONL rows, or one JSON array; blank lines and
    code-fence lines are skipped (a sub-agent's reply often carries them)."""
    text = path.read_text()
    stripped = text.strip()
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise SystemExit(f"{path}: expected a JSON array of rows")
        return data
    rows: List[Dict[str, Any]] = []
    for k, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{k}: not a JSON row ({e.msg})")
    return rows


def filter_ingest_command(
    args: argparse.Namespace,  # Parsed args for the `filter-ingest` subcommand
) -> int:  # Process exit code
    """Execute `filter-ingest`: validate proposer rows against their pack and
    write the filtering proposal set (durable inference output; the proposer's
    provenance + the pack digest ride the manifest). No graph, no journal —
    nothing is decided here."""
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    pack_path = Path(args.pack)
    try:
        pack = json.loads(pack_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"cannot read pack {pack_path}: {e}")
    if pack.get("format") != FILTER_PACK_FORMAT:
        raise SystemExit(f"{pack_path} is not a filter pack (format {pack.get('format')!r})")
    rows = _read_proposer_rows(Path(args.rows))
    try:
        valid = validate_proposal_rows(rows, pack)
    except ValueError as e:
        raise SystemExit(f"rows rejected: {e}")
    proposals = proposals_from_rows(valid, pack)
    out_root = (Path(args.out_dir) if args.out_dir
                else (ws.root / "proposals" if ws is not None else None))
    if out_root is None:
        raise SystemExit("proposal sets land workspace-local — pass --out-dir or run "
                         "inside a workspace (CJM_WORKSPACE)")
    proposer = {"kind": args.proposer_kind, "name": args.proposer,
                **({"model": args.model} if args.model else {}),
                **({"session": os.environ["CJM_SESSION"]} if os.environ.get("CJM_SESSION") else {})}
    res = write_filter_propset(pack, proposals, out_root=out_root, proposer=proposer, ws=ws)
    src = pack.get("source") or {}
    tag = _spine_tag(src.get("skeleton_hash"))
    md_path = Path(res["set_dir"]) / "proposals.md"
    manifest = json.loads(Path(res["manifest_path"]).read_text())
    md_path.write_text(render_filter_propset_markdown(manifest, proposals, pack))
    print(f"source: {src.get('title') or src.get('source_id')} · spine {tag} · "
          f"pack {pack.get('pack_id')}")
    print(f"proposal set {res['set_id']} -> {res['set_dir']}")
    print(f"  human view {md_path}")
    t1 = " · ".join(f"{k}x{v}" for k, v in sorted(res["counts"].items())) or "none"
    t2 = " · ".join(f"{k}x{v}" for k, v in sorted(res["tier2_counts"].items())) or "none"
    print(f"tier-1: {t1}\ntier-2: {t2}")
    print(f"next: filter-confirm --source {src.get('source_id')} --skeleton {tag}")
    return 0


def _pick_by_prefix(
    candidates: List[Dict[str, Any]],  # Rows carrying `key`
    key: str,                          # The id field
    prefix: str,                       # An id or prefix (case-insensitive; matches head or tail)
    what: str,                         # Noun for the refusal message
) -> Dict[str, Any]:  # The unique match
    """Resolve an id prefix to exactly one row — loud on 0 or many."""
    p = prefix.strip().lower()
    hits = [c for c in candidates
            if str(c.get(key) or "").lower().startswith(p)
            or str(c.get(key) or "").lower().endswith(p)]
    if len(hits) != 1:
        raise SystemExit(f"{what} {prefix!r} matches {len(hits)} candidates; need exactly one")
    return hits[0]


async def filter_confirm_command(
    args: argparse.Namespace,  # Parsed args for the `filter-confirm` subcommand
) -> int:  # Process exit code
    """Execute `filter-confirm`: the HEADLESS HITL worklist (bc8dbbdd pass-1
    doctrine) over one filtering proposal set.

    Without gestures it LISTS: the pending tier-1 walk (tier 2 behind --tier2,
    dim by doctrine), the live strata, the lane's watermark and the derived
    verdict summary — nothing written. With gestures it commits, journal-first:
    --accept / --accept-tier1 / --accept-all mint strata carrying the proposal
    id (accept IS the domain op); --retract supersedes a live stratum;
    --watermark asserts the filtering lane's annotated_through explicitly.
    Rejects are never marked — absence below the watermark derives them."""
    ws, manager, queue = await _open_graph_stack(args)
    cap = args.graph_capability
    try:
        sid, title, _media = await resolve_source_node(queue, cap, args.source)
        spines = await list_source_spines(queue, cap, sid, rendition_selector=args.rendition)
        skel = skeleton_hash_for(spines, args.skeleton)
        if ws is None:
            raise SystemExit("proposal sets are workspace-local — run inside a workspace "
                             "(CJM_WORKSPACE or --workspace)")
        sets = load_filter_proposal_sets(str(ws.root), sid, skeleton_hash=skel)
        if not sets:
            print(f"no filtering proposal set for {title or sid} · spine {_spine_tag(skel)} "
                  f"under {ws.root / 'proposals'} — run filter-pack + a proposer + filter-ingest")
            return 1
        chosen = (_pick_by_prefix([s["manifest"] for s in sets], "proposal_set_id", args.set,
                                  "proposal set") if args.set else sets[0]["manifest"])
        pset = next(s for s in sets if s["manifest"] is chosen)
        m, proposals = pset["manifest"], pset["proposals"]
        set_id = m.get("proposal_set_id")
        corrections, superseded = await load_source_corrections(queue, cap, sid)
        strata = active_strata(corrections, superseded)
        gates = await load_extraction_gates(queue, cap, sid, lane=FILTER_LANE)
        gate = gates.get(skel) or {}
        watermark = gate.get("annotated_through")
        window = (float((m.get("window") or {}).get("start") or 0.0),
                  (m.get("window") or {}).get("end"))

        gestures = bool(args.accept or args.accept_tier1 or args.accept_all
                        or args.relabel or args.accept_as_mark
                        or args.retract or args.watermark is not None)
        mark_ids = materialized_mark_ids(corrections, superseded)
        pending_all = pending_filter_proposals(proposals, strata, show_tier2=True,
                                               materialized=mark_ids)
        print(f"source: {title or sid}  ({sid}) · spine {_spine_tag(skel)}")
        print(f"set {set_id} · proposer {(m.get('model') or {}).get('kind')}:"
              f"{(m.get('model') or {}).get('name')} · window "
              f"{window[0]:.1f}-{(window[1] if window[1] is not None else 0.0):.1f}s"
              + (f" · {len(sets)} sets (newest shown; --set picks)" if len(sets) > 1 else ""))
        set_pids = {p.get("proposal_id") for p in proposals}
        print(f"lane watermark: {('%.1fs' % float(watermark)) if watermark is not None else 'none'}"
              f" · live strata: {len(strata)}"
              + (f" · marks from proposals: {len(mark_ids & set_pids)}" if mark_ids & set_pids else ""))
        if args.markdown:
            pack_path = ws.root / "packs" / f"{(m.get('pack') or {}).get('pack_id')}.json"
            pack = None
            if pack_path.is_file():
                try:
                    pack = json.loads(pack_path.read_text())
                except (OSError, json.JSONDecodeError):
                    pack = None
            md_path = Path(pset["path"]).parent / "proposals.md"
            md_path.write_text(render_filter_propset_markdown(m, proposals, pack))
            print(f"human view: {md_path}" + ("" if pack else "  (pack not found — quotes only)"))
        if not gestures:
            shown = [p for p in pending_all if args.tier2 or int(p.get("tier", 1)) == 1]
            hidden_t2 = sum(1 for p in pending_all if int(p.get("tier", 1)) == 2)
            print(f"pending: {len(shown)} shown"
                  + (f" · tier-2 {hidden_t2} hidden (--tier2 shows)" if hidden_t2 and not args.tier2 else ""))
            for p in shown:
                ev = p.get("evidence") or {}
                q = (ev.get("quote") or "")[:60]
                tier = "??" if int(p.get("tier", 1)) == 2 else "? "
                conf = p.get("confidence")
                print(f"  {tier}{_short(p.get('proposal_id'))}  {p.get('category'):>14}  "
                      f"{float(p.get('start_time') or 0):8.1f}-{float(p.get('end_time') or 0):8.1f}s"
                      f"  [{ev.get('from_i')}..{ev.get('to_i')}]"
                      + (f"  c={conf:.2f}" if isinstance(conf, (int, float)) else "")
                      + (f'  "{q}"' if q else ""))
                if p.get("rationale"):
                    for wrapped in textwrap.wrap(str(p["rationale"]), width=100):
                        print(f"        {wrapped}")
            if strata:
                print("live strata:")
                for c in strata:
                    p = c.get("payload") or {}
                    print(f"  {_short(c.get('id'))}  {p.get('category'):>14}  "
                          f"{float(p.get('start_time') or 0):8.1f}-{float(p.get('end_time') or 0):8.1f}s"
                          f"  x{len(p.get('segment_ids') or [])} segs · {c.get('actor')}"
                          + ("  (from proposal)" if p.get("proposal_id") else "  (human)"))
            b = bench_filter_proposals(proposals, strata, window, watermark=watermark,
                                       mark_ids=mark_ids)
            for tier_key, label in (("tier1", "tier-1"), ("tier2", "tier-2")):
                c = b["counts"][tier_key]
                if tier_key == "tier2" and not any(c.values()):
                    continue
                print(f"derived verdicts ({label}): " + " · ".join(f"{k} {v}" for k, v in c.items())
                      + (("  rates " + " ".join(f"{k}={v}" for k, v in b["rates"].items()))
                         if b["rates"] and tier_key == "tier1" else "")
                      + (f" · missed {len(b['missed'])}" if b["missed"] and tier_key == "tier1" else ""))
            print("gestures: --accept <id> · --relabel <id>:<class> · --accept-as-mark <id>[:<class>] "
                  "· --accept-tier1 · --accept-all · --retract <stratum> · --watermark <sec|end|none>")
            return 0

        db = _resolve_graph_db(args, manager, cap, ws)
        jp = sidecar_journal_path(db)
        sess = await start_session(queue, cap, [sid], journal_path=jp, purpose=args.purpose,
                                   actor=args.actor)
        to_accept: List[Dict[str, Any]] = []
        if args.accept_all:
            to_accept = list(pending_all)
        elif args.accept_tier1:
            to_accept = [p for p in pending_all if int(p.get("tier", 1)) == 1]
        for pref in (args.accept or []):
            p = _pick_by_prefix(pending_all, "proposal_id", pref, "pending proposal")
            if p not in to_accept:
                to_accept.append(p)
        relabels: List[Tuple[Dict[str, Any], str]] = []
        for spec in (args.relabel or []):
            pref, _, cls = spec.partition(":")
            if not cls.strip():
                raise SystemExit(f"--relabel needs PROPOSAL:CLASS, got {spec!r}")
            p = _pick_by_prefix(pending_all, "proposal_id", pref, "pending proposal")
            relabels.append((p, cls.strip()))
        as_marks: List[Tuple[Dict[str, Any], str]] = []
        for spec in (args.accept_as_mark or []):
            pref, _, cls = spec.partition(":")
            p = _pick_by_prefix(pending_all, "proposal_id", pref, "pending proposal")
            as_marks.append((p, cls.strip() or str(p.get("category"))))
        chosen_ids = {id(p) for p, _ in relabels} | {id(p) for p, _ in as_marks}
        to_accept = [p for p in to_accept if id(p) not in chosen_ids]
        done = 0
        for p, cls in [(p, None) for p in to_accept] + relabels:
            category = cls or str(p.get("category"))
            note = p.get("rationale")
            if cls:
                note = f"relabeled from {p.get('category')} by {args.actor}: {note or ''}".strip()
            cid = await commit_stratum_correction(
                queue, cap, sid, list(p.get("segment_ids") or []), category,
                sess.id, skeleton_hash=skel, start_time=p.get("start_time"),
                end_time=p.get("end_time"), proposal_id=p.get("proposal_id"),
                proposal_set_id=set_id, actor=args.actor, note=note,
                journal_path=jp)
            done += 1
            print(f"{'relabeled' if cls else 'accepted'} {_short(p.get('proposal_id'))} -> "
                  f"stratum {_short(cid)} {category} {float(p.get('start_time') or 0):.1f}-"
                  f"{float(p.get('end_time') or 0):.1f}s")
        for p, cls in as_marks:
            note = f"from proposal ({(m.get('model') or {}).get('name')}): {p.get('rationale') or ''}".strip()
            first = None
            for seg_id in (p.get("segment_ids") or []):
                mid = await commit_mark_correction(
                    queue, cap, sid, {"kind": "segment", "segment_id": seg_id}, cls,
                    sess.id, actor=args.actor, note=note, journal_path=jp,
                    proposal_id=p.get("proposal_id"), proposal_set_id=set_id)
                first = first or mid
            done += 1
            print(f"marked {_short(p.get('proposal_id'))} -> mark {_short(first)} {cls} "
                  f"x{len(p.get('segment_ids') or [])} seg(s) "
                  f"{float(p.get('start_time') or 0):.1f}-{float(p.get('end_time') or 0):.1f}s")
        for pref in (args.retract or []):
            c = _pick_by_prefix(strata, "id", pref, "live stratum")
            rid = await commit_stratum_retraction(queue, cap, sid, c["id"], sess.id,
                                                  actor=args.actor, note=args.note,
                                                  journal_path=jp)
            print(f"retracted stratum {_short(c['id'])} "
                  f"({(c.get('payload') or {}).get('category')}) via {_short(rid)}")
        if args.watermark is not None:
            wm_arg = str(args.watermark).strip().lower()
            wm: Optional[float]
            if wm_arg == "end":
                segs = await load_source_segments(queue, cap, sid,
                                                  rendition_selector=args.rendition,
                                                  skeleton_selector=args.skeleton)
                ends = [float(s.end_time) for s in segs if s.end_time is not None]
                if not ends:
                    raise SystemExit("--watermark end: the spine has no timed segments")
                wm = max(ends)
            elif wm_arg == "none":
                wm = None      # nothing visited — no negative region is valid below anything
            else:
                wm = float(args.watermark)
            gid = await commit_extraction_gate(
                queue, cap, sid, skel, str(gate.get("extraction_status") or "in_progress"),
                wm, session_id=sess.id, actor=args.actor, journal_path=jp, lane=FILTER_LANE)
            print(f"lane watermark asserted: annotated_through "
                  f"{('%.1fs' % wm) if wm is not None else 'none'} ({_short(gid)})")
        await set_session_status(queue, cap, sess.id, "completed", journal_path=jp,
                                 actor=args.actor)
        print(f"session {_short(sess.id)} completed · {done} accepted · journal "
              f"{jp or 'none'}")
    finally:
        await _close_graph_stack(manager, queue, cap)
    return 0


async def scan_command(
    args: argparse.Namespace,  # Parsed args for the `scan-mishomed` subcommand
) -> int:  # Process exit code (0 clean; 1 when mis-homed words found and --strict)
    """Execute `scan-mishomed`: flag authoritative FA words stranded outside
    every chunk of a spine (96edc646 verdict bc7ece7b — the mis-homed-text
    detector, productized as the standing QA gate).

    Two mechanisms strand real speech outside chunks — the carve's sliver
    guard (M1) and VAD-missed speech in inter-chunk gaps (M2) — and the fold
    then homes those words into the NEAREST chunk, whose audio does not
    contain them. The pipeline computed the incriminating word times and
    discarded them; this verb recovers them from the forced-alignment
    capability's CACHE (word-level items keyed by sha256 of the aligned text
    — a pragmatic read of a local durable artifact; if the cache schema ever
    moves, the v2 path is an align call through the capability seam, which
    cache-hits to the same rows). Join chain, fully typed: chunk.text_from ->
    Transcript(text, rendition_id) -> AudioRendition(audio_segment_id) ->
    AudioSegment(start offset). Classification against the source's LATEST
    proposal set (auto-discovered): a gap whose unexplained stretch >= 0.5s =
    vad-gap, else carve-sliver; no set = unclassified. Reads only."""
    ws = resolve_workspace(explicit=getattr(args, "workspace", None))
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    if args.manifests_dir is None:
        args.manifests_dir = (str(ws.substrate_data_dir / "manifests")
                              if ws is not None else ".cjm/manifests")
    fa_cache = (Path(args.fa_cache_db) if args.fa_cache_db
                else (ws.substrate_data_dir / "data" / "cjm-capability-qwen3-forced-aligner"
                      / "forced_alignments.db" if ws is not None else None))
    if fa_cache is None or not fa_cache.is_file():
        raise SystemExit(f"forced-alignment cache not found ({fa_cache}) — "
                         "pass --fa-cache-db")
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

        segs = await load_source_segments(
            queue, args.graph_capability, sid,
            rendition_selector=args.rendition, skeleton_selector=args.skeleton)
        chunks = sorted((s for s in segs if s.start_time is not None),
                        key=lambda s: float(s.start_time))
        if not chunks:
            raise SystemExit("empty spine (0 timed segments)")

        async def _props(node_id: str) -> Dict:
            node = await graph_task(queue, args.graph_capability, "get_node",
                                    node_id=node_id)
            if node is None:
                return {}
            d = node if isinstance(node, dict) else node.to_dict()
            return d.get("properties") or {}

        words: List[Dict] = []  # {"s","e","text"} in source seconds
        unmatched = 0
        fa = sqlite3.connect(f"file:{fa_cache}?mode=ro", uri=True)
        try:
            for tid in sorted({s.text_from for s in chunks if s.text_from}):
                tp = await _props(tid)
                text, rid = str(tp.get("text") or ""), tp.get("rendition_id")
                if not text or not rid:
                    unmatched += 1
                    continue
                base = None
                aseg_id = (await _props(rid)).get("audio_segment_id")
                if aseg_id:
                    base = (await _props(aseg_id)).get("start")
                if base is None:
                    unmatched += 1
                    continue
                th = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
                row = fa.execute("SELECT items FROM forced_alignments WHERE text_hash=? "
                                 "ORDER BY created_at DESC LIMIT 1", (th,)).fetchone()
                if not row:
                    unmatched += 1
                    continue
                for w in json.loads(row[0]):
                    words.append({"s": float(base) + float(w["start_time"]),
                                  "e": float(base) + float(w["end_time"]),
                                  "text": str(w.get("text") or "")})
        finally:
            fa.close()
        words.sort(key=lambda w: w["s"])

        # Mis-homed = the chunk the FOLD assigns the word to contains
        # (essentially) none of the word's audio. Assignment replicates the
        # CURRENT assign_words_to_chunks (word-rescue/v4 fold rule): argmax
        # overlap, with start-containment/nearest-edge only as the
        # zero-overlap fallback — so a word merely clipped by a boundary is
        # precision, not mis-homing. On spines folded before v4 the scan
        # mildly under-reports (older folds mis-homed MORE). The residue this
        # flags on a v4 spine = words FA placed FULLY inside verified event
        # spans — correction-lane material, structurally unfixable without
        # re-embedding the event.
        bounds = [(float(c.start_time), float(c.end_time)) for c in chunks]
        flagged = []
        for w in words:
            if w["e"] <= w["s"]:
                continue
            home = None
            best_ov = 0.0
            for s, e in bounds:
                ov = min(w["e"], e) - max(w["s"], s)
                if ov > best_ov:
                    best_ov, home = ov, (s, e)
            if best_ov <= 0.0:
                best_d = float("inf")
                for s, e in bounds:
                    if s <= w["s"] < e:
                        home = (s, e)
                        break
                    d = min(abs(w["s"] - s), abs(w["s"] - e))
                    if d < best_d:
                        best_d, home = d, (s, e)
            overlap = (min(w["e"], home[1]) - max(w["s"], home[0])) if home else 0.0
            if overlap < args.min_overlap:
                flagged.append(w)

        ps = load_event_proposal_set(str(ws.root), source_id=sid) if ws else None
        spans = sorted((float(r["start_time"]), float(r["end_time"]))
                       for r in (ps["proposals"] if ps else []))

        instances: List[Dict] = []
        for w in flagged:
            prev_c = max((c for c in chunks if float(c.end_time) <= w["s"] + 0.02),
                         key=lambda c: float(c.end_time), default=None)
            next_c = min((c for c in chunks if float(c.start_time) >= w["e"] - 0.02),
                         key=lambda c: float(c.start_time), default=None)
            key = (prev_c.index if prev_c else -1, next_c.index if next_c else -1)
            if instances and instances[-1]["key"] == key:
                instances[-1]["words"].append(w)
            else:
                instances.append({"key": key, "words": [w],
                                  "gap": (float(prev_c.end_time) if prev_c else 0.0,
                                          float(next_c.start_time) if next_c else float("inf"))})
        for inst in instances:
            g0, g1 = inst["gap"]
            cursor, unexplained = g0, 0.0
            for s, e in ((max(s, g0), min(e, g1)) for s, e in spans if s < g1 and e > g0):
                unexplained = max(unexplained, s - cursor)
                cursor = max(cursor, e)
            unexplained = max(unexplained, g1 - cursor)
            inst["mech"] = (("vad-gap" if unexplained >= 0.5 else "carve-sliver")
                            if ps else "unclassified")

        by: Dict[str, int] = {}
        for inst in instances:
            by[inst["mech"]] = by.get(inst["mech"], 0) + 1
        print(f"chunks {len(chunks)} · FA words {len(words)} · "
              f"transcripts without FA/aseg match {unmatched}")
        print(f"mis-homed words {len(flagged)} -> instances {len(instances)}"
              + (f"  ({' · '.join(f'{k} {v}' for k, v in sorted(by.items()))})"
                 if instances else ""))
        for inst in instances:
            txt = " ".join(w["text"] for w in inst["words"])
            print(f"  between #{inst['key'][0]} and #{inst['key'][1]} "
                  f"[{inst['mech']}] {txt!r} "
                  f"@ {inst['words'][0]['s']:.3f}-{inst['words'][-1]['e']:.3f}s")
        if instances and args.strict:
            return 1
    finally:
        await queue.stop()
        try:
            manager.unload_capability(args.graph_capability)
        except Exception as e:  # Best-effort teardown; never mask the scan outcome
            logger.warning(f"unload {args.graph_capability} failed: {e}")
    return 0


async def run_extract(
    queue: Any,                 # Started JobQueue over a manager holding the graph seat
    graph_capability: str,      # Graph capability instance id
    *,
    ws: Any = None,             # Resolved Workspace (None = cwd-relative datasets/)
    source: Optional[str] = None,        # Source node id / title substring filter
    rendition: Optional[str] = None,     # AudioRendition selector for multi-rendition sources
    include_purposes: Optional[List[str]] = None,  # Session purposes whose spans qualify as EXAMPLES
    output_dir: Optional[str] = None,    # Datasets root override
    graph_db_path: Optional[str] = None, # Journal-family provenance (else the seat's persisted config)
    manager: Any = None,        # Manager owning the seat (db-path fallback lookup)
    log: Any = print,           # Line sink (the flywheel page passes a collector)
) -> Optional[Dict[str, Any]]:  # Saved DatasetManifest dict (+ "_path"), or None when no Source matches
    """The extract fold on an ALREADY-OPEN graph seat (flywheel build leg 2,
    DECs d02a38d4 + 16159e09 + a5883992 + 8e05b87b; lifted out of the CLI for
    the correction TUI's cross-source flywheel page, DEC 82c463fe): per gated
    spine, labeled insert spans become examples (genuine-purpose sessions by
    default), speech + negative regions derive below the annotated_through
    watermark, and the dataset lands workspace-local under
    datasets/<dataset_id>/ with a DatasetManifest recording config, consumed
    journal family, split/augmentation policy as DATA, the observed open class
    vocabulary, and every spine's gate state at extraction time."""
    res = await graph_task(queue, graph_capability, "query_nodes",
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
    if source:
        needle = source.lower()
        sources = [s for s in sources
                   if s["id"] == source or needle in s["title"].lower()]
    if not sources:
        log("no matching Source nodes")
        return None
    purposes = list(include_purposes or ["genuine"])
    sres = await graph_task(queue, graph_capability, "query_nodes",
                            query=NodeQuery(label="CorrectionSession").to_dict())
    include_ids = set()
    for n in (sres.nodes or []):
        d = n.to_dict() if hasattr(n, "to_dict") else n
        purpose = str((d.get("properties") or {}).get("purpose") or "genuine")
        if purpose in purposes:
            include_ids.add(d["id"])

    dataset_id = new_dataset_id()
    created_at = time.time()
    out_root = (Path(output_dir) if output_dir
                else (ws.root / "datasets" if ws is not None else Path("datasets")))
    ddir = out_root / dataset_id
    events: List[Dict] = []
    regions: List[Dict] = []
    spine_records: List[Dict] = []
    vocab: Dict[str, int] = {}
    skipped_totals: Dict[str, int] = {}
    for src in sources:
        corrections, superseded = await load_source_corrections(
            queue, graph_capability, src["id"])
        spines = await list_source_spines(queue, graph_capability, src["id"],
                                          rendition_selector=rendition)
        gates = await load_extraction_gates(queue, graph_capability, src["id"])
        log(f"== {src['title'] or src['id']}  ({src['id']}) ==")
        if not spines:
            log("  (no decomposed spine)")
        for sp in spines:
            h = sp.get("skeleton_hash")
            gate = gates.get(h)
            status = str((gate or {}).get("extraction_status") or "in_progress")
            extractable = (status != "excluded"
                           and (gate or {}).get("annotated_through") is not None)
            # Gate first: an ineligible spine never pays the full-spine read.
            segs = (await load_source_segments(
                queue, graph_capability, src["id"],
                rendition_selector=rendition,
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
                log(f"  spine {_spine_tag(h)}: {r['status']} · annotated_through "
                    f"{wm_txt} — not extractable")
                continue
            o_rows, o_foreign = overlay_event_rows(
                src, h, r["overlays"], {s.id for s in segs})
            rec["overlays"] = len(o_rows)
            if o_foreign:
                skipped_totals["overlay_foreign_spine"] = (
                    skipped_totals.get("overlay_foreign_spine", 0) + o_foreign)
            log(f"  spine {_spine_tag(h)}: {r['status']} @ {wm_txt} — "
                f"examples {len(r['examples'])} · overlays {len(o_rows)} · "
                f"speech {len(r['speech'])} · negatives {len(r['negatives'])}"
                + (f" · skipped {r['skipped']}" if r["skipped"] else ""))
            for row in o_rows:
                vocab[row["label"]] = vocab.get(row["label"], 0) + 1
                events.append(row)
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
    db = graph_db_path or (
        (manager.instances[graph_capability].config or {}).get("db_path")
        if manager is not None else None)
    stem = sidecar_journal_path(db)[: -len(".jsonl")] if db else None
    journals = (sorted(str(p) for p in Path(db).parent.glob(Path(stem).name + "*.jsonl"))
                if db else [])
    manifest = DatasetManifest(
        dataset_id=dataset_id, created_at=created_at,
        config={"graph_capability": graph_capability,
                "source": source, "rendition": rendition,
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
    log(f"== DATASET {dataset_id} ==")
    log("  classes: "
        + (" · ".join(f"{k}x{v}" for k, v in sorted(vocab.items())) or "none"))
    log(f"  examples: {len(events)}  regions: {len(regions)}  "
        f"spines: {sum(1 for r in spine_records if r['eligible'])} extractable "
        f"of {len(spine_records)}")
    log(f"  manifest: {out}")
    d = manifest.to_dict()
    d["_path"] = str(out)
    return d
