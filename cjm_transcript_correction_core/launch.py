"""The shared launch surface every correction shell drives through: the
argument surface (build_parser) and the resolution ladder (resolve_settings).
Absorbed from the Textual shell's cli module (spine absorption 12f342f1) so
every shell imports the SAME ladder (DEC 0f11683d — the 6c574c89 extraction
replayed on the correction lane): a resolution drift between shells would
fork which stack a correction session opens."""

import argparse
import os
from typing import Any, Dict

from cjm_substrate.core.workspace import resolve_workspace


def build_parser() -> argparse.ArgumentParser:  # Configured CLI parser
    """The TUI driver's argument surface (mirrors correction-core's run/review args)."""
    p = argparse.ArgumentParser(
        prog="cjm-transcript-correction-tui",
        description="Keyboard-first correction loop over a transcription context graph "
                    "(document-order segment walk, VAD-chunk auto-play, fidelity edits).")
    p.add_argument("--graph-db-path", default=None,
                   help="The shared transcription graph db (the committed spine); "
                        "default: the graph capability's persisted config — under an "
                        "active workspace the config store is workspace-scoped, so "
                        "the workspace names the db (2ce81638)")
    p.add_argument("--source", default=None,
                   help="Source node id or title substring; omitted or ambiguous -> "
                        "the in-TUI source picker (correction status at a glance)")
    p.add_argument("--manifests-dir", default=None,
                   help="Capability manifests directory (default: the workspace's "
                        ".cjm/manifests when one is active, else .cjm/manifests under the cwd)")
    p.add_argument("--workspace", default=None,
                   help="Workspace root (5daadfc4; default: CJM_WORKSPACE env, else upward walk "
                        "from cwd). Supplies the manifests default and is exported so capability "
                        "workers resolve workspace-scoped paths and the config store "
                        "supplies the graph db default (2ce81638 discovery is built: "
                        "no --source -> in-TUI picker)")
    p.add_argument("--rendition", default=None,
                   help="AudioRendition selector when a source has more than one "
                        "(\"raw\" or a preprocessing substring); default: auto-select")
    p.add_argument("--skeleton", default=None,
                   help="Skeleton-spine selector when several coexist under one rendition "
                        "(sentence-split, DEC f1024568): \"legacy\" or a skeleton-hash prefix; "
                        "default: the in-TUI spine picker (choice persists in the sidecar)")
    p.add_argument("--actor", default="human",
                   help="Actor recorded on corrections + review markers")
    p.add_argument("--lane", choices=("walk", "assign", "propose", "annotate"), default=None,
                   help="Starting pass lane (multi-lane workbench, DEC cc55a7b5): "
                        "walk = the correction vocabulary, assign = speaker assignment, "
                        "annotate = word-span sample creation (fc42614d) "
                        "(tab cycles in-TUI; default: the sidecar-persisted preference, "
                        "else walk)")
    p.add_argument("--fa-cache-db", default=None,
                   help="Forced-alignment cache db supplying word timestamps for the "
                        "annotate lane's snap-to-word (default: the workspace's "
                        "cjm-capability-qwen3-forced-aligner cache; missing = spans "
                        "estimate from character fractions instead)")
    p.add_argument("--test", action="store_true",
                   help="Tag the minted session purpose=\"feature-test\" — feature-test "
                        "passes are structurally excludable from flywheel datasets "
                        "(noise hygiene, DEC c86714a4); genuine passes omit the tag")
    p.add_argument("--no-autoplay", action="store_true",
                   help="Do not auto-play the focused segment's VAD chunk")
    p.add_argument("--audio-device", default=None,
                   help="Output device index or name substring (default: the system "
                        "default sink — pipewire/pulse routing when available)")
    p.add_argument("--no-resume", action="store_true",
                   help="Start at segment 0 instead of the source's last-focused segment")
    p.add_argument("--shift-floor-ms", type=int, default=0,
                   help="Minimum milliseconds between held-key boundary shifts; 0 = ungoverned "
                        "(the async commit guard is the real governor — a 1ms floor read as "
                        "residual keystroke latency in the 2026-07-14 drive). "
                        "Measure key rates with tests_manual/keyrate_probe.py")
    p.add_argument("--nudge-step-ms", type=float, default=None,
                   help="Boundary time-nudge step per ,/. (end) or </> (start) press, "
                        "milliseconds. Adjustable IN-TUI with { } along the "
                        "5/10/20/50/100/200/500 ladder (the choice persists in the "
                        "sidecar); this flag overrides the persisted preference "
                        "(default: sidecar, else 100)")
    return p


def resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:  # Resolved launch surface
    """Resolve the launch settings every shell shares (5daadfc4 workspace
    resolution + the defaults derived from it), exporting CJM_WORKSPACE so
    capability workers resolve workspace-scoped paths and the config store
    supplies the graph db default (2ce81638). Extracted from main() so the Qt
    shell's driver reuses the EXACT ladder (DEC 0f11683d — the 6c574c89
    extraction replayed on the correction lane): a resolution drift between
    shells would fork which stack a correction session opens. Mutates
    args.manifests_dir to its resolved default (the shells read it there)."""
    ws = resolve_workspace(explicit=args.workspace)
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    if args.manifests_dir is None:
        args.manifests_dir = (str(ws.substrate_data_dir / "manifests")
                              if ws is not None else ".cjm/manifests")
    device = args.audio_device
    if device is not None and device.isdigit():
        device = int(device)
    return {"manifests_dir": args.manifests_dir,
            "audio_device": device,
            "shift_floor_s": args.shift_floor_ms / 1000.0,
            "purpose": "feature-test" if args.test else None}
