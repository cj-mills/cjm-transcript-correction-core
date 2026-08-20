"""Sidecar view-state + spine-picker helpers — the correction TUI's pure
state module (Textual-free by construction), gathered out of app.py's tail so
non-Textual shells (the Qt port) import bookmark/preference persistence and
the spine-picker labels without dragging the Textual app in (DEC 0f11683d)."""

import time
from typing import Any, Dict, Optional

from cjm_substrate.utils.sidecar import SidecarState
from cjm_transcript_correction_core.graph import LEGACY_SKELETON


def save_tui_state(
    graph_db_path: str,  # The graph db whose sidecar state file to write
    source_id: str,      # Source whose position is being remembered
    cursor: Optional[int],  # Last-focused segment position (None = leave as-is)
    speed: Optional[float] = None,  # Playback-rate preference (db-wide `_speed`; None = leave as-is)
    mark_class: Optional[str] = None,  # Last-used ⚑ class (db-wide `_mark_class`; None = leave as-is)
    insert_label: Optional[str] = None,  # Last-used ⊕ insert label (db-wide `_insert_label`; None = leave as-is)
    overlay_label: Optional[str] = None,  # Last-used ◈ overlay label (db-wide `_overlay_label`; None = leave as-is)
    nudge_step_ms: Optional[float] = None,  # Nudge-step preference (db-wide `_nudge_step_ms`; None = leave as-is)
    lane: Optional[str] = None,      # Pass-lane preference (db-wide `_lane`; None = leave as-is)
    fold_wordless: Optional[bool] = None,  # z fold preference (db-wide `_fold_wordless`; None = leave as-is)
    skeleton: Optional[str] = None,  # Chosen skeleton-spine selector (per-source; None = leave as-is)
    spines: Optional[int] = None,    # Spine-set size the choice was made against (re-prompt key)
) -> None:
    """Merge one source's view state into the sidecar state file.

    VIEW state, not knowledge — it lives in a local sidecar next to the db,
    never as a graph write (the cursor is where the eye was, not a decision;
    the spine CHOICE is a view preference too — the graph-asserted active
    spine stays deferred per DEC f1024568). Per-source entries MERGE so a
    cursor write never drops the spine choice and vice versa. Write failures
    are silently tolerated: losing a bookmark must never break the loop."""
    store = SidecarState(f"{graph_db_path}.tui-state.json")
    state = store.load()
    entry = dict(state.get(source_id) or {})
    if cursor is not None:
        entry["cursor"] = int(cursor)
    entry["ts"] = time.time()
    if skeleton is not None:
        entry["skeleton"] = str(skeleton)
    if spines is not None:
        entry["spines"] = int(spines)
    state[source_id] = entry
    if speed is not None:
        state["_speed"] = float(speed)
    if mark_class is not None:
        state["_mark_class"] = str(mark_class)
    if insert_label is not None:
        state["_insert_label"] = str(insert_label)
    if overlay_label is not None:
        state["_overlay_label"] = str(overlay_label)
    if nudge_step_ms is not None:
        state["_nudge_step_ms"] = float(nudge_step_ms)
    if lane is not None:
        state["_lane"] = str(lane)
    if fold_wordless is not None:
        state["_fold_wordless"] = bool(fold_wordless)
    store.write(state)


def load_tui_state(
    graph_db_path: str,  # The graph db whose sidecar state file to read
) -> Dict[str, Any]:  # {source_id: {"cursor": int, "ts": float}}; empty when absent/corrupt
    """Read the per-graph TUI sidecar state (last-focused positions)."""
    return SidecarState(f"{graph_db_path}.tui-state.json").load()


def spine_label(
    spine: Dict[str, Any],  # One list_source_spines row
) -> str:  # Picker-row config summary
    """One picker row's config summary for a skeleton spine (pure).

    Legacy (no skeleton_hash) reads as the incumbent VAD-only spine; split
    spines show their policy tag + a hash prefix (the persisted selector value
    stays the FULL hash — see selector_for_spine)."""
    h = spine.get("skeleton_hash")
    if not h:
        return "vad-only (pre-split)"
    tag = spine.get("split_policy") or "vad-only"
    return f"{tag} · {str(h).split(':')[-1][:8]}"


def selector_for_spine(
    spine: Dict[str, Any],  # One list_source_spines row
) -> str:  # The --skeleton selector naming this spine
    """The selector value a picker choice persists (pure): the full skeleton
    hash, or the LEGACY_SKELETON token for the pre-split spine."""
    return str(spine.get("skeleton_hash") or LEGACY_SKELETON)
