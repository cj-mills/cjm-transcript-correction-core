"""THE PER-VERB CRASH PROOF of the correction core (eeda5977 leg L3 slice E3, rulings 6683f49e +
3de795e4), on the layer's harness (`cjm_context_graph_layer.testing.run_proof`).

Every journaled correction verb runs as a step in a child process on a real SQLite graph, uncrashed
and then crashed at every op boundary (a unit torn mid-append, after its append, after each apply);
a human step is healed by the next writer (a torn unit, which landed nothing, is the user's redo),
the seed's pipeline extension re-run. Each run's live db equals its journal's replay element for
element, and every crashed run's journal equals the uncrashed one, op for op. Two app-shaped
gestures ride along as units: a filter-fix (a mark, a text fix, its marker, the discharge) and a
relabel (an insert's removal and its replacement) -- correction-qt's multi-commit gestures.

Needs the graph capability: CJM_PROOF_MANIFESTS, else this repo's .cjm/manifests."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from cjm_context_graph_layer.journal import journal_extend, write_window
from cjm_context_graph_layer.testing import crash_variants, ProofStep, run_proof
from cjm_transcript_correction_core import graph as G

MANIFESTS = os.environ.get("CJM_PROOF_MANIFESTS") or str(Path(__file__).resolve().parents[1] / ".cjm" / "manifests")
pytestmark = pytest.mark.skipif(not (Path(MANIFESTS) / "cjm-capability-graph-sqlite.json").exists(),
                                reason=f"graph capability not installed at {MANIFESTS}")

SRC = "5f0c0a1e-0000-4000-8000-00000000c0c0"
SEG = [f"5f0c0a1e-0000-4000-8000-00000000c00{i}" for i in range(6)]
SKELETON = "sha256:proof-skeleton"


def _j(ctx):  # Every verb journals to the run's sidecar
    return {"journal_path": ctx.journal}


def _sess(ctx):
    return ctx.record("session", "session-start")["id"]


def _node(ctx, step, verb):  # The first node a step's op minted
    return ctx.record(step, verb)["wires"]["nodes"][0]


async def seed(ctx):  # Layer 0 as a pipeline leaves it: a Source and its fine spine
    nodes = [{"id": SRC, "label": "Source", "properties": {"title": "Proof"}}]
    nodes += [{"id": s, "label": "Segment",
               "properties": {"index": i, "source_id": SRC, "skeleton_hash": SKELETON,
                              "start_time": 10.0 * i, "end_time": 10.0 * (i + 1), "text": f"segment {i}"}}
              for i, s in enumerate(SEG)]
    edges = [{"id": f"proof-part-{s}", "source_id": s, "target_id": SRC, "relation_type": "PART_OF",
              "properties": {}} for s in SEG]
    await journal_extend(ctx.queue, ctx.graph_id, nodes, edges, verb="spine-extension",
                         actor="pipeline:proof", args={"source_id": SRC}, **_j(ctx))


async def session(ctx):
    await G.start_session(ctx.queue, ctx.graph_id, [SRC], **_j(ctx))


async def text(ctx):
    await G.commit_text_correction(ctx.queue, ctx.graph_id, SRC, SEG[0], "fixed", _sess(ctx),
                                   old_text="segment 0", **_j(ctx))


async def reedit(ctx):
    await G.commit_text_correction(ctx.queue, ctx.graph_id, SRC, SEG[0], "fixed again", _sess(ctx),
                                   old_text="fixed", supersedes_id=_node(ctx, "text", "text-correction")["id"],
                                   **_j(ctx))


async def shift(ctx):
    await G.commit_boundary_shift_correction(ctx.queue, ctx.graph_id, SRC, SEG[0], SEG[1], "again",
                                             "push", _sess(ctx), **_j(ctx))


async def prune(ctx):
    await G.commit_prune_correction(ctx.queue, ctx.graph_id, SRC, [SimpleNamespace(id=SEG[4])],
                                    _sess(ctx), **_j(ctx))


async def amend(ctx):
    prior = _node(ctx, "prune", "prune")
    await G.commit_prune_amendment(ctx.queue, ctx.graph_id, {"id": prior["id"], **prior["properties"]},
                                   [SEG[4]], _sess(ctx), **_j(ctx))


async def mark(ctx):
    await G.commit_mark_correction(ctx.queue, ctx.graph_id, SRC, {"kind": "segment", "segment_id": SEG[2]},
                                   "needs-review", _sess(ctx), **_j(ctx))


async def dismiss(ctx):
    await G.commit_mark_dismissal(ctx.queue, ctx.graph_id, SRC, _node(ctx, "mark", "mark")["id"],
                                  _sess(ctx), **_j(ctx))


async def filter_fix(ctx):  # correction-qt's filter-fix + its discharge: ONE gesture, ONE unit
    async with write_window(ctx.queue, ctx.graph_id, ctx.journal, unit=True):
        mark_id = await G.commit_mark_correction(ctx.queue, ctx.graph_id, SRC,
                                                 {"kind": "segment", "segment_id": SEG[3]},
                                                 "needs-review", _sess(ctx), actor="capability:proof",
                                                 **_j(ctx))
        fix = await G.commit_text_correction(ctx.queue, ctx.graph_id, SRC, SEG[3], "segment three",
                                             _sess(ctx), old_text="segment 3", **_j(ctx))
        await G.discharge_mark(ctx.queue, ctx.graph_id, SRC, mark_id, fix, _sess(ctx), **_j(ctx))


async def markers(ctx):
    await G.record_review_markers(ctx.queue, ctx.graph_id, _sess(ctx), [(SEG[5], "ok")], **_j(ctx))


async def stratum(ctx):
    await G.commit_stratum_correction(ctx.queue, ctx.graph_id, SRC, [SEG[5]], "music", _sess(ctx),
                                      skeleton_hash=SKELETON, **_j(ctx))


async def stratum_retract(ctx):
    await G.commit_stratum_retraction(ctx.queue, ctx.graph_id, SRC, _node(ctx, "stratum", "stratum")["id"],
                                      _sess(ctx), **_j(ctx))


async def nudge(ctx):
    await G.commit_time_nudge_correction(ctx.queue, ctx.graph_id, SRC,
                                         [{"segment_id": SEG[1], "edge": "end", "old_time": 20.0,
                                           "new_time": 19.5}], _sess(ctx), step_s=-0.5, **_j(ctx))


async def insert(ctx):
    await G.commit_chunk_insert_correction(ctx.queue, ctx.graph_id, SRC, SEG[1], 20.0, 21.0, _sess(ctx),
                                           before_segment_id=SEG[2], label="music", **_j(ctx))


async def relabel(ctx):  # correction-qt's relabel: the insert's removal + its replacement, ONE unit
    old = ctx.record("insert", "chunk-insert")["id"]
    async with write_window(ctx.queue, ctx.graph_id, ctx.journal, unit=True):
        await G.commit_chunk_insert_removal(ctx.queue, ctx.graph_id, SRC, old, _sess(ctx), **_j(ctx))
        await G.commit_chunk_insert_correction(ctx.queue, ctx.graph_id, SRC, SEG[1], 20.0, 21.0,
                                               _sess(ctx), before_segment_id=SEG[2], label="laughter",
                                               **_j(ctx))


async def split(ctx):
    await G.commit_chunk_split_correction(ctx.queue, ctx.graph_id, SRC, SEG[2], 25.0, "segment", "2",
                                          30.0, _sess(ctx), SEG[2], before_segment_id=SEG[3],
                                          old_text="segment 2", **_j(ctx))


async def split_remove(ctx):
    op = ctx.record("split", "chunk-split")
    group = [n["id"] for n in op["wires"]["nodes"] if n["id"] != op["id"]]
    await G.commit_chunk_split_removal(ctx.queue, ctx.graph_id, SRC, op["id"], group, _sess(ctx), **_j(ctx))


async def entity(ctx):
    await G.commit_speaker_entity(ctx.queue, ctx.graph_id, "Ada", _sess(ctx), **_j(ctx))


async def rename(ctx):
    await G.rename_speaker_entity(ctx.queue, ctx.graph_id, _node(ctx, "entity", "speaker-entity")["id"],
                                  "Ada Lovelace", _sess(ctx), **_j(ctx))


async def assign(ctx):
    await G.commit_speaker_assign_correction(ctx.queue, ctx.graph_id, SRC, [SEG[0], SEG[1]],
                                             _node(ctx, "entity", "speaker-entity")["id"], _sess(ctx),
                                             **_j(ctx))


async def gate(ctx):
    await G.commit_extraction_gate(ctx.queue, ctx.graph_id, SRC, SKELETON, "in_progress", 30.0,
                                   session_id=_sess(ctx), **_j(ctx))


async def overlay(ctx):
    await G.commit_speech_overlay_correction(
        ctx.queue, ctx.graph_id, SRC,
        {"kind": "span", "segment_id": SEG[5], "char_start": 0, "char_end": 7, "text_snapshot": "segment"},
        "laughter", 50.0, 51.0, "segment", _sess(ctx), snap="estimated", **_j(ctx))


async def overlay_remove(ctx):
    await G.commit_speech_overlay_removal(ctx.queue, ctx.graph_id, SRC,
                                          _node(ctx, "overlay", "speech-overlay")["id"], _sess(ctx), **_j(ctx))


async def purpose(ctx):
    await G.set_session_purpose(ctx.queue, ctx.graph_id, _sess(ctx), "feature-test", **_j(ctx))


async def status(ctx):
    await G.set_session_status(ctx.queue, ctx.graph_id, _sess(ctx), "completed", **_j(ctx))


STEPS = [
    ProofStep("seed", seed, ("spine-extension",), retried=True),
    ProofStep("session", session, ("session-start",)),
    ProofStep("text", text, ("text-correction", "review-markers"), unit=True),
    ProofStep("reedit", reedit, ("text-correction", "review-markers"), unit=True),
    ProofStep("shift", shift, ("boundary-shift", "review-markers"), unit=True),
    ProofStep("prune", prune, ("prune", "review-markers"), unit=True),
    ProofStep("amend", amend, ("prune-amendment",)),
    ProofStep("mark", mark, ("mark",)),
    ProofStep("dismiss", dismiss, ("mark-dismiss",)),
    ProofStep("filter_fix", filter_fix, ("mark", "text-correction", "review-markers", "mark-discharge"),
              unit=True),
    ProofStep("markers", markers, ("review-markers",)),
    ProofStep("stratum", stratum, ("stratum",)),
    ProofStep("stratum_retract", stratum_retract, ("stratum-retract",)),
    ProofStep("nudge", nudge, ("time-nudge",)),
    ProofStep("insert", insert, ("chunk-insert",)),
    ProofStep("relabel", relabel, ("chunk-insert-remove", "chunk-insert"), unit=True),
    ProofStep("split", split, ("chunk-split",)),
    ProofStep("split_remove", split_remove, ("chunk-split-remove",)),
    ProofStep("entity", entity, ("speaker-entity",)),
    ProofStep("rename", rename, ("entity-rename",)),
    ProofStep("assign", assign, ("speaker-assign",)),
    ProofStep("gate", gate, ("extraction-gate",)),
    ProofStep("overlay", overlay, ("speech-overlay",)),
    ProofStep("overlay_remove", overlay_remove, ("speech-overlay-remove",)),
    ProofStep("purpose", purpose, ("session-purpose",)),
    ProofStep("status", status, ("session-status",)),
]


@pytest.fixture(scope="module")
def uncrashed(tmp_path_factory):
    return run_proof(tmp_path_factory.mktemp("uncrashed"), MANIFESTS, STEPS)


def test_every_correction_verb_is_proven_by_a_step(uncrashed):
    from cjm_transcript_correction_core.journal import correction_replay_handlers
    proven = {v for s in STEPS for v in s.ops} - {"spine-extension"}
    assert proven == set(correction_replay_handlers())
    assert all(uncrashed["counts"])


@pytest.mark.parametrize("variant", crash_variants(STEPS), ids=lambda v: f"{v[0]}@{v[1]}")
def test_a_crashed_verb_heals_to_the_uncrashed_run(tmp_path, uncrashed, variant):
    crashed = run_proof(tmp_path, MANIFESTS, STEPS, variant)
    assert crashed["counts"] == uncrashed["counts"]
    assert crashed["journal"] == uncrashed["journal"]
