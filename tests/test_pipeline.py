"""Tests for cjm_transcript_correction_core.pipeline — pure-logic checks (no runtime).

Projected from the pipeline notebook's smoke-check cell at the golden-reference flip."""
from cjm_transcript_correction_core.models import SpineSegment
from cjm_transcript_correction_core.pipeline import compute_worklist, resolve_graph_db_path

SEGS = [
    SpineSegment(id="a", index=0, text="The art of war"),
    SpineSegment(id="b", index=1, text=""),
    SpineSegment(id="c", index=2, text="is of vital importance."),
    SpineSegment(id="d", index=3, text="the general"),
]


def test_worklist_flags_and_decided_drop():
    wl = compute_worklist(SEGS, review_markers={})
    ids = {it.segment.id for it in wl}
    assert "b" in ids                                  # empty flagged
    wl2 = compute_worklist(SEGS, review_markers={"b": "corrected"})
    assert "b" not in {it.segment.id for it in wl2}    # decided -> dropped


def test_variant_divergence_reaches_worklist():
    # intra-graph variant divergence reaches the worklist
    wl3 = compute_worklist(SEGS, review_markers={},
                           variants={"a": {"whisper": "The artist of war"}})
    assert any("transcriber-divergence" in it.flags for it in wl3 if it.segment.id == "a")


def test_resolve_graph_db_path():
    m = {"capabilities": {"cjm-capability-graph-sqlite": {"db_path": "/tmp/g.db"}}}
    assert resolve_graph_db_path(m, "cjm-capability-graph-sqlite") == "/tmp/g.db"
    assert resolve_graph_db_path(m, "cjm-capability-graph-sqlite", override="/x") == "/x"
