"""Tests for cjm_transcript_correction_core.models — correction overlay data shapes.

Projected from the models notebook's smoke-check cell at the golden-reference flip."""
from cjm_transcript_correction_core.models import (
    Correction,
    CorrectionManifest,
    CorrectionRelations,
    CorrectionSession,
    SpineSegment,
    WorklistItem,
    new_run_id,
)


def test_correction_graph_node_mapping():
    c = Correction(correction_type="grouping", status="applied", session_id="s1",
                   payload={"operation": "prune_empty", "pruned_segment_ids": ["a", "b"]})
    node = c.to_graph_node()
    assert node.label == "Correction"
    assert node.properties["correction_type"] == "grouping"
    assert node.properties["payload"]["operation"] == "prune_empty"
    assert "id" not in node.properties              # id maps to the structural field, not properties
    assert "rationale" not in node.properties       # None excluded by exclude_none


def test_correction_session_node():
    sess = CorrectionSession(scope=["src1"])
    assert sess.to_graph_node().label == "CorrectionSession"
    assert sess.to_graph_node().properties["status"] == "in_progress"


def test_spine_segment_and_worklist_item():
    assert SpineSegment(id="n1", index=3, text="  ").is_empty
    assert not SpineSegment(id="n2", index=4, text="hi").is_empty
    seg = SpineSegment(id="n3", index=5, text="hi", text_from="t-acc",
                       text_slices=[{"transcript": "t-acc", "start": 0, "end": 2,
                                     "content_hash": "sha256:x"}])
    assert seg.text_slices[0]["transcript"] == "t-acc"
    assert WorklistItem(segment=SpineSegment(id="n2", index=4, text="hi"), flags=["x"]).index == 4


def test_correction_relations_registry():
    assert set(CorrectionRelations.all()) == {"CORRECTS", "SUPERSEDES", "DERIVED_FROM", "REVIEWED"}


def test_manifest_shape_and_run_id():
    m = CorrectionManifest(run_id="r", created_at=0.0, config={}, decomp_manifest="/tmp/d.json",
                           graph_db_path="/tmp/g.db", session_id="s1")
    md = m.to_dict()
    assert md["format"] == "cjm-transcript-correction-core/run-manifest"
    assert md["version"] == "0.2.0" and md["sources"] == [] and "secondary_manifest" not in md
    assert new_run_id().startswith("correct_")


def test_manifest_save_round_trip(tmp_path):
    import json
    m = CorrectionManifest(run_id="r", created_at=0.0, config={}, decomp_manifest="/tmp/d.json",
                           graph_db_path="/tmp/g.db", session_id="s1")
    out = m.save(tmp_path / "runs" / "m.json")
    assert json.loads(out.read_text())["run_id"] == "r"
