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
    # purpose=None (genuine pass) stays OFF the node — absence is the genuine marker
    assert "purpose" not in sess.to_graph_node().properties
    tagged = CorrectionSession(scope=["src1"], purpose="feature-test")
    assert tagged.to_graph_node().properties["purpose"] == "feature-test"


def test_spine_segment_and_worklist_item():
    assert SpineSegment(id="n1", index=3, text="  ").is_empty
    assert not SpineSegment(id="n2", index=4, text="hi").is_empty
    seg = SpineSegment(id="n3", index=5, text="hi", text_from="t-acc",
                       text_slices=[{"transcript": "t-acc", "start": 0, "end": 2,
                                     "content_hash": "sha256:x"}])
    assert seg.text_slices[0]["transcript"] == "t-acc"
    assert WorklistItem(segment=SpineSegment(id="n2", index=4, text="hi"), flags=["x"]).index == 4


def test_correction_relations_registry():
    assert set(CorrectionRelations.all()) == {"CORRECTS", "SUPERSEDES", "DERIVED_FROM", "REVIEWED", "ASSIGNS", "GATES"}


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


def test_recommended_insert_labels_slate():
    """DEC 3d3fa2a8 + the C.1 drive: the insert-label slate is DATA (open
    vocabulary, the RECOMMENDED_MARK_CLASSES regime) — every entry must be a
    valid label (alnum-led — punctuation-led tokens are gesture-reserved)."""
    from cjm_transcript_correction_core.models import RECOMMENDED_INSERT_LABELS
    assert "inhale" in RECOMMENDED_INSERT_LABELS
    assert all(c[:1].isalnum() for c in RECOMMENDED_INSERT_LABELS)
    assert len(set(RECOMMENDED_INSERT_LABELS)) == len(RECOMMENDED_INSERT_LABELS)


def test_entity_and_speaker_vocabulary():
    """DEC 4ec6a49c + 484e2d74: Entity = stable identity handle (provisional =
    description, not identification); ASSIGNS joins the relation registry; the
    speaker mark classes seed the slate (44afb2df deviation marks)."""
    from cjm_transcript_correction_core.models import (Entity, CorrectionRelations,
                                                       RECOMMENDED_MARK_CLASSES)
    e = Entity(canonical_name="HH montage narrator", provisional=True)
    n = e.to_graph_node()
    assert n.label == "Entity" and n.id == e.id
    assert n.properties["provisional"] is True and n.properties["kind"] == "person"
    assert n.properties["canonical_name"] == "HH montage narrator"
    assert "id" not in n.properties
    assert "ASSIGNS" in CorrectionRelations.all()
    for mc in ("speaker-merge", "voiced-quote", "persona-shift", "speaker-unresolved",
               "false-start"):
        assert mc in RECOMMENDED_MARK_CLASSES


def test_dataset_manifest_shape_and_save(tmp_path):
    """DEC 16159e09: the DatasetManifest extends the chainable pattern — format
    tag + version + consumed pointers + policies as DATA — and saves via the
    same WS-token discipline as the run manifest."""
    import json
    from cjm_transcript_correction_core.models import DatasetManifest, new_dataset_id

    did = new_dataset_id()
    assert did.startswith("dataset_")

    m = DatasetManifest(
        dataset_id=did, created_at=1.0,
        config={"graph_capability": "cjm-capability-graph-sqlite",
                "include_purposes": ["genuine"]},
        graph_db_path="/x/context_graph.db",
        journals=["/x/context_graph.writes.jsonl"],
        session_purpose_policy={"include": ["genuine"], "unset_means": "genuine"},
        split_policy={"policy": "tail-reservation"},
        augmentation_policy={"policy": "none"},
        class_vocabulary={"inhale": 2},
        spines=[{"source_id": "s1", "skeleton_hash": "sha256:abc",
                 "extraction_status": "in_progress", "annotated_through": 2016.2,
                 "eligible": True, "examples": 2}],
        files={"events": "events.jsonl", "regions": "regions.jsonl"},
        counts={"examples": 2, "negative_regions": 3})
    d = m.to_dict()
    assert d["format"] == "cjm-transcript-correction-core/dataset-manifest"
    assert d["version"] == "0.1.0"
    assert d["split_policy"]["policy"] == "tail-reservation"   # policy is DATA
    assert d["spines"][0]["annotated_through"] == 2016.2       # gate @ extraction time

    out = m.save(tmp_path / "ds" / "manifest.json")
    loaded = json.loads(out.read_text())
    assert loaded == json.loads(json.dumps(d))                  # round-trips losslessly
