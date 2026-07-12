"""Tests for cjm_transcript_correction_core.graph — overlay construction + effective spine.

Projected from the graph notebook's two pure-check cells at the golden-reference flip
(no runtime; graph I/O paths are exercised by the e2e harnesses in tests_manual/)."""
from cjm_transcript_correction_core.graph import (
    active_corrections,
    build_prune_correction,
    build_text_correction,
    project_effective_spine,
)
from cjm_transcript_correction_core.models import SpineSegment

SEGS = [
    SpineSegment(id="a", index=0, text="hello"),
    SpineSegment(id="b", index=1, text=""),
    SpineSegment(id="c", index=2, text="world"),
]


def test_build_prune_correction():
    empties = [s for s in SEGS if s.is_empty]
    node, edges = build_prune_correction("src1", empties, session_id="s1")
    assert node["label"] == "Correction"
    assert node["properties"]["payload"]["source_id"] == "src1"
    assert node["properties"]["payload"]["pruned_segment_ids"] == ["b"]
    assert len(edges) == 1 and edges[0]["relation_type"] == "DERIVED_FROM"
    assert edges[0]["target_id"] == "b" and edges[0]["source_id"] == node["id"]
    # structural overlay edges are deterministic (layer make_edge): ids follow the (new) node
    node2, edges2 = build_prune_correction("src1", empties, session_id="s1")
    assert edges2[0]["id"] != edges[0]["id"] or node2["id"] == node["id"]


def test_project_effective_spine_prune_replace_supersede():
    empties = [s for s in SEGS if s.is_empty]
    node, edges = build_prune_correction("src1", empties, session_id="s1")
    props = dict(node["properties"])
    props["id"] = node["id"]
    eff = project_effective_spine(SEGS, [props])
    assert [s.id for s in eff] == ["a", "c"]               # prune drops the empty segment

    repl = {"id": "corr1", "correction_type": "text_content", "status": "applied",
            "payload": {"operation": "replace_text", "segment_id": "a", "new_text": "HELLO"}}
    assert project_effective_spine(SEGS, [repl])[0].text == "HELLO"

    sup = dict(props)
    sup["status"] = "superseded"
    assert [s.id for s in project_effective_spine(SEGS, [sup])] == ["a", "b", "c"]  # superseded ignored


def test_latest_wins_ordering_from_layer():
    # latest-wins ordering now comes from the LAYER (created_at)
    r1 = {"id": "c1", "correction_type": "text_content", "created_at": 1.0,
          "payload": {"operation": "replace_text", "segment_id": "a", "new_text": "first"}}
    r2 = {"id": "c2", "correction_type": "text_content", "created_at": 2.0,
          "payload": {"operation": "replace_text", "segment_id": "a", "new_text": "second"}}
    assert project_effective_spine(SEGS, [r1, r2])[0].text == "second"


def test_build_text_correction_and_active_filter():
    tn, te = build_text_correction("src1", "segX", "fixed text", session_id="s1",
                                   supersedes_id="prevCorr")
    assert tn["properties"]["correction_type"] == "text_content"
    assert tn["properties"]["payload"]["segment_id"] == "segX"
    assert tn["properties"]["payload"]["new_text"] == "fixed text"
    assert tn["properties"]["payload"]["source_id"] == "src1"
    assert {e["relation_type"] for e in te} == {"CORRECTS", "SUPERSEDES"}
    corr_edge = next(e for e in te if e["relation_type"] == "CORRECTS")
    assert corr_edge["target_id"] == "segX" and corr_edge["source_id"] == tn["id"]
    assert next(e for e in te if e["relation_type"] == "SUPERSEDES")["target_id"] == "prevCorr"
    assert [c["id"] for c in active_corrections(
        [{"id": "a"}, {"id": "b"}, {"id": "c"}], {"b"})] == ["a", "c"]
