"""Tests for cjm_transcript_correction_core.graph — overlay construction + effective spine.

Projected from the graph notebook's two pure-check cells at the golden-reference flip
(no runtime; graph I/O paths are exercised by the e2e harnesses in tests_manual/)."""
from cjm_transcript_correction_core.graph import (
    active_corrections,
    build_boundary_shift_correction,
    build_prune_amendment,
    build_prune_correction,
    build_reject_review,
    build_text_correction,
    corrections_to_edits,
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


def test_build_boundary_shift_correction():
    node, edges = build_boundary_shift_correction(
        "src1", "a", "b", "wor", "push", session_id="s1")
    p = node["properties"]["payload"]
    assert node["properties"]["correction_type"] == "grouping"
    assert p["operation"] == "shift_boundary"
    assert p["boundary_after"] == "a" and p["right_segment_id"] == "b"
    assert p["text"] == "wor" and p["direction"] == "push"
    assert [(e["relation_type"], e["target_id"]) for e in edges] == [
        ("CORRECTS", "a"), ("CORRECTS", "b")]
    try:
        build_boundary_shift_correction("src1", "a", "b", "x", "sideways", session_id="s1")
        assert False, "invalid direction must raise"
    except ValueError:
        pass


def test_boundary_shift_projection_push_and_pull():
    # push: FA-misassigned whole words move right, single-space junction (DEC on 58b2e0a0)
    segs = [SpineSegment(id="a", index=0, text="Mr. Gorbachev, tear"),
            SpineSegment(id="b", index=1, text="down this wall.")]
    node, _ = build_boundary_shift_correction("src1", "a", "b", "tear", "push", session_id="s1")
    props = dict(node["properties"])
    props["id"] = node["id"]
    eff = project_effective_spine(segs, [props])
    assert [s.text for s in eff] == ["Mr. Gorbachev,", "tear down this wall."]

    # pull: the mirror — the right unit's head belongs to the left
    node2, _ = build_boundary_shift_correction("src1", "a", "b", "down", "pull", session_id="s1")
    props2 = dict(node2["properties"])
    props2["id"] = node2["id"]
    eff2 = project_effective_spine(segs, [props2])
    assert [s.text for s in eff2] == ["Mr. Gorbachev, tear down", "this wall."]

    # empty-neighbor (the falsified-D14 class): push into a starved chunk, both sides clean
    starved = [SpineSegment(id="a", index=0, text="largest naval battle in history"),
               SpineSegment(id="b", index=1, text="")]
    node3, _ = build_boundary_shift_correction("src1", "a", "b", "in history", "push", session_id="s1")
    props3 = dict(node3["properties"])
    props3["id"] = node3["id"]
    eff3 = project_effective_spine(starved, [props3])
    assert [s.text for s in eff3] == ["largest naval battle", "in history"]


def test_reject_review_and_proposed_exclusion():
    # a proposed correction never enters the effective view (awaiting a verdict)
    prop = {"id": "p1", "correction_type": "text_content", "status": "proposed",
            "payload": {"operation": "replace_text", "segment_id": "a", "new_text": "NOPE"}}
    assert project_effective_spine(SEGS, [prop])[0].text == "hello"

    # reject-as-supersede: the review node SUPERSEDES the proposal
    node, edges = build_reject_review("src1", "p1", session_id="s1", rationale="wrong word")
    assert node["properties"]["correction_type"] == "review"
    assert node["properties"]["payload"]["operation"] == "reject"
    assert node["properties"]["payload"]["rejected_id"] == "p1"
    assert len(edges) == 1
    assert edges[0]["relation_type"] == "SUPERSEDES" and edges[0]["target_id"] == "p1"

    # a review node maps to NO spine edit
    props = dict(node["properties"])
    props["id"] = node["id"]
    assert corrections_to_edits([props]) == []

    # the active filter drops the rejected proposal (as _superseded_ids would report)
    assert [c["id"] for c in active_corrections([prop, props], {"p1"})] == [node["id"]]


def test_prune_amendment_rescues_boundary_shift_target():
    # the falsified-D14 rescue: prune covers an empty chunk, a boundary shift
    # gives it text, the amendment must un-prune it or projection drops the text
    segs = [SpineSegment(id="a", index=0, text="largest naval battle in history"),
            SpineSegment(id="b", index=1, text=""),
            SpineSegment(id="c", index=2, text="the end")]
    prune_node, _ = build_prune_correction("src1", [segs[1]], session_id="s1")
    prune = dict(prune_node["properties"])
    prune["id"] = prune_node["id"]
    prune["created_at"] = 1.0

    shift_node, _ = build_boundary_shift_correction("src1", "a", "b", "in history", "push",
                                                    session_id="s1")
    shift = dict(shift_node["properties"])
    shift["id"] = shift_node["id"]
    shift["created_at"] = 2.0

    # without the amendment the pruned position swallows the moved text
    eff = project_effective_spine(segs, [prune, shift])
    assert [s.id for s in eff] == ["a", "c"]

    amend_node, amend_edges = build_prune_amendment(prune, ["b"], session_id="s1")
    amend = dict(amend_node["properties"])
    amend["id"] = amend_node["id"]
    amend["created_at"] = 3.0
    assert amend["payload"]["pruned_segment_ids"] == []
    assert amend["payload"]["pruned_count"] == 0
    assert amend["payload"]["source_id"] == "src1"
    sup = [e for e in amend_edges if e["relation_type"] == "SUPERSEDES"]
    assert len(sup) == 1 and sup[0]["target_id"] == prune["id"]

    # the amendment SUPERSEDES the prune -> active set = shift + amendment
    active = active_corrections([prune, shift, amend], {prune["id"]})
    eff2 = project_effective_spine(segs, active)
    assert [s.id for s in eff2] == ["a", "b", "c"]
    assert eff2[0].text == "largest naval battle" and eff2[1].text == "in history"
