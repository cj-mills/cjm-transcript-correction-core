"""Tests for cjm_transcript_correction_core.graph — overlay construction + effective spine.

Projected from the graph notebook's two pure-check cells at the golden-reference flip
(no runtime; graph I/O paths are exercised by the e2e harnesses in tests_manual/)."""
from cjm_transcript_correction_core.graph import (
    active_corrections,
    build_boundary_shift_correction,
    build_mark_correction,
    build_prune_amendment,
    build_prune_correction,
    build_reject_review,
    build_text_correction,
    corrections_to_edits,
    LEGACY_SKELETON,
    open_marks,
    project_effective_spine,
    reanchor_span,
    spine_where_for,
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


def test_build_mark_correction_anchor_shapes():
    # segment: one CORRECTS edge, non-mutating payload vocabulary
    node, edges = build_mark_correction("src1", {"kind": "segment", "segment_id": "a"},
                                        "suspect", session_id="s1", note="check later")
    props = node["properties"]
    assert props["correction_type"] == "mark"
    assert props["payload"]["operation"] == "mark"
    assert props["payload"]["mark_class"] == "suspect"
    assert props["rationale"] == "check later"
    assert [(e["relation_type"], e["target_id"]) for e in edges] == [("CORRECTS", "a")]
    # boundary: CORRECTS x2 (the shift gesture's coordinates; seams allowed)
    _, edges = build_mark_correction(
        "src1", {"kind": "boundary", "boundary_after": "a", "right_segment_id": "b"},
        "repeat-omission", session_id="s1")
    assert [(e["relation_type"], e["target_id"]) for e in edges] == [
        ("CORRECTS", "a"), ("CORRECTS", "b")]
    # span: offsets + verbatim snapshot ride the anchor; supersession is an edge
    node, edges = build_mark_correction(
        "src1", {"kind": "span", "segment_id": "c", "char_start": 0, "char_end": 5,
                 "text_snapshot": "world"},
        "homophone-substitution", session_id="s1", supersedes_id="m0")
    assert node["properties"]["payload"]["anchor"]["text_snapshot"] == "world"
    assert ("SUPERSEDES", "m0") in [(e["relation_type"], e["target_id"]) for e in edges]


def test_build_mark_correction_validation():
    def rejects(anchor, mark_class="suspect"):
        try:
            build_mark_correction("src1", anchor, mark_class, session_id="s1")
        except ValueError:
            return True
        return False
    assert rejects({"kind": "segment", "segment_id": "a"}, mark_class="   ")
    # punctuation-led classes are reserved for gestures (the '`-`' junk-mark drive find)
    assert rejects({"kind": "segment", "segment_id": "a"}, mark_class="-")
    assert rejects({"kind": "segment", "segment_id": "a"}, mark_class="`-`")
    assert rejects({"kind": "sentence", "segment_id": "a"})
    assert rejects({"kind": "segment"})
    assert rejects({"kind": "boundary", "boundary_after": "a"})
    # a span without its snapshot could never re-anchor — refused at build time
    assert rejects({"kind": "span", "segment_id": "a", "char_start": 0, "char_end": 2})


def test_mark_never_touches_projection():
    """The DEC 2a231843 invariant: a mark is invisible to the effective view BY
    CONSTRUCTION — corrections_to_edits has no arm for correction_type "mark"."""
    node, _ = build_mark_correction("src1", {"kind": "segment", "segment_id": "b"},
                                    "hesitation-omission", session_id="s1")
    props = dict(node["properties"])
    props["id"] = node["id"]
    assert corrections_to_edits([props]) == []
    out = project_effective_spine(SEGS, [props])
    assert [(s.id, s.text) for s in out] == [(s.id, s.text) for s in SEGS]


def test_open_marks_lifecycle():
    m1, _ = build_mark_correction("src1", {"kind": "segment", "segment_id": "a"},
                                  "suspect", session_id="s1")
    m2, _ = build_mark_correction("src1", {"kind": "boundary", "boundary_after": "a",
                                           "right_segment_id": "b"},
                                  "repeat-omission", session_id="s1")
    fix, _ = build_text_correction("src1", "a", "Hello", session_id="s1",
                                   supersedes_id=m1["id"])
    rows = []
    for n in (m1, m2, fix):
        p = dict(n["properties"])
        p["id"] = n["id"]
        rows.append(p)
    # open until something supersedes: the discharging correction closes m1
    assert [m["id"] for m in open_marks(rows, set())] == [m1["id"], m2["id"]]
    assert [m["id"] for m in open_marks(rows, {m1["id"]})] == [m2["id"]]
    # a dismissal review (reject-as-supersede) closes m2 and is not itself a mark
    rej, _ = build_reject_review("src1", m2["id"], session_id="s1")
    p = dict(rej["properties"])
    p["id"] = rej["id"]
    assert open_marks(rows + [p], {m1["id"], m2["id"]}) == []


def test_reanchor_span():
    a = {"char_start": 6, "char_end": 11, "text_snapshot": "world"}
    assert reanchor_span(a, "hello world") == (6, 11)         # exact offsets verified
    assert reanchor_span(a, "well, hello world") == (12, 17)  # edited text -> snapshot re-found
    assert reanchor_span(a, "goodbye moon") is None           # gone -> degrade to segment level
    # multiple occurrences: the one nearest the recorded start wins
    assert reanchor_span({"char_start": 0, "char_end": 2, "text_snapshot": "ab"},
                         "ab ab") == (0, 2)
    assert reanchor_span({"char_start": 3, "char_end": 5, "text_snapshot": "ab"},
                         "ab ab") == (3, 5)
    assert reanchor_span({"text_snapshot": ""}, "x") is None


def test_spine_where_for_selector_semantics():
    legacy = {"skeleton_hash": None, "split_policy": None, "segments": 950}
    split = {"skeleton_hash": "sha256:abc123def", "split_policy": "sentence-split/v1",
             "segments": 1100}
    # Sole spine (either kind): no filter needed — reads stay unscoped.
    assert spine_where_for([legacy]) == []
    assert spine_where_for([split]) == []
    # Coexisting spines + auto: refuse loudly (unfiltered reads would MIX them).
    try:
        spine_where_for([legacy, split])
        assert False, "auto over coexisting spines must refuse"
    except ValueError as e:
        assert "--skeleton" in str(e)
    # Explicit selectors: legacy -> prop-absent filter; hash/hex-tail prefix -> eq.
    [p] = spine_where_for([legacy, split], LEGACY_SKELETON)
    assert (p.prop, p.op) == ("skeleton_hash", "is_null")
    for sel in ("sha256:abc123def", "sha256:abc", "abc123", "ABC"):
        [p] = spine_where_for([legacy, split], sel)
        assert (p.prop, p.op, p.value) == ("skeleton_hash", "eq", "sha256:abc123def")
    # A selector matching nothing (or a missing legacy spine) refuses with the roster.
    for bad in ("nope", LEGACY_SKELETON):
        try:
            spine_where_for([split], bad)
            assert False, f"selector {bad!r} must refuse"
        except ValueError:
            pass


def test_projection_ignores_foreign_spine_corrections():
    # Corrections load SOURCE-wide but anchor by segment id; parallel spines
    # share no ids (DEC f1024568), so another spine's edits must be inert on
    # this one — not a layer SpineEditError (the 2026-07-22 split-spine crash).
    spine = [SpineSegment(id="n1", index=0, text="hello world"),
             SpineSegment(id="n2", index=1, text="foo")]
    foreign = [
        {"id": "c1", "correction_type": "grouping", "status": "applied",
         "created_at": 1.0,
         "payload": {"operation": "shift_boundary", "boundary_after": "old1",
                     "right_segment_id": "old2", "text": "word",
                     "direction": "push"}},
        {"id": "c2", "correction_type": "grouping", "status": "applied",
         "created_at": 2.0,
         "payload": {"operation": "prune_empty", "pruned_segment_ids": ["old3"]}},
        {"id": "c3", "correction_type": "text_content", "status": "applied",
         "created_at": 3.0,
         "payload": {"operation": "replace_text", "segment_id": "old1",
                     "new_text": "nope"}},
    ]
    out = project_effective_spine(spine, foreign)
    assert [(s.id, s.text) for s in out] == [("n1", "hello world"), ("n2", "foo")]
    # ...while THIS spine's own corrections still apply.
    own = [{"id": "c4", "correction_type": "text_content", "status": "applied",
            "created_at": 4.0,
            "payload": {"operation": "replace_text", "segment_id": "n2",
                        "new_text": "bar"}}]
    assert [s.text for s in project_effective_spine(spine, foreign + own)] \
        == ["hello world", "bar"]


def test_time_nudge_build_and_projection():
    """3f9948d6: a timing correction nudges segment boundary TIMES through the
    effective projection — welded point cuts move both edges in ONE atomic
    correction, repeated nudges chain latest-wins per edge, foreign-spine
    edits drop, and layer-0 times stay untouched (non-destructive)."""
    from cjm_transcript_correction_core.graph import (apply_time_nudges,
                                                      build_time_nudge_correction,
                                                      project_effective_spine)
    node, edges = build_time_nudge_correction(
        "src-1",
        [{"segment_id": "a", "edge": "end", "old_time": 5.0, "new_time": 5.1},
         {"segment_id": "b", "edge": "start", "old_time": 5.0, "new_time": 5.1}],
        "sess-1", boundary_words={"left": "history", "right": "The"}, step_s=0.1)
    assert node["label"] == "Correction"
    assert node["properties"]["correction_type"] == "timing"
    assert node["properties"]["payload"]["boundary_words"] == {"left": "history", "right": "The"}
    assert {e["target_id"] for e in edges} == {"a", "b"}

    segs = [SpineSegment(id="a", index=0, text="one", start_time=0.0, end_time=5.0),
            SpineSegment(id="b", index=1, text="two", start_time=5.0, end_time=9.0)]

    def nudge(created, edits):
        return {"id": f"c{created}", "correction_type": "timing", "status": "applied",
                "created_at": created,
                "payload": {"operation": "time_nudge", "source_id": "src-1",
                            "edits": edits}}

    weld = nudge(1.0, [{"segment_id": "a", "edge": "end", "old_time": 5.0, "new_time": 5.1},
                       {"segment_id": "b", "edge": "start", "old_time": 5.0, "new_time": 5.1}])
    again = nudge(2.0, [{"segment_id": "a", "edge": "end", "old_time": 5.1, "new_time": 5.2},
                        {"segment_id": "b", "edge": "start", "old_time": 5.1, "new_time": 5.2}])
    foreign = nudge(3.0, [{"segment_id": "zz-other-spine", "edge": "end",
                           "old_time": 1.0, "new_time": 2.0}])
    out = apply_time_nudges(segs, [again, weld, foreign])   # order-independent input
    assert (out[0].end_time, out[1].start_time) == (5.2, 5.2)   # latest-wins chain
    assert out[0].start_time == 0.0 and out[1].end_time == 9.0  # untouched edges keep layer-0
    assert segs[0].end_time == 5.0                              # non-destructive
    # composes through the effective projection (text edits + nudges together)
    projected = project_effective_spine(segs, [weld])
    assert projected[0].end_time == 5.1 and projected[1].start_time == 5.1
    assert projected[0].text == "one"
