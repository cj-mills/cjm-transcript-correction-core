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


def test_chunk_insert_build_and_projection():
    """DEC 3d3fa2a8: an insertion Correction synthesizes a chunk the skeleton
    never cut — overlay-projected (synthetic id = the Correction's node id),
    placed after its left flank, text latest-wins across the payload and
    replace_text corrections targeting the synthetic id, nudges compose after
    insertion (zero-width inserts grow by their edges), and supersession
    removes it through the ordinary active filter."""
    from cjm_transcript_correction_core.graph import (apply_chunk_inserts,
                                                      build_chunk_insert_correction)
    node, edges = build_chunk_insert_correction(
        "src-1", "a", 4.5, 6.0, "sess-1", before_segment_id="b", label="inhale")
    assert node["label"] == "Correction"
    p = node["properties"]["payload"]
    assert node["properties"]["correction_type"] == "insertion"
    assert p["operation"] == "chunk_insert" and p["label"] == "inhale"
    assert p["after_segment_id"] == "a" and p["before_segment_id"] == "b"
    assert p["text"] == ""                       # born empty
    assert [(e["relation_type"], e["target_id"]) for e in edges] == [
        ("CORRECTS", "a"), ("CORRECTS", "b")]
    try:
        build_chunk_insert_correction("src-1", "a", 6.0, 4.5, "sess-1")
        assert False, "negative span must raise"
    except ValueError:
        pass

    segs = [SpineSegment(id="a", index=0, text="one", start_time=0.0, end_time=4.5),
            SpineSegment(id="b", index=1, text="two", start_time=6.0, end_time=9.0)]
    ins = dict(node["properties"])
    ins["id"] = node["id"]
    eff = project_effective_spine(segs, [ins])
    assert [s.id for s in eff] == ["a", node["id"], "b"]
    assert eff[1].text == "" and (eff[1].start_time, eff[1].end_time) == (4.5, 6.0)
    assert eff[1].index == 0                     # layer-0 coordinate of the left flank

    # missed speech arrives by e-edit: replace_text targets the SYNTHETIC id
    # (invisible to the layer projection — applied at the insert stage)
    txt = {"id": "t1", "correction_type": "text_content", "created_at": 2.0,
           "payload": {"operation": "replace_text", "segment_id": node["id"],
                       "new_text": "dispatch audio"}}
    assert project_effective_spine(segs, [ins, txt])[1].text == "dispatch audio"

    # nudges compose AFTER insertion: a zero-width insert grows by its edges
    zw, _ = build_chunk_insert_correction("src-1", "a", 4.5, 4.5, "sess-1")
    zwp = dict(zw["properties"])
    zwp["id"] = zw["id"]
    grow = {"id": "n1", "correction_type": "timing", "created_at": 3.0,
            "payload": {"operation": "time_nudge", "source_id": "src-1",
                        "edits": [{"segment_id": zw["id"], "edge": "end",
                                   "old_time": 4.5, "new_time": 4.62}]}}
    out = project_effective_spine(segs, [zwp, grow])
    assert (out[1].start_time, out[1].end_time) == (4.5, 4.62)

    # a proposed insertion never enters the effective view (awaiting a verdict)
    prop = dict(zwp)
    prop["status"] = "proposed"
    assert [s.id for s in apply_chunk_inserts(segs, [prop])] == ["a", "b"]

    # foreign-spine inserts drop, not error (the f1024568 spine-scoping rule)
    foreign = {"id": "f1", "correction_type": "insertion",
               "payload": {"operation": "chunk_insert", "source_id": "src-1",
                           "after_segment_id": "zz", "before_segment_id": "zz2",
                           "start_time": 1.0, "end_time": 2.0, "text": ""}}
    assert [s.id for s in apply_chunk_inserts(segs, [foreign])] == ["a", "b"]

    # a vanished left flank (pruned away downstream) falls back to the right flank
    fallback = {"id": "f2", "correction_type": "insertion",
                "payload": {"operation": "chunk_insert", "source_id": "src-1",
                            "after_segment_id": "gone", "before_segment_id": "b",
                            "start_time": 5.0, "end_time": 5.5, "text": ""}}
    assert [s.id for s in apply_chunk_inserts(segs, [fallback])] == ["a", "f2", "b"]

    # several inserts STACK in one gap (inhale · um · inhale): shared layer-0
    # anchor, ordered by (start_time, created_at) — the C.1 drive find
    i1 = {"id": "g1", "correction_type": "insertion", "created_at": 1.0,
          "payload": {"operation": "chunk_insert", "source_id": "src-1",
                      "after_segment_id": "a", "before_segment_id": "b",
                      "start_time": 4.5, "end_time": 4.8, "text": "", "label": "inhale"}}
    i2 = {"id": "g2", "correction_type": "insertion", "created_at": 2.0,
          "payload": {"operation": "chunk_insert", "source_id": "src-1",
                      "after_segment_id": "a", "before_segment_id": "b",
                      "start_time": 4.8, "end_time": 4.8, "text": "", "label": "um"}}
    assert [s.id for s in apply_chunk_inserts(segs, [i2, i1])] == ["a", "g1", "g2", "b"]

    # removal = reject-as-supersede: the ordinary active filter excludes it
    assert active_corrections([zwp], {zw["id"]}) == []


def test_speaker_assign_build_and_active_projection():
    """DEC d6df3a8e: the assignment op envelope — verdict vocabulary enforced,
    entity binding via ASSIGNS + canonical_form, reassignment supersedes, and
    the active projection is latest-wins per segment (apply_time_nudges regime)."""
    import pytest
    from cjm_transcript_correction_core.graph import (active_speaker_assignments,
                                                      build_speaker_assign_correction)
    node, edges = build_speaker_assign_correction(
        "src-1", ["seg-a", "seg-b"], "ent-1", "sess-1", verdict="accept",
        proposal={"turn_start": 1.0, "turn_end": 9.5, "cluster": "SPK_00",
                  "confidence": 0.91})
    p = node["properties"]
    assert p["correction_type"] == "speaker" and p["canonical_form"] == "ent-1"
    assert p["payload"]["verdict"] == "accept"
    assert p["payload"]["proposal"]["cluster"] == "SPK_00"
    rels = [(e["relation_type"], e["target_id"]) for e in edges]
    assert ("CORRECTS", "seg-a") in rels and ("CORRECTS", "seg-b") in rels
    assert ("ASSIGNS", "ent-1") in rels
    # reassignment supersedes; unknown verdicts and empty spans refuse
    node2, edges2 = build_speaker_assign_correction(
        "src-1", ["seg-a"], "ent-2", "sess-1", verdict="name",
        supersedes_id=node["id"])
    assert ("SUPERSEDES", node["id"]) in [(e["relation_type"], e["target_id"])
                                          for e in edges2]
    assert "proposal" not in node2["properties"]["payload"]  # unassisted walk
    with pytest.raises(ValueError):
        build_speaker_assign_correction("src-1", ["seg-a"], "ent-1", "sess-1",
                                        verdict="narrator")
    with pytest.raises(ValueError):
        build_speaker_assign_correction("src-1", [], "ent-1", "sess-1")
    with pytest.raises(ValueError):
        build_speaker_assign_correction("src-1", ["seg-a"], "", "sess-1")
    # active projection: latest-wins per segment, superseded excluded
    def corr(cid, created, seg_ids, ent, verdict="name"):
        return {"id": cid, "correction_type": "speaker", "created_at": created,
                "payload": {"operation": "speaker_assign", "source_id": "src-1",
                            "segment_ids": seg_ids, "entity_id": ent,
                            "verdict": verdict}}
    a = corr("c1", 1.0, ["seg-a", "seg-b"], "ent-1")
    b = corr("c2", 2.0, ["seg-b"], "ent-2", verdict="cluster-merge")
    dead = corr("c0", 3.0, ["seg-a"], "ent-9")
    act = active_speaker_assignments([b, a, dead], superseded_ids={"c0"})
    assert act["seg-a"]["entity_id"] == "ent-1"
    assert act["seg-b"]["entity_id"] == "ent-2"
    assert act["seg-b"]["verdict"] == "cluster-merge"
    assert act["seg-b"]["correction_id"] == "c2"


def test_aggregate_session_purposes():
    """d915d545(a): per-source purpose mix — absent purpose = genuine, multi-
    source scopes count once per source, purposes stay an open vocabulary."""
    from cjm_transcript_correction_core.graph import aggregate_session_purposes
    def sess(purpose, scope):
        props = {"scope": scope}
        if purpose is not None:
            props["purpose"] = purpose
        return {"id": "s", "properties": props}
    mix = aggregate_session_purposes([
        sess("feature-test", ["src-1"]),
        sess("feature-test", ["src-1", "src-2"]),
        sess(None, ["src-1"]),
        sess("spike", ["src-3"]),
        sess("feature-test", []),          # empty scope contributes nothing
    ])
    assert mix["src-1"] == {"feature-test": 2, "genuine": 1}
    assert mix["src-2"] == {"feature-test": 1}
    assert mix["src-3"] == {"spike": 1}
    assert set(mix) == {"src-1", "src-2", "src-3"}


def test_chunk_split_composes_from_existing_verbs():
    """Work item 99c1d2ba: a chunk SPLIT is three EXISTING verbs (right-half
    chunk_insert + left-half replace_text + end-truncating time_nudge) that the
    projection's text -> inserts -> nudges composition lands as [left | right]
    with a WELDED point cut — no new projection arm. The rationale carries the
    chunk-split group marker, and splitting the SYNTHETIC right half again
    works uniformly (the insert-stage text/nudge lanes)."""
    from cjm_transcript_correction_core.graph import build_chunk_split_corrections

    segs = [SpineSegment(id="a", index=0, text="alpha beta gamma",
                         start_time=0.0, end_time=6.0),
            SpineSegment(id="b", index=1, text="tail", start_time=6.0, end_time=9.0)]
    nodes, edges, ids = build_chunk_split_corrections(
        "src-1", "a", 2.0, "alpha", "beta gamma", 6.0, "sess-1", "a",
        before_segment_id="b", old_text="alpha beta gamma",
        boundary_words={"left": "alpha", "right": "beta"})
    assert [n["properties"]["correction_type"] for n in nodes] == [
        "insertion", "text_content", "timing"]
    ins, txt, ndg = nodes
    assert ids == {"insert_id": ins["id"], "text_id": txt["id"], "nudge_id": ndg["id"]}
    # the group marker: three ops, one human decision
    group = f"chunk-split:{ins['id']}"
    assert ins["properties"]["rationale"] == "chunk-split"
    assert txt["properties"]["rationale"] == group
    assert ndg["properties"]["rationale"] == group
    assert ins["properties"]["payload"]["text"] == "beta gamma"
    assert ndg["properties"]["payload"]["edits"] == [
        {"segment_id": "a", "edge": "end", "old_time": 6.0, "new_time": 2.0}]

    corrs = []
    for n in nodes:
        c = dict(n["properties"])
        c["id"] = n["id"]
        corrs.append(c)
    eff = project_effective_spine(segs, corrs)
    assert [s.id for s in eff] == ["a", ids["insert_id"], "b"]
    assert (eff[0].text, eff[0].start_time, eff[0].end_time) == ("alpha", 0.0, 2.0)
    assert (eff[1].text, eff[1].start_time, eff[1].end_time) == ("beta gamma", 2.0, 6.0)
    # the new seam is WELDED: the existing nudge machinery owns its precision

    # splitting the SYNTHETIC right half again: replace_text/nudge ride the
    # insert-stage lanes, the new piece stacks under the shared layer-0 anchor
    n2, e2, ids2 = build_chunk_split_corrections(
        "src-1", ids["insert_id"], 4.0, "beta", "gamma", 6.0, "sess-1", "a",
        before_segment_id="b")
    for n in n2:
        c = dict(n["properties"])
        c["id"] = n["id"]
        corrs.append(c)
    eff2 = project_effective_spine(segs, corrs)
    assert [s.id for s in eff2] == ["a", ids["insert_id"], ids2["insert_id"], "b"]
    assert [(s.text, s.start_time, s.end_time) for s in eff2] == [
        ("alpha", 0.0, 2.0), ("beta", 2.0, 4.0), ("gamma", 4.0, 6.0),
        ("tail", 6.0, 9.0)]

    # guards: empty halves and a cut at/after the end refuse loudly
    for bad in ((" ", "gamma"), ("beta", "")):
        try:
            build_chunk_split_corrections("src-1", "a", 2.0, bad[0].strip(), bad[1],
                                          6.0, "sess-1", "a")
            assert False, "empty half must raise"
        except ValueError:
            pass
    try:
        build_chunk_split_corrections("src-1", "a", 6.0, "alpha", "beta", 6.0,
                                      "sess-1", "a")
        assert False, "split at the end must raise"
    except ValueError:
        pass


def test_insert_rank_and_split_removal():
    """FINDING 131ba57a: (a) same-anchor same-start siblings order by
    (start_time, RANK, created_at) — a later insert lands BEFORE a split's
    right half when the walked cursor said so; (b) x on a split right half
    UNSPLITS: one review node supersedes the whole group and the projection
    returns to the pre-split segment exactly."""
    from cjm_transcript_correction_core.graph import (build_chunk_insert_correction,
                                                      build_chunk_split_corrections,
                                                      find_chunk_split_group)

    segs = [SpineSegment(id="a", index=0, text="alpha beta gamma",
                         start_time=0.0, end_time=6.0),
            SpineSegment(id="b", index=1, text="tail", start_time=6.0, end_time=9.0)]
    nodes, _, ids = build_chunk_split_corrections(
        "src-1", "a", 2.0, "alpha", "beta gamma", 6.0, "sess-1", "a",
        before_segment_id="b")
    corrs = []
    for n in nodes:
        c = dict(n["properties"])
        c["id"] = n["id"]
        corrs.append(c)
    for c in corrs:
        c["created_at"] = 1.0

    # the user scenario: from the LEFT half, insert the inhale AT the weld —
    # created LATER but rank -1 places it BETWEEN the halves
    inh, _ = build_chunk_insert_correction("src-1", "a", 2.0, 2.0, "sess-1",
                                           before_segment_id="b", label="inhale",
                                           rank=-1.0)
    ic = dict(inh["properties"])
    ic["id"] = inh["id"]
    ic["created_at"] = 2.0
    assert ic["payload"]["rank"] == -1.0
    eff = project_effective_spine(segs, corrs + [ic])
    assert [s.id for s in eff] == ["a", inh["id"], ids["insert_id"], "b"]
    # without the rank, creation order buries it below the right half
    ic0 = dict(ic)
    ic0["payload"] = {k: v for k, v in ic["payload"].items() if k != "rank"}
    eff0 = project_effective_spine(segs, corrs + [ic0])
    assert [s.id for s in eff0] == ["a", ids["insert_id"], inh["id"], "b"]

    # unsplit: the group resolves off the rationale marker, and superseding
    # ALL members restores the pre-split segment exactly
    group = find_chunk_split_group(corrs, ids["insert_id"])
    assert sorted(group) == sorted([ids["text_id"], ids["nudge_id"]])
    assert find_chunk_split_group(corrs, "not-a-split") == []
    active = active_corrections(corrs, {ids["insert_id"], *group})
    assert active == []
    eff2 = project_effective_spine(segs, active)
    assert [(s.id, s.text, s.start_time, s.end_time) for s in eff2] == [
        ("a", "alpha beta gamma", 0.0, 6.0), ("b", "tail", 6.0, 9.0)]


def test_correction_stats_accounting():
    """The manual-tally retirement (drive ask 2026-07-27): active labeled
    spans / splits / open marks fold into counts — split right halves are
    boundary events (never labeled spans), superseded ops and discharged
    marks drop, unlabeled inserts stay visible."""
    from cjm_transcript_correction_core.graph import correction_stats

    def corr(cid, ctype, payload, rationale=None, created=1.0):
        return {"id": cid, "correction_type": ctype, "payload": payload,
                "rationale": rationale, "created_at": created}

    corrections = [
        corr("i1", "insertion", {"operation": "chunk_insert", "label": "inhale"}),
        corr("i2", "insertion", {"operation": "chunk_insert", "label": "inhale"}),
        corr("i3", "insertion", {"operation": "chunk_insert"}),          # unlabeled
        corr("i4", "insertion", {"operation": "chunk_insert", "label": "um"}),
        corr("sp", "insertion", {"operation": "chunk_insert", "text": "tail"},
             rationale="chunk-split"),
        corr("tx", "text_content", {"operation": "replace_text", "segment_id": "a"}),
        corr("nd", "timing", {"operation": "time_nudge", "edits": []}),
        corr("m1", "mark", {"operation": "mark", "mark_class": "inhale",
                            "anchor": {"kind": "segment", "segment_id": "a"}}),
        corr("m2", "mark", {"operation": "mark", "mark_class": "overlapping-speech",
                            "anchor": {"kind": "segment", "segment_id": "a"}}),
        corr("m3", "mark", {"operation": "mark", "mark_class": "inhale",
                            "anchor": {"kind": "segment", "segment_id": "b"}}),
    ]
    st = correction_stats(corrections, {"i4", "m3"})   # um removed, one mark discharged
    assert st["insert_labels"] == {"inhale": 2, "(unlabeled)": 1}
    assert st["splits"] == 1
    assert st["mark_classes"] == {"inhale": 1, "overlapping-speech": 1}
    assert st["open_marks"] == 2
    assert st["ops"]["insertion"] == 4 and st["ops"]["text_content"] == 1
    assert st["active"] == 8

    # the caller's genuine-only cut: filtered corrections, GLOBAL superseded set
    genuine = [c for c in corrections if c["id"] not in ("i2", "m2")]
    st2 = correction_stats(genuine, {"i4", "m3"})
    assert st2["insert_labels"] == {"inhale": 1, "(unlabeled)": 1}
    assert st2["mark_classes"] == {"inhale": 1}


def test_extraction_gate_build_and_latest_wins():
    """DEC 8e05b87b: gate assertions are append-only spine state — build shape
    (node + GATES edge), status validation, and the latest-wins per-spine fold."""
    from cjm_transcript_correction_core.graph import (build_extraction_gate_assertion,
                                                      latest_extraction_gates)
    import pytest

    node, edges = build_extraction_gate_assertion(
        "src1", "sha256:abc", "in_progress", 2016.2, session_id="sess1")
    assert node["label"] == "ExtractionGate"
    p = node["properties"]
    assert p["source_id"] == "src1" and p["skeleton_hash"] == "sha256:abc"
    assert p["extraction_status"] == "in_progress"
    assert p["annotated_through"] == 2016.2
    assert len(edges) == 1 and edges[0]["relation_type"] == "GATES"
    assert edges[0]["target_id"] == "src1" and edges[0]["source_id"] == node["id"]

    # the LEGACY spine (None hash) must round-trip explicitly, never be ambiguous
    lnode, _ = build_extraction_gate_assertion("src1", None, "excluded", None)
    assert lnode["properties"]["skeleton_hash"] is None
    assert "annotated_through" not in lnode["properties"]

    with pytest.raises(ValueError):
        build_extraction_gate_assertion("src1", None, "done", 1.0)  # not a status
    with pytest.raises(ValueError):
        build_extraction_gate_assertion("src1", None, "signed_off", -3.0)

    # latest-wins per (skeleton_hash): a rescind is just a newer assertion
    a1 = {"id": "g1", "skeleton_hash": "sha256:abc", "extraction_status": "in_progress",
          "annotated_through": 100.0, "created_at": 1.0}
    a2 = {"id": "g2", "skeleton_hash": "sha256:abc", "extraction_status": "signed_off",
          "annotated_through": 2500.0, "created_at": 2.0}
    a3 = {"id": "g3", "skeleton_hash": None, "extraction_status": "excluded",
          "created_at": 1.5}
    live = latest_extraction_gates([a2, a3, a1])   # order-independent
    assert live["sha256:abc"]["id"] == "g2"        # newest wins
    assert live[None]["extraction_status"] == "excluded"
    assert live.get("sha256:other") is None        # absent spine = caller's default


def test_skeleton_hash_for_selector_semantics():
    """The gate's spine-identity resolver mirrors spine_where_for: auto on a sole
    spine, "legacy" = None, prefix match on hash, loud refusal when ambiguous."""
    from cjm_transcript_correction_core.graph import skeleton_hash_for
    import pytest

    sole = [{"skeleton_hash": None, "split_policy": None, "segments": 10}]
    assert skeleton_hash_for(sole, None) is None
    assert skeleton_hash_for([], None) is None      # pre-decomposition: sole spine to come

    h = "sha256:abcdef012345"
    both = [{"skeleton_hash": None, "split_policy": None, "segments": 10},
            {"skeleton_hash": h, "split_policy": "sentence-v1", "segments": 12}]
    with pytest.raises(ValueError):
        skeleton_hash_for(both, None)                # coexisting spines refuse auto
    assert skeleton_hash_for(both, "legacy") is None
    assert skeleton_hash_for(both, "abcdef") == h    # hex-tail prefix
    assert skeleton_hash_for([both[1]], None) == h   # sole split spine auto-resolves


def test_labeled_insert_spans_fold():
    """Leg-2 positive-span fold (DEC 16159e09): final times from the effective
    projection (nudges grow zero-width inserts), text latest-wins, split right
    halves and superseded inserts drop, unlabeled inserts ride with label None,
    and provenance is the full op chain on the synthetic id."""
    from cjm_transcript_correction_core.graph import labeled_insert_spans
    from cjm_transcript_correction_core.models import SpineSegment

    segments = [
        SpineSegment(id="a", index=0, text="hello world", start_time=0.0, end_time=10.0),
        SpineSegment(id="b", index=1, text="", start_time=10.0, end_time=20.0),
        SpineSegment(id="c", index=2, text="more text", start_time=20.0, end_time=30.0),
    ]

    def corr(cid, ctype, payload, session="s1", rationale=None, created=1.0):
        return {"id": cid, "correction_type": ctype, "payload": payload,
                "session_id": session, "rationale": rationale, "created_at": created}

    corrections = [
        # zero-width inhale, grown by a later nudge (the isolation pattern)
        corr("i1", "insertion", {"operation": "chunk_insert", "after_segment_id": "a",
                                 "start_time": 10.0, "end_time": 10.0, "text": "",
                                 "label": "inhale"}, session="g", created=1.0),
        corr("n1", "timing", {"operation": "time_nudge",
                              "edits": [{"segment_id": "i1", "edge": "end",
                                         "old_time": 10.0, "new_time": 10.6}]},
             session="g", created=2.0),
        # hesitation-marker whose verbatim text arrives by e-edit
        corr("i2", "insertion", {"operation": "chunk_insert", "after_segment_id": "b",
                                 "start_time": 15.0, "end_time": 15.4, "text": "",
                                 "label": "hesitation-marker"}, session="g", created=3.0),
        corr("t2", "text_content", {"operation": "replace_text", "segment_id": "i2",
                                    "new_text": "um"}, session="g", created=4.0),
        # split right half: a boundary decision, never a labeled span
        corr("sp", "insertion", {"operation": "chunk_insert", "after_segment_id": "a",
                                 "start_time": 5.0, "end_time": 10.0, "text": "world"},
             rationale="chunk-split", created=5.0),
        # superseded insert drops from the effective view entirely
        corr("i3", "insertion", {"operation": "chunk_insert", "after_segment_id": "c",
                                 "start_time": 22.0, "end_time": 22.3,
                                 "label": "inhale"}, created=6.0),
        # unlabeled insert rides with label None (occupies, never examples)
        corr("i4", "insertion", {"operation": "chunk_insert", "after_segment_id": "c",
                                 "start_time": 25.0, "end_time": 25.2, "text": ""},
             session="t", created=7.0),
    ]
    spans = labeled_insert_spans(segments, corrections, {"i3"})
    assert [s["insert_id"] for s in spans] == ["i1", "i2", "i4"]

    i1, i2, i4 = spans
    assert i1["label"] == "inhale" and not i1["speech"]
    assert i1["start_time"] == 10.0 and i1["end_time"] == 10.6   # nudge grew the edge
    assert i1["op_ids"] == ["i1", "n1"] and i1["session_id"] == "g"
    assert i2["label"] == "hesitation-marker"
    assert i2["text"] == "um" and i2["speech"]                   # e-edit latest-wins
    assert i2["op_ids"] == ["i2", "t2"]
    assert i4["label"] is None and i4["session_id"] == "t"


def test_negative_regions_watermark():
    """DEC 8e05b87b: label absence is a true negative only BELOW the watermark —
    no watermark yields NOTHING, spans clip/merge, the tail is never emitted."""
    from cjm_transcript_correction_core.graph import negative_regions

    assert negative_regions([], None) == []            # nothing visited
    assert negative_regions([(2.0, 4.0)], 0.0) == []
    assert negative_regions([], 10.0) == [(0.0, 10.0)]  # visited, all unaccounted
    # overlap merge + clip at the watermark
    assert negative_regions([(2.0, 4.0), (3.0, 6.0), (8.0, 12.0)], 10.0) \
        == [(0.0, 2.0), (6.0, 8.0)]
    # spans above the watermark and zero-width spans occupy nothing
    assert negative_regions([(12.0, 15.0)], 10.0) == [(0.0, 10.0)]
    assert negative_regions([(5.0, 5.0)], 10.0) == [(0.0, 10.0)]


def test_extract_spine_dataset_gate_and_policy():
    """The leg-2 orchestrating fold: the gate decides eligibility, the purpose
    policy cuts EXAMPLES (skips counted, never silent), every span occupies for
    the negative fold whatever its purpose, and speech clips at the watermark."""
    from cjm_transcript_correction_core.graph import extract_spine_dataset
    from cjm_transcript_correction_core.models import SpineSegment

    segments = [
        SpineSegment(id="a", index=0, text="hello", start_time=0.0, end_time=10.0),
        SpineSegment(id="b", index=1, text="", start_time=10.0, end_time=20.0),
        SpineSegment(id="c", index=2, text="tail speech", start_time=20.0, end_time=30.0),
    ]

    def ins(cid, start, end, label=None, session="g"):
        p = {"operation": "chunk_insert", "after_segment_id": "b",
             "start_time": start, "end_time": end, "text": ""}
        if label:
            p["label"] = label
        return {"id": cid, "correction_type": "insertion", "payload": p,
                "session_id": session, "created_at": start}

    corrections = [
        ins("i1", 10.0, 10.6, label="inhale", session="g"),
        ins("i2", 12.0, 12.5, label="inhale", session="t"),   # excluded purpose
        ins("i3", 13.0, 13.5),                                # unlabeled
        ins("i4", 16.0, 16.5, label="inhale", session="g"),   # reserved tail
    ]
    gate = {"extraction_status": "in_progress", "annotated_through": 15.0}
    r = extract_spine_dataset(segments, corrections, set(), gate,
                              include_session_ids={"g"})
    assert r["eligible"] and r["status"] == "in_progress" and r["watermark"] == 15.0
    assert [e["insert_id"] for e in r["examples"]] == ["i1"]
    assert r["skipped"] == {"session_purpose": 1, "unlabeled": 1, "above_watermark": 1}
    # speech: 'a' rides, 'c' starts above the watermark, 'b' is empty
    assert [(s["segment_id"], s["start_time"], s["end_time"]) for s in r["speech"]] \
        == [("a", 0.0, 10.0)]
    # negatives: EVERY span occupies (i2/i3 excluded from examples stay UNKNOWN,
    # never negative); i4 sits in the reserved tail
    assert [(g["start_time"], g["end_time"]) for g in r["negatives"]] \
        == [(10.6, 12.0), (12.5, 13.0), (13.5, 15.0)]

    # the gate decides eligibility: excluded spine / no watermark = nothing
    assert extract_spine_dataset(segments, corrections, set(),
                                 {"extraction_status": "excluded",
                                  "annotated_through": 15.0})["eligible"] is False
    assert extract_spine_dataset(segments, corrections, set(), None)["eligible"] is False
    got = extract_spine_dataset(segments, corrections, set(),
                                {"extraction_status": "signed_off"})
    assert got["eligible"] is False and got["examples"] == []


def test_gap_insert_run_resorts_by_effective_times():
    """FINDING 2ba9e368 (the welded-carve inversion): a split's right half is
    BORN at the pre-carve cut (payload start before the isolation insert),
    then nudged PAST it — birth order must yield to effective times at
    projection, while full ties (zero-width weld stacks) keep the rank/birth
    order (131ba57a)."""
    from cjm_transcript_correction_core.graph import reorder_gap_inserts
    segs = [SpineSegment(id="a", index=0, text="head text",
                         start_time=0.1, end_time=16.0),
            SpineSegment(id="b", index=1, text="next", start_time=16.3, end_time=19.5)]

    def _ins(cid, start, end, created, label=None, rank=None):
        p = {"operation": "chunk_insert", "after_segment_id": "a",
             "start_time": start, "end_time": end}
        if label:
            p["label"] = label
        if rank is not None:
            p["rank"] = rank
        return {"id": cid, "correction_type": "insertion", "status": "applied",
                "created_at": created, "payload": p}

    corrections = [
        _ins("half", 10.8, 15.95, 100.0),            # split right half, born at the cut
        _ins("inh", 11.0, 11.0, 200.0, label="inhale"),  # isolation insert at the weld
        {"id": "n1", "correction_type": "timing", "status": "applied",
         "created_at": 300.0,
         "payload": {"operation": "time_nudge", "segment_id": "inh",
                     "edits": [{"segment_id": "inh", "edge": "end",
                                "old_time": 11.0, "new_time": 11.3}]}},
        {"id": "n2", "correction_type": "timing", "status": "applied",
         "created_at": 301.0,
         "payload": {"operation": "time_nudge", "segment_id": "half",
                     "edits": [{"segment_id": "half", "edge": "start",
                                "old_time": 10.8, "new_time": 11.3},
                               {"segment_id": "a", "edge": "end",
                                "old_time": 16.0, "new_time": 11.0}]}},
    ]
    projected = project_effective_spine(segs, corrections)
    assert [s.id for s in projected] == ["a", "inh", "half", "b"]
    assert (projected[1].start_time, projected[1].end_time) == (11.0, 11.3)
    assert (projected[2].start_time, projected[2].end_time) == (11.3, 15.95)

    # Full-tie weld stack: identical effective starts keep rank order (stable
    # sort; lower rank first — the walked-order tie-break).
    tied = [
        _ins("t1", 12.0, 12.0, 100.0, rank=0.0),
        _ins("t2", 12.0, 12.0, 200.0, rank=-1.0),
    ]
    out = reorder_gap_inserts(project_effective_spine(segs, tied), tied)
    assert [s.id for s in out] == ["a", "t2", "t1", "b"]


def test_build_speech_overlay_correction_shape_and_validation():
    from cjm_transcript_correction_core.graph import build_speech_overlay_correction
    anchor = {"kind": "span", "segment_id": "c", "char_start": 0, "char_end": 5,
              "text_snapshot": "world"}
    node, edges = build_speech_overlay_correction(
        "src1", anchor, "hesitation-marker", 10.0, 10.6, "world", session_id="s1",
        words=[{"s": 10.0, "e": 10.6, "text": "world"}], snap="fa-word",
        note="drive sample")
    props = node["properties"]
    assert props["correction_type"] == "annotation"
    assert props["payload"]["operation"] == "speech_overlay"
    assert props["payload"]["label"] == "hesitation-marker"
    assert props["payload"]["snap"] == "fa-word"
    assert props["payload"]["anchor"]["text_snapshot"] == "world"
    assert props["payload"]["words"][0]["e"] == 10.6
    assert [(e["relation_type"], e["target_id"]) for e in edges] == [("CORRECTS", "c")]
    # supersession is an edge (re-annotate / relabel)
    _, edges = build_speech_overlay_correction(
        "src1", anchor, "false-start", 10.0, 10.6, "world", session_id="s1",
        supersedes_id="o0")
    assert ("SUPERSEDES", "o0") in [(e["relation_type"], e["target_id"]) for e in edges]

    def rejects(**kw):
        args = dict(source_id="src1", anchor=anchor, label="word-repeat",
                    start_time=10.0, end_time=10.6, text="world", session_id="s1")
        args.update(kw)
        try:
            build_speech_overlay_correction(**args)
        except ValueError:
            return True
        return False
    assert rejects(label="  ")
    assert rejects(label="-x")           # punctuation-led labels reserved for gestures
    assert rejects(anchor={"kind": "segment", "segment_id": "c"})  # span-only identity
    assert rejects(text="   ")           # a sample without its words is no sample
    assert rejects(end_time=10.0)        # zero/negative duration


def test_speech_overlay_never_touches_projection():
    """The fc42614d invariant: overlays are spans OVER words — they never cut
    the spine; corrections_to_edits has no arm for correction_type "annotation"."""
    from cjm_transcript_correction_core.graph import build_speech_overlay_correction
    node, _ = build_speech_overlay_correction(
        "src1", {"kind": "span", "segment_id": "a", "char_start": 0, "char_end": 5,
                 "text_snapshot": "hello"},
        "hesitation-marker", 1.0, 1.4, "hello", session_id="s1")
    props = dict(node["properties"])
    props["id"] = node["id"]
    assert corrections_to_edits([props]) == []
    out = project_effective_spine(SEGS, [props])
    assert [(s.id, s.text) for s in out] == [(s.id, s.text) for s in SEGS]


def test_speech_overlay_lifecycle_and_spans_fold():
    from cjm_transcript_correction_core.graph import (active_speech_overlays,
                                                      build_speech_overlay_correction,
                                                      speech_overlay_spans)
    o1, _ = build_speech_overlay_correction(
        "src1", {"kind": "span", "segment_id": "a", "char_start": 0, "char_end": 5,
                 "text_snapshot": "hello"},
        "hesitation-marker", 5.0, 5.4, "hello", session_id="s1", snap="fa-word")
    o2, _ = build_speech_overlay_correction(
        "src1", {"kind": "span", "segment_id": "c", "char_start": 0, "char_end": 5,
                 "text_snapshot": "world"},
        "word-repeat", 2.0, 2.5, "world", session_id="s2", snap="estimated")
    rows = []
    for n in (o1, o2):
        p = dict(n["properties"])
        p["id"] = n["id"]
        rows.append(p)
    assert [c["id"] for c in active_speech_overlays(rows, set())] == [o1["id"], o2["id"]]
    # removal = reject-as-supersede (the mark-dismiss shape)
    rej, _ = build_reject_review("src1", o1["id"], session_id="s1")
    rp = dict(rej["properties"])
    rp["id"] = rej["id"]
    assert [c["id"] for c in active_speech_overlays(rows + [rp], {o1["id"]})] == [o2["id"]]
    # the extraction fold: time-ordered records carrying span + label + text + anchor
    spans = speech_overlay_spans(rows, set())
    assert [r["overlay_id"] for r in spans] == [o2["id"], o1["id"]]  # (start_time) order
    r = spans[1]
    assert r["label"] == "hesitation-marker" and r["text"] == "hello"
    assert r["segment_id"] == "a" and r["char_end"] == 5 and r["snap"] == "fa-word"
    assert r["session_id"] == "s1" and r["op_ids"] == [o1["id"]]


def test_extract_spine_dataset_overlay_cuts():
    """Overlays ride the same watermark + purpose cuts as insert examples and
    never occupy negative regions (they live INSIDE speech)."""
    from cjm_transcript_correction_core.graph import (build_speech_overlay_correction,
                                                      extract_spine_dataset)
    from cjm_transcript_correction_core.models import SpineSegment
    segs = [SpineSegment(id="a", index=0, text="hello there", start_time=0.0, end_time=2.0),
            SpineSegment(id="b", index=1, text="again", start_time=6.0, end_time=7.0)]
    rows = []
    for label, seg, s, e, sess in (("hesitation-marker", "a", 0.5, 0.9, "s1"),
                                   ("word-repeat", "a", 1.2, 1.6, "s2"),
                                   ("false-start", "b", 6.2, 6.6, "s1")):
        n, _ = build_speech_overlay_correction(
            "src1", {"kind": "span", "segment_id": seg, "char_start": 0, "char_end": 5,
                     "text_snapshot": "hello"}, label, s, e, "hello", session_id=sess)
        p = dict(n["properties"])
        p["id"] = n["id"]
        rows.append(p)
    gate = {"extraction_status": "in_progress", "annotated_through": 5.0}
    out = extract_spine_dataset(segs, rows, set(), gate, include_session_ids={"s1"})
    assert [r["label"] for r in out["overlays"]] == ["hesitation-marker"]
    assert out["skipped"]["overlay_session_purpose"] == 1     # s2 cut by purpose policy
    assert out["skipped"]["overlay_above_watermark"] == 1     # 6.2 > watermark 5.0
    # negatives derive from speech + insert spans only — the overlay at
    # 0.5-0.9 sits inside speech [0,2] and must not re-shape the gaps
    assert out["negatives"] == [{"start_time": 2.0, "end_time": 5.0}]
