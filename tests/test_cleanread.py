"""Tests for cjm_transcript_correction_core.cleanread — L1 of the reading ladder
(design 6752db0a; the twice-subtraction from bbf8bafd (a)): filler lines drop,
filterable overlay spans cut with a visible marker, KEEP labels never cut,
word-repeat keeps its last unit, an emptied line drops, elisions are recorded
and rendered in an L1 pack whose digest differs from L0. Pure — no runtime."""
from cjm_transcript_correction_core.cleanread import (CLEAN_READ_FILTER_LABELS, ELISION_MARKER,
                                                      clean_read, clean_read_pack_read,
                                                      clean_read_segments, clean_read_summary,
                                                      repeat_survivor, subtract_spans)
from cjm_transcript_correction_core.models import SpineSegment
from cjm_transcript_correction_core.spans import FILTERABLE_OVERLAY_LABELS
from cjm_transcript_correction_core.strata import build_filter_pack, render_filter_pack

SEGS = [
    SpineSegment(id="s0", index=0, text="Um", start_time=0.0, end_time=0.4),
    SpineSegment(id="s1", index=1, text="", start_time=0.4, end_time=0.6),   # wordless: never a line
    SpineSegment(id="s2", index=2, text="Um, so we have the the data center.", start_time=0.6, end_time=4.0),
    SpineSegment(id="s3", index=3, text="You know, it's much, much larger, uh, than before.",
                 start_time=4.0, end_time=9.0),
    SpineSegment(id="s4", index=4, text="Okay.", start_time=9.0, end_time=9.3),
    SpineSegment(id="s5", index=5, text="Uh, um,", start_time=9.3, end_time=9.9),
    SpineSegment(id="s6", index=6, text="I think we— what we found was that that works.", start_time=9.9, end_time=13.0),
]


def _stratum(sid, cat, ids):
    return {"id": sid, "correction_type": "stratum", "status": "applied", "actor": "human",
            "payload": {"operation": "classify", "category": cat, "segment_ids": ids,
                        "start_time": 0.0, "end_time": 1.0}}


def _overlay(oid, seg_id, cs, ce, snap, label):
    return {"id": oid, "correction_type": "annotation", "status": "applied", "actor": "human",
            "payload": {"operation": "speech_overlay", "label": label, "text": snap,
                        "anchor": {"kind": "span", "segment_id": seg_id, "char_start": cs,
                                   "char_end": ce, "text_snapshot": snap},
                        "start_time": 0.0, "end_time": 0.1}}


STRATA = [_stratum("f1", "filler", ["s0"]), _stratum("f2", "filler", ["s4"])]
OVERLAYS = [
    _overlay("o1", "s2", 0, 3, "Um,", "hesitation-marker"),
    _overlay("o2", "s2", 15, 22, "the the", "word-repeat"),
    _overlay("o3", "s3", 0, 9, "You know,", "discourse-marker"),
    _overlay("o4", "s3", 15, 25, "much, much", "emphasis-repeat"),      # KEEP: never cut
    _overlay("o5", "s3", 34, 37, "uh,", "hesitation-marker"),
    _overlay("o6", "s5", 0, 3, "Uh,", "hesitation-marker"),
    _overlay("o7", "s5", 4, 7, "um,", "hesitation-marker"),           # s5 is emptied by its overlays
    _overlay("o8", "s6", 0, 11, "I think we—", "false-start"),
    _overlay("o9", "s6", 33, 42, "that that", "coincidental-repeat"),  # KEEP
]


def test_filter_labels_match_the_span_lane():
    assert tuple(CLEAN_READ_FILTER_LABELS) == tuple(FILTERABLE_OVERLAY_LABELS)


def test_repeat_survivor_keeps_the_last_unit():
    assert "the the"[slice(*repeat_survivor("the the"))] == "the"
    assert "we, we"[slice(*repeat_survivor("we, we"))] == "we"
    assert "and and and"[slice(*repeat_survivor("and and and"))] == "and"
    assert "Do we, do we"[slice(*repeat_survivor("Do we, do we"))] == "do we"
    assert "the, the the"[slice(*repeat_survivor("the, the the"))] == "the"   # not a clean unit: last token


def test_subtract_spans_leaves_a_visible_marker_and_tidies_seams():
    t = "You know, it's much, much larger, uh, than before."
    assert subtract_spans(t, [(0, 9), (34, 37)]) == "[…] it's much, much larger, […] than before."
    assert subtract_spans(t, [(0, 9), (34, 37)], marker="") == "it's much, much larger, than before."
    assert subtract_spans("Um, so we have the the data center.", [(0, 3), (15, 19)], marker="") == \
        "so we have the data center."


def test_subtract_spans_keeps_a_scope_operator_the_source_wrote():
    # user-caught on the first content-strata walk: `uh thrust::reduce` reached a proposer as
    # `[…] thrust:reduce` — the seam tidy collapsed ADJACENT colons, and a row flagged the "typo"
    assert subtract_spans("uh thrust::reduce kernel", [(0, 2)]) == "[…] thrust::reduce kernel"
    assert subtract_spans("uh thrust::reduce kernel", [(0, 2)], marker="") == "thrust::reduce kernel"
    assert subtract_spans("so std::vector, um, std::string", [(16, 19)], marker="") == "so std::vector, std::string"
    assert subtract_spans("um ::max is global", [(0, 2)], marker="") == "::max is global"
    # the seams it exists for still close
    assert subtract_spans("a, um, b", [(3, 5)], marker="") == "a, b"
    assert subtract_spans("um, so we go", [(0, 2)], marker="") == "so we go"


def test_clean_read_subtracts_twice_with_visible_elisions():
    lines = clean_read(SEGS, STRATA, OVERLAYS)
    assert [ln["id"] for ln in lines] == ["s2", "s3", "s6"]
    by = {ln["id"]: ln for ln in lines}
    assert by["s2"]["text"] == "[…] so we have […] the data center."
    assert by["s2"]["raw_text"] == "Um, so we have the the data center."
    assert [(m["label"], m["cut"]) for m in by["s2"]["elisions"]] == [("hesitation-marker", "Um,"),
                                                                       ("word-repeat", "the ")]
    assert by["s2"]["dropped_before"] == [{"id": "s0", "index": 0, "why": ["filler"]}]   # s1 is wordless
    assert by["s3"]["text"] == "[…] it's much, much larger, […] than before."           # emphasis kept
    assert by["s3"]["dropped_before"] == []
    assert by["s6"]["dropped_before"] == [{"id": "s4", "index": 4, "why": ["filler"]},
                                          {"id": "s5", "index": 5, "why": ["emptied by hesitation-marker"]}]
    assert by["s6"]["text"] == "[…] what we found was that that works."                   # coincidental kept
    assert clean_read_summary(lines) == {"lines_kept": 3, "lines_dropped": 3, "spans_cut": 5}
    # a clean-verbatim consumer takes the same projection without markers
    plain = clean_read(SEGS, STRATA, OVERLAYS, marker="")
    assert [ln["text"] for ln in plain] == ["so we have the data center.",
                                            "it's much, much larger, than before.",
                                            "what we found was that that works."]


def test_clean_read_reanchors_after_an_edit_and_records_a_vanished_span():
    edited = [SpineSegment(id="s2", index=2, text="Um, so we now have the the data center.",
                           start_time=0.6, end_time=4.0),
              SpineSegment(id="s3", index=3, text="It's larger than before.", start_time=4.0, end_time=9.0)]
    lines = clean_read(edited, [], OVERLAYS)
    assert lines[0]["text"] == "[…] so we now have […] the data center."     # 'the the' re-located by snapshot
    gone = [m for m in lines[1]["elisions"] if m["cut"] is None]
    assert [m["label"] for m in gone] == ["discourse-marker", "hesitation-marker"]   # visible, not silent
    assert lines[1]["text"] == "It's larger than before."


def test_l1_pack_carries_the_read_and_renders_the_gaps():
    lines = clean_read(SEGS, STRATA, OVERLAYS)
    l1 = build_filter_pack("src", "t", "sha256:skel", clean_read_segments(lines, SEGS),
                           read=clean_read_pack_read(lines))
    l0 = build_filter_pack("src", "t", "sha256:skel", SEGS)
    assert l1["read"] == {"layer": "clean", "marker": ELISION_MARKER, "lines_kept": 3,
                          "lines_dropped": 3, "spans_cut": 5}          # gaps ride the rows, not the record
    assert l1["digest"] != l0["digest"] and l1["version"] == "0.2.0"
    assert [r["id"] for r in l1["segments"]] == ["s2", "s3", "s6"]
    assert l1["segments"][0]["dropped_before"] == [{"id": "s0", "index": 0, "why": ["filler"]}]
    assert "dropped_before" not in l1["segments"][1]
    brief = render_filter_pack(l1)
    assert "CLEAN READ (L1): 3 filler line(s) and 5 disfluent span(s)" in brief
    assert "(… 1 line(s) elided: filler …)\n[0] 00:00.6–00:04.0  […] so we have" in brief
    assert "(… 2 line(s) elided: emptied by hesitation-marker, filler …)\n[2]" in brief
    assert "(…" not in render_filter_pack(l0)
