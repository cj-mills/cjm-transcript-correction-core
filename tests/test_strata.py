"""Tests for cjm_transcript_correction_core.strata — the filtering lane's domain
half (DECs 304fd984 + 9d4c0a38): pack build/render, proposer-row validation,
proposal-set write/load round-trip, the stratum op, the pending worklist, the
derived-verdict bench join, the per-consumer exclusion query, and the gate's
lane fold. Pure — no runtime."""
import json

import pytest

from cjm_transcript_correction_core.graph import (build_stratum_correction,
                                                  build_extraction_gate_assertion,
                                                  corrections_to_edits,
                                                  latest_extraction_gates)
from cjm_transcript_correction_core.models import RECOMMENDED_STRATUM_CLASSES, SpineSegment
from cjm_transcript_correction_core.strata import (FILTER_LANE, FILTER_PACK_FORMAT,
                                                   FILTER_PROPOSAL_SET_FORMAT,
                                                   active_strata, bench_filter_proposals,
                                                   build_filter_pack, exclude_strata,
                                                   load_filter_proposal_sets, pack_digest,
                                                   pending_filter_proposals,
                                                   proposals_from_rows, render_filter_pack,
                                                   render_filter_propset_markdown,
                                                   select_span_segments,
                                                   STRATUM_GLOSSES, validate_proposal_rows,
                                                   write_filter_propset)

SEGS = [
    SpineSegment(id="s0", index=0, text="Opening credits, read by the author.", start_time=0.0, end_time=3.0),
    SpineSegment(id="s1", index=1, text="", start_time=3.0, end_time=3.4),   # wordless — left out of packs
    SpineSegment(id="s2", index=2, text="Today we talk about learning.", start_time=3.4, end_time=6.0),
    SpineSegment(id="s3", index=3, text="By the way, I use Notion for this.", start_time=6.0, end_time=8.5),
    SpineSegment(id="s4", index=4, text="Back to the main idea.", start_time=8.5, end_time=11.0),
    SpineSegment(id="s5", index=5, text="This episode is sponsored by X.", start_time=11.0, end_time=14.0),
]


def _pack(**kw):
    return build_filter_pack("src", "Chapter 1", "sha256:skel", SEGS, content_hash="sha256:media", **kw)


def _stratum(sid, cat, ids, start, end, proposal_id=None, created_at=1.0, status="applied"):
    return {"id": sid, "correction_type": "stratum", "status": status, "actor": "human",
            "created_at": created_at,
            "payload": {"operation": "classify", "source_id": "src", "category": cat,
                        "segment_ids": ids, "start_time": start, "end_time": end,
                        "proposal_id": proposal_id}}


# ---- vocabulary ----

def test_recommended_stratum_classes_are_glossed_class_tokens():
    assert all(c[:1].isalnum() for c in RECOMMENDED_STRATUM_CLASSES)
    assert len(set(RECOMMENDED_STRATUM_CLASSES)) == len(RECOMMENDED_STRATUM_CLASSES)
    assert set(RECOMMENDED_STRATUM_CLASSES) <= set(STRATUM_GLOSSES)
    assert "main-topic" not in RECOMMENDED_STRATUM_CLASSES   # absence IS main-topic
    # quotation: proposer-minted on LG ch04 (8 block quotes), ratified by the user 2026-09-01
    assert "quotation" in RECOMMENDED_STRATUM_CLASSES and "delimiters" in STRATUM_GLOSSES["quotation"]


def test_select_span_segments_is_containment_over_text_segments():
    """The span-edit gesture: the human re-states a run as a time span over the
    CURRENT effective view — contained text segments only (a neighbour touching
    the edge stays out; empties carry nothing to classify), spine order."""
    run = select_span_segments(SEGS, 3.4, 8.5)
    assert [s.id for s in run] == ["s2", "s3"]                # s1 empty, s4 starts AT 8.5 (out)
    assert [s.id for s in select_span_segments(SEGS, 3.0, 8.5)] == ["s2", "s3"]   # s1 in-span but empty
    assert [s.id for s in select_span_segments(SEGS, 3.43, 8.47)] == ["s2", "s3"]  # tolerance absorbs 0.05 jitter
    assert [s.id for s in select_span_segments(SEGS, 3.6, 8.5)] == ["s3"]         # s2 not contained
    assert select_span_segments(SEGS, 20.0, 30.0) == []
    shuffled = list(reversed(SEGS))
    assert [s.id for s in select_span_segments(shuffled, 0.0, 14.0)] == ["s0", "s2", "s3", "s4", "s5"]


# ---- pack ----

def test_pack_numbers_text_segments_only_and_binds_source():
    pack = _pack()
    assert pack["format"] == FILTER_PACK_FORMAT
    rows = pack["segments"]
    assert [r["id"] for r in rows] == ["s0", "s2", "s3", "s4", "s5"]   # s1 (empty) skipped
    assert [r["i"] for r in rows] == [0, 1, 2, 3, 4]
    assert rows[1]["index"] == 2                               # spine index rides along
    assert pack["source"] == {"source_id": "src", "title": "Chapter 1",
                              "content_hash": "sha256:media", "skeleton_hash": "sha256:skel"}
    assert pack["window"] == {"start": 0.0, "end": 14.0}
    assert [v["category"] for v in pack["vocabulary"]] == list(RECOMMENDED_STRATUM_CLASSES)
    assert pack["digest"] == pack_digest(pack) and pack["digest"].startswith("sha256:")


def test_pack_window_clips_and_existing_strata_render_in_pack_coordinates():
    strata = [_stratum("st1", "apparatus", ["s0"], 0.0, 3.0),
              _stratum("st2", "sponsor", ["s5"], 11.0, 14.0)]
    pack = _pack(window=(3.0, 9.0), strata=strata)
    assert [r["id"] for r in pack["segments"]] == ["s2", "s3", "s4"]
    assert pack["window"] == {"start": 3.0, "end": 9.0}
    assert pack["existing_strata"] == []       # both strata fall outside the window
    full = _pack(strata=strata)
    assert [(e["category"], e["from_i"], e["to_i"]) for e in full["existing_strata"]] \
        == [("apparatus", 0, 0), ("sponsor", 4, 4)]


def test_pack_digest_ignores_ids_and_timestamps():
    a, b = _pack(), _pack()
    assert a["pack_id"] != b["pack_id"] and a["digest"] == b["digest"]


def test_render_filter_pack_carries_brief_contract_and_lines():
    strata = [_stratum("st1", "apparatus", ["s0"], 0.0, 3.0)]
    md = render_filter_pack(_pack(strata=strata))
    assert "## Vocabulary" in md and "`tangent`" in md and STRATUM_GLOSSES["tangent"] in md
    assert "## Output contract" in md and '"from_i"' in md
    assert "`apparatus` lines 0–0" in md and "do not re-propose" in md
    assert "[2] 00:06.0–00:08.5  By the way, I use Notion for this." in md
    assert md.index("## Vocabulary") < md.index("## Output contract") < md.index("## Transcript")


# ---- proposer rows ----

def test_validate_rows_normalizes_and_rejects_contract_breaks():
    pack = _pack()
    ok = validate_proposal_rows([
        {"category": "tool-mention", "from_i": 2, "to_i": 2, "confidence": 0.9,
         "rationale": "Notion", "quote": "I use Notion"},
        {"category": "sponsor", "from_i": 4, "to_i": 4, "tier": 2},
    ], pack)
    assert ok[0]["tier"] == 1 and ok[0]["confidence"] == 0.9
    assert ok[1]["tier"] == 2 and ok[1]["confidence"] is None and ok[1]["rationale"] is None
    with pytest.raises(ValueError, match="row 1: category"):
        validate_proposal_rows([{"category": "-bad", "from_i": 0, "to_i": 0}], pack)
    with pytest.raises(ValueError, match="outside the pack"):
        validate_proposal_rows([{"category": "tangent", "from_i": 0, "to_i": 9}], pack)
    with pytest.raises(ValueError, match="inverted"):
        validate_proposal_rows([{"category": "tangent", "from_i": 3, "to_i": 1}], pack)
    with pytest.raises(ValueError, match="tier"):
        validate_proposal_rows([{"category": "tangent", "from_i": 0, "to_i": 0, "tier": 3}], pack)
    with pytest.raises(ValueError, match="confidence"):
        validate_proposal_rows([{"category": "tangent", "from_i": 0, "to_i": 0, "confidence": 1.5}], pack)
    with pytest.raises(ValueError, match="overlaps"):
        validate_proposal_rows([{"category": "tangent", "from_i": 0, "to_i": 2},
                                {"category": "tangent", "from_i": 2, "to_i": 3}], pack)
    # different categories may overlap (a sponsor read that is also a tool mention)
    validate_proposal_rows([{"category": "sponsor", "from_i": 4, "to_i": 4},
                            {"category": "tool-mention", "from_i": 4, "to_i": 4}], pack)


def test_proposals_from_rows_resolve_pack_positions_to_spine_identity():
    pack = _pack()
    rows = validate_proposal_rows([
        {"category": "sponsor", "from_i": 4, "to_i": 4, "quote": "sponsored by X"},
        {"category": "apparatus", "from_i": 0, "to_i": 1, "tier": 2, "confidence": 0.4},
    ], pack)
    props = proposals_from_rows(rows, pack)
    assert [p["category"] for p in props] == ["apparatus", "sponsor"]   # time order
    a, s = props
    assert a["segment_ids"] == ["s0", "s2"] and a["start_time"] == 0.0 and a["end_time"] == 6.0
    assert s["segment_ids"] == ["s5"] and s["evidence"] == {
        "pack_id": pack["pack_id"], "from_i": 4, "to_i": 4, "quote": "sponsored by X"}
    assert s["label"] == "sponsor" and a["score"] == 0.4 and a["tier"] == 2
    assert len({p["proposal_id"] for p in props}) == 2


# ---- proposal set round-trip ----

def test_write_and_load_filter_propset(tmp_path):
    pack = _pack()
    props = proposals_from_rows(validate_proposal_rows([
        {"category": "sponsor", "from_i": 4, "to_i": 4},
        {"category": "tool-mention", "from_i": 2, "to_i": 2, "tier": 2},
    ], pack), pack)
    root = tmp_path / "proposals"
    res = write_filter_propset(pack, props, out_root=root,
                               proposer={"kind": "claude-code-subagent", "name": "reader-1"})
    m = json.loads((tmp_path / "proposals" / res["set_id"] / "manifest.json").read_text())
    assert m["format"] == FILTER_PROPOSAL_SET_FORMAT
    assert m["source"]["skeleton_hash"] == "sha256:skel" and m["window"]["end"] == 14.0
    assert m["pack"] == {"pack_id": pack["pack_id"], "digest": pack["digest"], "segments": 5}
    assert m["model"]["kind"] == "claude-code-subagent"
    assert m["counts"] == {"sponsor": 1} and m["tier2_counts"] == {"tool-mention": 1}
    assert m["classes"] == ["sponsor", "tool-mention"]
    # a foreign-format set in the same root is ignored; another source's set too
    (root / "other").mkdir()
    (root / "other" / "manifest.json").write_text(json.dumps({"format": "x/other"}))
    write_filter_propset(build_filter_pack("src2", "t", None, SEGS), [], out_root=root,
                         proposer={"kind": "api"})
    sets = load_filter_proposal_sets(str(tmp_path), "src")
    assert len(sets) == 1 and sets[0]["manifest"]["proposal_set_id"] == res["set_id"]
    assert [p["category"] for p in sets[0]["proposals"]] == ["tool-mention", "sponsor"]  # time order
    assert load_filter_proposal_sets(str(tmp_path), "src", skeleton_hash="sha256:else") == []
    assert load_filter_proposal_sets(str(tmp_path / "nowhere"), "src") == []


def test_render_filter_propset_markdown_joins_pack_runs(tmp_path):
    pack = _pack()
    props = proposals_from_rows(validate_proposal_rows([
        {"category": "sponsor", "from_i": 4, "to_i": 4, "confidence": 0.8,
         "rationale": "A sponsor read.", "quote": "sponsored by X"},
        {"category": "apparatus", "from_i": 0, "to_i": 1, "tier": 2},
    ], pack), pack)
    res = write_filter_propset(pack, props, out_root=tmp_path / "proposals",
                               proposer={"kind": "api", "name": "m"})
    manifest = json.loads((tmp_path / "proposals" / res["set_id"] / "manifest.json").read_text())
    md = render_filter_propset_markdown(manifest, props, pack)
    assert md.startswith("# Filtering proposals — Chapter 1")
    # time order: apparatus (0.0s) before sponsor (11.0s); spine index range from the pack
    assert md.index("**apparatus**") < md.index("**sponsor**")
    assert "`??` **apparatus** · 00:00.0–00:06.0 · spine 0–2" in md
    assert "`?` **sponsor** · 00:11.0–00:14.0 · spine 5–5 · c=0.80" in md
    assert "**Why:** A sponsor read." in md and "**Quote:** “sponsored by X”" in md
    # the run is bold, the context line either side is italic
    assert "> **[4]** 00:11.0 · spine 5 — This episode is sponsored by X." in md
    assert "> [3] 00:08.5 · spine 4 — _Back to the main idea._" in md
    # without a pack the set still renders, by pack lines
    assert "pack lines 4..4" in render_filter_propset_markdown(manifest, props, None)


# ---- the stratum op ----

def test_build_stratum_correction_shape_and_no_effective_edit():
    node, edges = build_stratum_correction(
        "src", ["s3"], "tool-mention", "sess", skeleton_hash="sha256:skel",
        start_time=6.0, end_time=8.5, proposal_id="p1", proposal_set_id="set1",
        actor="human", note="Notion")
    props = node["properties"]
    assert node["label"] == "Correction" and props["correction_type"] == "stratum"
    assert props["payload"]["operation"] == "classify"
    assert props["payload"]["category"] == "tool-mention"
    assert props["payload"]["segment_ids"] == ["s3"] and props["payload"]["proposal_id"] == "p1"
    assert props["rationale"] == "Notion"
    assert [(e["relation_type"], e["target_id"]) for e in edges] == [("CORRECTS", "s3")]
    # a stratum never touches the effective view
    d = dict(props); d["id"] = node["id"]
    assert corrections_to_edits([d]) == []
    # reclassify = supersession
    _n2, e2 = build_stratum_correction("src", ["s3"], "tangent", "sess",
                                       supersedes_id=node["id"])
    assert ("SUPERSEDES", node["id"]) in [(e["relation_type"], e["target_id"]) for e in e2]
    with pytest.raises(ValueError):
        build_stratum_correction("src", [], "tangent", "sess")
    with pytest.raises(ValueError):
        build_stratum_correction("src", ["s3"], "-", "sess")


def test_active_strata_and_exclusion_query():
    a = _stratum("a", "sponsor", ["s5"], 11.0, 14.0)
    b = _stratum("b", "tool-mention", ["s3"], 6.0, 8.5)
    c = _stratum("c", "tangent", ["s3"], 6.0, 8.5, created_at=2.0)   # superseded below
    p = _stratum("p", "apparatus", ["s0"], 0.0, 3.0, status="proposed")
    live = active_strata([a, b, c, p, {"id": "m", "correction_type": "mark"}], {"c"})
    assert [s["id"] for s in live] == ["b", "a"]                      # time order, c + p out
    notes = exclude_strata(SEGS, live, ["sponsor", "tangent", "apparatus"])
    assert [s.id for s in notes] == ["s0", "s1", "s2", "s3", "s4"]     # sponsor dropped, tool kept
    research = exclude_strata(SEGS, live, ["tool-mention"])
    assert "s3" not in [s.id for s in research]


# ---- worklist + derived verdicts ----

def _props():
    pack = _pack()
    return pack, proposals_from_rows(validate_proposal_rows([
        {"category": "apparatus", "from_i": 0, "to_i": 0},
        {"category": "tool-mention", "from_i": 2, "to_i": 2},
        {"category": "sponsor", "from_i": 4, "to_i": 4},
        {"category": "tangent", "from_i": 3, "to_i": 3, "tier": 2},
    ], pack), pack)


def test_pending_hides_tier2_and_drops_materialized():
    _pack_, props = _props()
    by_cat = {p["category"]: p for p in props}
    strata = [_stratum("x", "apparatus", ["s0"], 0.0, 3.0, proposal_id=by_cat["apparatus"]["proposal_id"]),
              _stratum("y", "sponsor", ["s5"], 11.2, 14.0)]   # same category, overlapping, no id carry
    pend = pending_filter_proposals(props, strata)
    assert [p["category"] for p in pend] == ["tool-mention"]
    assert [p["category"] for p in pending_filter_proposals(props, strata, show_tier2=True)] \
        == ["tool-mention", "tangent"]


def test_mark_family_routing_materializes_and_benches_as_accepted():
    """Class-family routing: a proposer's mark-family row accepted AS a mark
    carries the proposal id on the mark payload — the worklist drops it and the
    bench reads it ACCEPTED (family mark), never rejected below the watermark."""
    from cjm_transcript_correction_core.graph import build_mark_correction
    from cjm_transcript_correction_core.strata import materialized_mark_ids
    _pack_, props = _props()
    by_cat = {p["category"]: p for p in props}
    pid = by_cat["tool-mention"]["proposal_id"]
    node, _edges = build_mark_correction("src", {"kind": "segment", "segment_id": "s3"},
                                         "homophone-substitution", "sess",
                                         proposal_id=pid, proposal_set_id="set1")
    assert node["properties"]["payload"]["proposal_id"] == pid
    plain, _ = build_mark_correction("src", {"kind": "segment", "segment_id": "s3"}, "suspect", "sess")
    assert "proposal_id" not in plain["properties"]["payload"]
    d = dict(node["properties"]); d["id"] = node["id"]
    mids = materialized_mark_ids([d], set())
    assert mids == {pid} and materialized_mark_ids([d], {node["id"]}) == set()
    pend = pending_filter_proposals(props, [], materialized=mids)
    assert "tool-mention" not in [p["category"] for p in pend]
    b = bench_filter_proposals(props, [], (0.0, 14.0), watermark=14.0, mark_ids=mids)
    v = {r["category"]: r for r in b["verdicts"]}
    assert v["tool-mention"]["verdict"] == "accepted" and v["tool-mention"]["family"] == "mark"
    assert v["apparatus"]["verdict"] == "rejected"


def test_bench_filter_proposals_derives_verdicts_below_watermark():
    _pack_, props = _props()
    by_cat = {p["category"]: p for p in props}
    strata = [
        _stratum("x", "apparatus", ["s0"], 0.0, 3.0, proposal_id=by_cat["apparatus"]["proposal_id"]),
        _stratum("y", "research-mark", ["s3"], 6.0, 8.5),          # tool-mention relabeled
        _stratum("z", "sponsor", ["s4", "s5"], 8.5, 14.0),         # sponsor edited (grown)
        _stratum("h", "disfluency", ["s2"], 3.4, 6.0),             # human-minted, unproposed = missed
    ]
    b = bench_filter_proposals(props, strata, (0.0, 14.0), watermark=14.0)
    v = {r["category"]: r["verdict"] for r in b["verdicts"]}
    assert v == {"apparatus": "accepted", "tool-mention": "relabeled",
                 "sponsor": "edited", "tangent": "unaccepted"}
    assert b["counts"]["tier1"] == {"accepted": 1, "edited": 1, "relabeled": 1,
                                    "rejected": 0, "unvisited": 0}
    assert b["rates"] == {"accepted": 0.333, "edited": 0.333, "relabeled": 0.333, "rejected": 0.0}
    assert [m["stratum_id"] for m in b["missed"]] == ["h"]
    # no watermark = nothing visited: an unmatched tier-1 row is UNVISITED, not rejected
    b2 = bench_filter_proposals(props, [], (0.0, 14.0))
    assert b2["counts"]["tier1"]["unvisited"] == 3 and b2["rates"] == {}
    # a watermark mid-source rejects only below it
    b3 = bench_filter_proposals(props, [], (0.0, 14.0), watermark=7.0)
    assert {r["category"]: r["verdict"] for r in b3["verdicts"] if r["tier"] == 1} \
        == {"apparatus": "rejected", "tool-mention": "rejected", "sponsor": "unvisited"}


# ---- the gate's lane fold ----

def test_extraction_gate_lanes_fold_separately():
    n_main, _ = build_extraction_gate_assertion("src", "h", "in_progress", 100.0)
    n_lane, _ = build_extraction_gate_assertion("src", "h", "in_progress", 40.0, lane=FILTER_LANE)
    assert "lane" not in n_main["properties"] and n_lane["properties"]["lane"] == FILTER_LANE
    rows = []
    for k, n in enumerate((n_main, n_lane)):
        d = dict(n["properties"]); d["id"] = n["id"]; d["created_at"] = float(k + 1)
        rows.append(d)
    main = latest_extraction_gates(rows)
    lane = latest_extraction_gates(rows, lane=FILTER_LANE)
    assert main["h"]["annotated_through"] == 100.0        # the newer LANE row did not displace it
    assert lane["h"]["annotated_through"] == 40.0
    assert latest_extraction_gates(rows, lane="other") == {}
