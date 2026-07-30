"""Tests for the leg-4 proposal plumbing — proposal-set discovery, spine
anchoring, and the reserved-tail verdict join (all pure / filesystem-only;
DECs 8e05b87b + 8cf12c22: verdicts DERIVE, never stored)."""
import json

from cjm_transcript_correction_core.graph import bench_event_proposals
from cjm_transcript_correction_core.models import SpineSegment
from cjm_transcript_correction_core.signals import (event_span_proposals,
                                                    load_event_proposal_set)


def _seg(i, start, end, text="words"):
    return SpineSegment(id=f"seg-{i}", index=i, text=text, start_time=start, end_time=end)


def _prop(pid, start, end, label="inhale", score=0.9):
    return {"proposal_id": pid, "label": label,
            "start_time": start, "end_time": end, "score": score}


def _insert(iid, start, end, label="inhale", session="s1"):
    return {"insert_id": iid, "label": label, "start_time": start, "end_time": end,
            "text": "", "speech": False, "session_id": session,
            "session_ids": [session], "op_ids": [iid]}


# ---- event_span_proposals (the propose lane's anchor fold) ----

def test_proposals_anchor_to_preceding_segment():
    """A gap proposal anchors to the segment it would be inserted AFTER — the
    last segment whose start precedes it (the chunk-insert after-anchor)."""
    segs = [_seg(0, 0.0, 10.0), _seg(1, 10.5, 20.0), _seg(2, 20.5, 30.0)]
    anchored = event_span_proposals(segs, [_prop("p1", 10.1, 10.4), _prop("p2", 20.1, 20.3)])
    assert set(anchored) == {"seg-0", "seg-1"}
    assert anchored["seg-0"][0]["proposal_id"] == "p1"
    assert anchored["seg-1"][0]["proposal_id"] == "p2"


def test_proposal_before_first_segment_anchors_to_it():
    segs = [_seg(0, 5.0, 10.0)]
    anchored = event_span_proposals(segs, [_prop("p1", 1.0, 1.3)])
    assert anchored == {"seg-0": [_prop("p1", 1.0, 1.3)]}


def test_proposals_overlapping_materialized_inserts_drop():
    segs = [_seg(0, 0.0, 10.0), _seg(1, 10.5, 20.0)]
    props = [_prop("p1", 10.1, 10.4), _prop("p2", 15.0, 15.3)]
    anchored = event_span_proposals(segs, props, occupied=[(10.0, 10.45)])
    assert [p["proposal_id"] for ps in anchored.values() for p in ps] == ["p2"]


def test_multiple_proposals_stack_on_one_anchor_in_time_order():
    segs = [_seg(0, 0.0, 10.0), _seg(1, 30.0, 40.0)]
    anchored = event_span_proposals(
        segs, [_prop("late", 20.0, 20.3), _prop("early", 12.0, 12.3)])
    assert [p["proposal_id"] for p in anchored["seg-0"]] == ["early", "late"]


# ---- bench_event_proposals (the verdict join) ----

def test_bench_accept_edit_reject_missed():
    proposals = [
        _prop("pa", 100.0, 100.3),   # materialized exactly -> accepted
        _prop("pe", 200.0, 200.3),   # materialized, end moved 0.4s -> edited
        _prop("pr", 300.0, 300.3),   # never materialized -> rejected
    ]
    inserts = [
        _insert("ia", 100.0, 100.3),
        _insert("ie", 200.05, 200.7),
        _insert("im", 400.0, 400.4),  # manual insert no proposal covered -> missed
    ]
    report = bench_event_proposals(proposals, inserts, (50.0, None), tolerance=0.15)
    verdicts = {v["proposal_id"]: v["verdict"] for v in report["verdicts"]}
    assert verdicts == {"pa": "accepted", "pe": "edited", "pr": "rejected"}
    assert report["counts"] == {"accepted": 1, "edited": 1, "rejected": 1,
                                "proposals": 3, "missed": 1}
    assert report["missed"][0]["insert_id"] == "im"
    assert report["rates"]["accepted"] == round(1 / 3, 4)


def test_bench_matching_is_label_scoped_and_windowed():
    proposals = [_prop("p1", 100.0, 100.3)]
    inserts = [
        _insert("wrong-label", 100.0, 100.3, label="hesitation-marker"),
        _insert("below-window", 10.0, 10.3),
    ]
    report = bench_event_proposals(proposals, inserts, (50.0, 200.0))
    assert report["verdicts"][0]["verdict"] == "rejected"
    # the below-window insert is out of scope entirely: not missed either
    assert report["counts"]["missed"] == 1  # only the wrong-label one, in-window unmatched


def test_bench_one_to_one_greedy_match():
    # two proposals near one insert: only the closer one matches
    proposals = [_prop("near", 100.0, 100.3), _prop("far", 100.6, 100.9)]
    inserts = [_insert("i1", 100.0, 100.3)]
    report = bench_event_proposals(proposals, inserts, (0.0, None))
    verdicts = {v["proposal_id"]: v["verdict"] for v in report["verdicts"]}
    assert verdicts == {"near": "accepted", "far": "rejected"}


# ---- load_event_proposal_set (discovery) ----

def _write_set(root, set_id, created_at, content_hash="sha256:abc", source_id="src-1"):
    d = root / "proposals" / set_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "format": "cjm-capability-pyannote/proposal-set-manifest",
        "proposal_set_id": set_id, "created_at": created_at,
        "source": {"content_hash": content_hash, "source_id": source_id},
        "files": {"proposals": "proposals.jsonl"},
    }))
    (d / "proposals.jsonl").write_text(json.dumps(_prop(f"{set_id}-p", 1.0, 1.2)) + "\n")


def test_load_event_proposal_set_latest_match_wins(tmp_path):
    _write_set(tmp_path, "propset_a", 100.0)
    _write_set(tmp_path, "propset_b", 200.0)
    _write_set(tmp_path, "propset_other", 300.0, content_hash="sha256:zzz",
               source_id="src-2")
    got = load_event_proposal_set(str(tmp_path), content_hash="sha256:abc")
    assert got["manifest"]["proposal_set_id"] == "propset_b"
    assert got["proposals"][0]["proposal_id"] == "propset_b-p"
    # source_id fallback matches too
    got = load_event_proposal_set(str(tmp_path), source_id="src-2")
    assert got["manifest"]["proposal_set_id"] == "propset_other"
    assert load_event_proposal_set(str(tmp_path), content_hash="sha256:nope") is None


def test_load_event_proposal_set_no_dir(tmp_path):
    assert load_event_proposal_set(str(tmp_path), content_hash="sha256:abc") is None
