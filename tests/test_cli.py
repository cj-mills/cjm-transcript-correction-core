"""Tests for cjm_transcript_correction_core.cli — parser smoke checks (no runtime).

Fresh projection at the golden-reference flip: the notebook's eval:false check cell
still exercised the stage-5-retired --secondary-manifest flag (stale-superseded);
these tests cover the live run/review surface instead."""
import pytest

from cjm_transcript_correction_core.cli import build_parser


def test_run_defaults_and_flags():
    p = build_parser()
    ns = p.parse_args(["run", "/tmp/decomp.json", "-y"])
    assert ns.command == "run" and ns.yes
    assert ns.graph_capability == "cjm-capability-graph-sqlite"
    assert ns.graph_db_path is None
    assert ns.rendition is None       # auto-select the decomposed rendition
    assert ns.session is None and ns.reopen is False
    assert ns.no_prune is False
    assert ns.actor == "human"


def test_secondary_manifest_is_retired():
    # stage 5: the cross-transcriber diff is intra-graph; the flag is gone
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["run", "/tmp/decomp.json", "--secondary-manifest", "/tmp/vox.json"])


def test_review_subcommand():
    p = build_parser()
    ns = p.parse_args(["review", "/tmp/decomp.json", "--review-max", "5",
                       "--session", "sess1", "--reopen", "--rendition", "raw"])
    assert ns.command == "review"
    assert ns.review_max == 5
    assert ns.session == "sess1" and ns.reopen is True
    assert ns.rendition == "raw"


def test_gate_subcommand():
    """The gate surface (DEC 8e05b87b): show mode needs no --source; assert mode
    validates --status against the EXTRACTION_STATUSES vocabulary."""
    p = build_parser()
    ns = p.parse_args(["gate"])
    assert ns.command == "gate" and ns.status is None and ns.source is None
    ns = p.parse_args(["gate", "--source", "Chris", "--status", "in_progress",
                       "--annotated-through", "2016.2"])
    assert ns.status == "in_progress" and ns.annotated_through == "2016.2"
    ns = p.parse_args(["gate", "--source", "x", "--status", "signed_off",
                       "--annotated-through", "end"])
    assert ns.annotated_through == "end"
    with pytest.raises(SystemExit):
        p.parse_args(["gate", "--status", "done"])   # not in the vocabulary


def test_filter_lane_subcommands():
    """The filtering lane's headless surface (DECs 304fd984 + 9d4c0a38; pass 1
    per bc8dbbdd): pack -> ingest -> confirm. Confirm without gestures lists;
    the batch-accept is an EXPLICIT flag, never a default."""
    p = build_parser()
    ns = p.parse_args(["filter-pack", "--source", "Seven", "--skeleton", "abcd",
                       "--window", "0", "600"])
    assert ns.command == "filter-pack" and ns.window == [0.0, 600.0] and ns.out_dir is None
    ns = p.parse_args(["filter-ingest", "--pack", "p.json", "--rows", "r.jsonl",
                       "--proposer", "reader-1"])
    assert ns.proposer_kind == "claude-code-subagent" and ns.model is None
    ns = p.parse_args(["filter-confirm", "--source", "Seven"])
    assert ns.accept is None and ns.accept_tier1 is False and ns.accept_all is False
    assert ns.retract is None and ns.watermark is None and ns.tier2 is False
    ns = p.parse_args(["filter-confirm", "--source", "Seven", "--accept", "a1", "--accept", "b2",
                       "--retract", "s9", "--watermark", "end", "--purpose", "feature-test"])
    assert ns.accept == ["a1", "b2"] and ns.retract == ["s9"] and ns.watermark == "end"
    with pytest.raises(SystemExit):
        p.parse_args(["filter-ingest", "--pack", "p.json"])   # rows + proposer required


def test_extract_subcommand():
    """The extract surface (flywheel leg 2): sibling of stats sharing the
    workspace/graph plumbing; --include-purpose repeats (default None = the
    genuine-only policy applied downstream)."""
    p = build_parser()
    ns = p.parse_args(["extract"])
    assert ns.command == "extract"
    assert ns.include_purpose is None and ns.source is None and ns.output_dir is None
    assert ns.graph_capability == "cjm-capability-graph-sqlite"
    ns = p.parse_args(["extract", "--source", "Chris",
                       "--include-purpose", "genuine",
                       "--include-purpose", "feature-test",
                       "--output-dir", "/tmp/ds", "--rendition", "raw"])
    assert ns.include_purpose == ["genuine", "feature-test"]
    assert ns.source == "Chris" and ns.output_dir == "/tmp/ds"


def test_overlay_event_rows_writer():
    """The overlay WRITER half (the fold computed overlays; extract never
    emitted them — caught 2026-08-04): rows carry kind=speech_overlay with
    snap/words/provenance, and the anchoring-spine cut drops overlays whose
    anchor segment belongs to another spine (no double emission across
    multiple eligible gated spines)."""
    from cjm_transcript_correction_core.cli import overlay_event_rows
    src = {"id": "src1", "title": "T", "path": "/a.mp3", "content_hash": "sha256:x"}
    overlays = [
        {"overlay_id": "o1", "segment_id": "seg1", "label": "hesitation-marker",
         "text": "uh", "start_time": 1.0, "end_time": 1.2, "snap": "nudged",
         "words": [{"s": 1.0, "e": 1.2, "text": "uh"}],
         "session_id": "s1", "op_ids": ["o1"]},
        {"overlay_id": "o2", "segment_id": "OTHER-SPINE", "label": "word-repeat",
         "text": "so so", "start_time": 2.0, "end_time": 2.4, "snap": "fa-word",
         "words": [], "session_id": "s1", "op_ids": ["o2"]},
    ]
    rows, foreign = overlay_event_rows(src, "sha256:skel", overlays, {"seg1", "seg2"})
    assert foreign == 1 and len(rows) == 1
    row = rows[0]
    assert row["kind"] == "speech_overlay" and row["overlay_id"] == "o1"
    assert row["skeleton_hash"] == "sha256:skel" and row["segment_id"] == "seg1"
    assert row["label"] == "hesitation-marker" and row["snap"] == "nudged"
    assert row["split"] == "train"
    assert row["provenance"] == {"tag": "real", "sessions": ["s1"], "op_ids": ["o1"]}
    assert row["words"] == [{"s": 1.0, "e": 1.2, "text": "uh"}]


def test_export_wordless_propset_subcommand():
    """The export surface (f5d080b9 direction a): sibling of transfer-wordless
    sharing its source/spine selectors; the exported set is the transfer donor
    set, so the flags mirror transfer minus the destination/commit half."""
    p = build_parser()
    ns = p.parse_args(["export-wordless-propset", "--source", "Chris",
                       "--from-skeleton", "1223b0ab"])
    assert ns.command == "export-wordless-propset"
    assert ns.source == "Chris" and ns.from_skeleton == "1223b0ab"
    assert ns.labels is None and ns.out_dir is None and ns.dry_run is False
    assert ns.graph_capability == "cjm-capability-graph-sqlite"
    ns = p.parse_args(["export-wordless-propset", "--source", "x",
                       "--from-skeleton", "legacy", "--labels", "inhale",
                       "--labels", "click", "--out-dir", "/tmp/ps", "--dry-run"])
    assert ns.labels == ["inhale", "click"] and ns.out_dir == "/tmp/ps" and ns.dry_run
    with pytest.raises(SystemExit):
        p.parse_args(["export-wordless-propset", "--source", "x"])  # --from-skeleton required


def test_transfer_wordless_no_splits_flag():
    """Speaker splits transfer by default (54aac7d3); --no-splits leaves
    them, riding beside the existing tolerance / dry-run flags."""
    p = build_parser()
    ns = p.parse_args(["transfer-wordless", "--source", "x",
                       "--from-skeleton", "a", "--to-skeleton", "b"])
    assert ns.no_splits is False and ns.tolerance == 0.05 and ns.dry_run is False
    ns = p.parse_args(["transfer-wordless", "--source", "x", "--from-skeleton", "a",
                       "--to-skeleton", "b", "--no-splits", "--dry-run"])
    assert ns.no_splits and ns.dry_run


def test_plan_transfer_rows_anchors_dups_and_unanchored():
    """The event half of the transfer plan (pure, factored out of the CLI
    verb for the in-app driver, 9af9793a): source-time anchoring after the
    last layer-0 segment starting at or before the donor, tail donors get a
    None right flank, a donor before the spine is unanchored, and a same-
    label destination insert within tolerance is a dup — a different label
    at the same time is not."""
    from types import SimpleNamespace

    from cjm_transcript_correction_core.cli import plan_transfer_rows

    to_l0 = [SimpleNamespace(id="s0", index=0, start_time=10.0),
             SimpleNamespace(id="s1", index=1, start_time=20.0),
             SimpleNamespace(id="s2", index=2, start_time=30.0)]
    donors = [{"start": 9.0, "end": 9.5, "label": "inhale", "rank": 0.0},    # before the spine
              {"start": 12.0, "end": 12.3, "label": "inhale", "rank": 1.0},  # gap after s0 (dup below)
              {"start": 20.0, "end": 20.2, "label": "click", "rank": 0.0},   # at s1's start -> after s1
              {"start": 31.0, "end": 31.4, "label": "inhale", "rank": 2.0}]  # tail -> no right flank
    plan, dups, un = plan_transfer_rows(donors, [("inhale", 12.03)], to_l0, 0.05)
    assert (dups, un) == (1, 1)
    assert [(p["label"], p["after_id"], p["before_id"]) for p in plan] == [
        ("click", "s1", "s2"), ("inhale", "s2", None)]
    assert plan[1]["rank"] == 2.0 and plan[1]["start"] == 31.0
    other = plan_transfer_rows([donors[1]], [("click", 12.0)], to_l0, 0.05)
    assert other[1] == 0 and other[0][0]["after_id"] == "s0" and other[0][0]["before_id"] == "s1"
    # a rerun over the landed rows transfers nothing (idempotency)
    landed = [(p["label"], p["start"]) for p in plan]
    assert plan_transfer_rows(donors[2:], landed, to_l0, 0.05) == ([], 2, 0)


def test_write_wordless_propset_manifest(tmp_path):
    """The export engine's write half (factored for the in-app verb,
    9af9793a): a proposal set in EVENT_PROPOSAL_SET_FORMAT under out_root —
    tier-1 rows in donor order, the human-effective-layer provenance, and
    the source binding PropsetIndex.for_source joins on (content hash +
    path + skeleton hash)."""
    import json
    from pathlib import Path

    from cjm_transcript_correction_core.cli import write_wordless_propset
    from cjm_transcript_correction_core.signals import EVENT_PROPOSAL_SET_FORMAT

    media = tmp_path / "a.wav"
    media.write_bytes(b"abc")
    plan = {"donors": [{"start": 1.0, "end": 1.25, "label": "inhale", "rank": 0.5},
                       {"start": 3.0, "end": 3.1, "label": "click", "rank": 0.0}],
            "word_bearing": 0, "counts": {"inhale": 1, "click": 1},
            "from_hash": "sha256:abc", "window_end": 60.0}
    res = write_wordless_propset(plan, out_root=tmp_path / "proposals",
                                 source_id="src-1", media_path=str(media),
                                 from_skeleton="abc", ws=None)
    assert res["set_id"].startswith("propset_") and res["classes"] == ["click", "inhale"]
    assert Path(res["set_dir"]).parent == tmp_path / "proposals"
    m = json.loads(Path(res["manifest_path"]).read_text())
    assert m["format"] == EVENT_PROPOSAL_SET_FORMAT
    assert m["proposal_set_id"] == res["set_id"] and m["version"] == "0.2.0"
    assert m["model"] == {"kind": "human-effective-layer", "from_skeleton_hash": "sha256:abc"}
    assert m["source"]["source_id"] == "src-1" and m["source"]["path"] == str(media)
    assert m["source"]["content_hash"].startswith("sha256:")
    assert m["source"]["skeleton_hash"] == "sha256:abc"
    assert m["config"]["from_skeleton"] == "abc" and m["config"]["labels"] is None
    assert m["classes"] == ["click", "inhale"] and m["counts"] == {"inhale": 1, "click": 1}
    assert m["window"] == {"start": 0.0, "end": 60.0}
    rows = [json.loads(line) for line in
            (Path(res["set_dir"]) / "proposals.jsonl").read_text().splitlines()]
    assert [(r["label"], r["start_time"], r["end_time"], r["score"], r["tier"])
            for r in rows] == [("inhale", 1.0, 1.25, 0.5, 1), ("click", 3.0, 3.1, 0.0, 1)]
    # no media on disk: path-only binding, no hash
    res2 = write_wordless_propset(plan, out_root=tmp_path / "proposals",
                                  source_id="src-1", media_path="/nope.wav",
                                  from_skeleton="abc", ws=None)
    assert json.loads(Path(res2["manifest_path"]).read_text())["source"]["content_hash"] is None


def test_scan_mishomed_subcommand():
    """The scan surface (96edc646 verdict bc7ece7b): read-only QA gate sharing
    the source/spine selectors; --strict flips it into a nonzero-exit gate."""
    p = build_parser()
    ns = p.parse_args(["scan-mishomed", "--source", "Chris", "--skeleton", "ffdfd489"])
    assert ns.command == "scan-mishomed"
    assert ns.source == "Chris" and ns.skeleton == "ffdfd489"
    assert ns.fa_cache_db is None and ns.min_overlap == 0.03 and ns.strict is False
    ns = p.parse_args(["scan-mishomed", "--source", "x", "--skeleton", "legacy",
                       "--fa-cache-db", "/tmp/fa.db", "--min-overlap", "0.05", "--strict"])
    assert ns.fa_cache_db == "/tmp/fa.db" and ns.min_overlap == 0.05 and ns.strict
    with pytest.raises(SystemExit):
        p.parse_args(["scan-mishomed", "--source", "x"])  # --skeleton required
