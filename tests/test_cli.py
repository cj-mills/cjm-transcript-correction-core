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
