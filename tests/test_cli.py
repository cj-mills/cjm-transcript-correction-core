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
