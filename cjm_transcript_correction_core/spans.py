"""Spans — the overlay-proposal lane's domain half (design bbf8bafd, extending the
reading ladder 6752db0a; ruling 83a314a6 on what measures it).

SUB-LINE DISFLUENCY IS THE SPEECH OVERLAY, PROPOSED BY AGENTS AND REFINED BY THE
HUMAN — never a new stratum payload. A stratum is a run of WHOLE segments; the
sub-line record already exists: the annotate lane's speech overlay (DEC 4e05a066 —
a non-mutating annotation Correction with a text-indexed span anchor, FA-snapped
times, a snap trust stamp, an open label vocabulary). This module puts that record
behind the filtering lane's seam:

    span pack  ->  proposer  ->  span proposal set  ->  confirm (accept = overlay op)

* the PACK is a filter pack whose `row_kind` is `span`: the same numbered lines,
  an OVERLAY label slate with glosses (the four filterable labels plus the two
  keep labels, so the boundary is stated), and the overlays ALREADY active over
  the window rendered as do-not-re-propose lines (in-context examples of the
  human's own practice);
* PROPOSERS QUOTE, NEVER COUNT: a row = label + i (ONE line) + text (the verbatim
  words) + nth (1-based occurrence when the words repeat in the line) + tier +
  confidence + rationale; ingest resolves it to whole WORD TOKENS by normalized
  token-sequence match (case / punctuation-insensitive — the segment_word_tokens
  unit the annotate lane selects by) and records the anchor with the line's OWN
  substring as text_snapshot; a span that cannot be found, or is ambiguous
  without nth, is refused loudly with its row number; no character offset ever
  comes from a model;
* a SPAN SET IS ITS OWN FORMAT, so the filter lane's set loader never offers one
  for stratum accepts (a span set accepted as strata would classify whole lines
  under an overlay label);
* ACCEPT = the existing overlay commit plus proposal provenance; times are
  snapped AT ACCEPT from the forced-alignment words (ingest has no graph — its
  times are char-fraction estimates for ordering only); a line whose text
  drifted since the pack re-anchors through `reanchor_span` or refuses;
* VERDICTS DERIVE (nothing stored): proposal-id carry, else same-segment
  character overlap — same label with matching tokens = accepted, moved edges =
  edited, another label = relabeled, nothing below the lane's watermark =
  rejected;
* a LEXICON TIER needs no model: bare um / uh tokens are proposed by pattern as
  tier 1 under their own capability actor; `like` / `you know` / `I mean` stay
  agent-judged, as do false starts and repeats, where the keep labels are the
  judgment.
"""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cjm_transcript_correction_core.graph import reanchor_span
from cjm_transcript_correction_core.models import SpineSegment
from cjm_transcript_correction_core.strata import (_fmt_ts, _pack_rows, _render_pack_lines,
                                                   FILTER_PACK_FORMAT, load_filter_proposal_sets,
                                                   pack_digest, write_filter_propset)


def _spine():  # The pure word-token helpers live in spine.py, which imports cli at load — resolve late
    from cjm_transcript_correction_core import spine
    return spine


def _norm_token(t: str) -> str:  # Case/punctuation-insensitive comparison form (spine._norm_token)
    return _spine()._norm_token(t)


def segment_word_tokens(text: str) -> List[Tuple[int, int, str]]:  # spine.segment_word_tokens
    return _spine().segment_word_tokens(text)


SPAN_PACK_ROW_KIND = "span"
FILTER_PACK_VERSION_SPAN = "0.3.0"   # A filter pack whose rows are spans (row_kind = span)
SPAN_PROPOSAL_SET_FORMAT = "cjm-transcript-correction-core/span-proposal-set"
SPAN_PROPOSAL_SET_VERSION = "0.1.0"
SPAN_LANE = "annotate"   # The lane tag on the per-spine gate's watermark assertions
LEXICON_ACTOR = "capability:span-lexicon"   # The pattern tier's provenance (no model)
HESITATION_LEXICON = ("um", "uh", "umm", "uhh", "uhm", "erm")   # Bare tokens the lexicon tier proposes

# The human-grown overlay vocabulary as read on the data-center episode (386 active
# overlays, 2026-09-17): the boundary between what a reader FILTERS and what it must
# KEEP is stated in the slate itself — subtracting a kept label would change meaning.
FILTERABLE_OVERLAY_LABELS = ("hesitation-marker", "discourse-marker", "word-repeat", "false-start")
KEEP_OVERLAY_LABELS = ("emphasis-repeat", "coincidental-repeat")
OVERLAY_GLOSSES: Dict[str, str] = {
    "hesitation-marker": ("a filled pause — um / uh / er and the like; a bare token that carries "
                          "no content (the lexicon tier already proposes the bare um / uh — raise "
                          "the ones it cannot see: 'like', 'you know', 'I mean', 'kind of' used as "
                          "hesitation, NOT as content: 'things like tensors' is content)"),
    "discourse-marker": ("a spoken connective that structures the delivery but carries no content — "
                         "'you know', 'right', 'so', 'I mean', 'okay' as a turn-taking or pacing "
                         "device; a connective doing LOGICAL work ('so' = therefore, 'but') is "
                         "content and stays"),
    "word-repeat": ("a stutter or verbatim repeat where only ONE instance carries the meaning — "
                    "'the the', 'we, we'; quote the WHOLE repeated run, including the surviving "
                    "instance"),
    "false-start": ("words the speaker abandons and restarts — 'I think we— what we found was'; "
                    "quote ONLY the abandoned words, never the restart. An abandoned FRAGMENT is ONE "
                    "false-start ('Um, not like' before a fresh sentence), not its filler words one by "
                    "one. A CUED SELF-REPAIR ('…is known at compile time, uh sorry, at runtime…') has "
                    "two parts and BOTH are rows: the cue ('uh sorry,') AND the retracted words ('at "
                    "compile time,') — even when the retracted words read fluently or sit on an "
                    "EARLIER line; leaving them is the worst miss, because a reader then meets the "
                    "withdrawn claim with no sign it was withdrawn"),
    "emphasis-repeat": ("a KEEP label: a repeat that is rhetorical emphasis — 'much, much larger', "
                        "'no, no, no' — subtracting it would change meaning; propose it so the "
                        "boundary is explicit, it is never filtered"),
    "coincidental-repeat": ("a KEEP label: adjacent identical words that are both content — "
                            "'that that', 'had had' — never filtered; propose it only when a "
                            "reader might mistake it for a stutter"),
}


def overlay_label_slate(
    labels: Optional[Sequence[str]] = None,  # Labels to name (default: the filterable + keep slate)
) -> List[Dict[str, Any]]:  # [{"label", "gloss", "keep"}] in slate order
    """The slate a span pack carries: every label with its gloss and whether a
    reader KEEPS it (the boundary stated, never inferred from the name)."""
    names = list(labels) if labels else list(FILTERABLE_OVERLAY_LABELS) + list(KEEP_OVERLAY_LABELS)
    return [{"label": lab, "gloss": OVERLAY_GLOSSES.get(lab, ""),
             "keep": lab in KEEP_OVERLAY_LABELS} for lab in names]


def build_span_pack(
    source_id: str,                          # The Source node id
    title: str,                              # Display title (rendered; not identity)
    skeleton_hash: Optional[str],            # The spine the pack reads (None = legacy)
    segments: List[SpineSegment],            # The EFFECTIVE spine (corrections applied), index order
    *,
    content_hash: Optional[str] = None,      # Source media content hash (run-independent binding)
    window: Optional[Tuple[float, Optional[float]]] = None,  # (start, end) source seconds; None = whole spine
    overlays: Optional[List[Dict[str, Any]]] = None,  # ACTIVE speech-overlay correction dicts (rendered as do-not-re-propose)
    labels: Optional[Sequence[str]] = None,  # The label slate (default: the filterable + keep slate)
    speakers: Optional[Dict[str, Optional[str]]] = None,  # segment id -> speaker display name (None = no speakers)
    margin: int = 0,                         # Text segments of READ-ONLY context either side of the window
    context_only: Optional[Dict[str, str]] = None,  # segment id -> the accepted stratum class that ELIDES the line
) -> Dict[str, Any]:  # The pack (JSON-serializable)
    """Build the span proposer's input: the filter pack's numbered lines with
    `row_kind` = span, the overlay label slate (glossed, keep-flagged) and the
    overlays already active over the window. The same source binding, window
    and digest regime as a stratum pack — `filter-ingest` refuses it by
    row_kind, `span-ingest` requires it.

    `context_only` (finding ec370add, user-caught): the clean read drops the
    accepted filler lines FIRST and subtracts spans only inside the lines that
    remain, so the span pack follows the same order — a line an accepted
    line-level stratum already elides stays in the pack, NUMBERED and readable
    (a false start often runs through one), but is marked `context_only`:
    the brief says so, the lexicon skips it, `validate_span_rows` refuses a
    row on it. None = every line proposable (a dataset-purposed pass)."""
    rows, before, after, win = _pack_rows(segments, window=window, speakers=speakers, margin=margin)
    marked = sorted({context_only[r["id"]] for r in rows if r["id"] in (context_only or {})})
    for r in rows:
        if r["id"] in (context_only or {}):
            r["context_only"] = context_only[r["id"]]
    pos_by_id = {r["id"]: r["i"] for r in rows}
    existing: List[Dict[str, Any]] = []
    for c in (overlays or []):
        p = c.get("payload") or {}
        anchor = p.get("anchor") or {}
        i = pos_by_id.get(anchor.get("segment_id"))
        if i is None:
            continue
        existing.append({"overlay_id": c.get("id"), "i": i, "label": p.get("label"),
                         "text": str(p.get("text") or anchor.get("text_snapshot") or ""),
                         "char_start": anchor.get("char_start"), "char_end": anchor.get("char_end"),
                         "actor": c.get("actor")})
    existing.sort(key=lambda e: (e["i"], e["char_start"] if e["char_start"] is not None else -1))
    pack = {
        "format": FILTER_PACK_FORMAT,
        "version": FILTER_PACK_VERSION_SPAN,
        "row_kind": SPAN_PACK_ROW_KIND,
        "pack_id": f"pack_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        "created_at": time.time(),
        "source": {"source_id": source_id, "title": title,
                   "content_hash": content_hash, "skeleton_hash": skeleton_hash},
        "window": win,
        "overlay_vocabulary": overlay_label_slate(labels),
        "existing_overlays": existing,
        "segments": rows,
    }
    if margin and margin > 0:
        pack["context"] = {"before": before[-int(margin):], "after": after[:int(margin)]}
    if marked:
        pack["context_only_strata"] = marked
    pack["digest"] = pack_digest(pack)
    return pack


def is_span_pack(pack: Dict[str, Any]) -> bool:  # True when the pack's rows are spans
    return (pack.get("format") == FILTER_PACK_FORMAT
            and pack.get("row_kind") == SPAN_PACK_ROW_KIND)


SPAN_OUTPUT_CONTRACT = """\
## Output contract

Write ONE JSON object per line (JSONL). Each row proposes ONE overlay span inside ONE
pack line:

    {"label": "<label>", "i": <int>, "text": "<the words, verbatim from line i>",
     "nth": <int, only when those words occur more than once in the line>,
     "tier": 1|2, "confidence": <0.0-1.0>, "rationale": "<one sentence>"}

* `label`: ONLY a label from the slate above — this is a class-scoped pass; the KEEP
  labels exist so you can say "this repeat is content" explicitly.
* `i`: the pack line number (the `[i]` prefix). A span never crosses lines — a false
  start that runs over a line break is one row per line.
* `text`: the exact contiguous words as printed on that line (punctuation may be
  included or left off; matching is case- and punctuation-insensitive over WHOLE
  words). Quote the words — never count characters or positions.
* `nth`: when the same words occur more than once in the line, which occurrence
  (1-based, left to right). Omit it when the words occur once.
* `tier`: 1 = you are confident enough that a human batch-accept is reasonable;
  2 = borderline, audition only (dim in the walk, never batch-accepted).
* `confidence`: your own calibration, 0..1.

Do not re-propose a span listed under "Already annotated". Rows only — no prose before
or after, no code fences.
"""


def render_span_pack(pack: Dict[str, Any]) -> str:  # The proposer brief (markdown)
    """Render a span pack as the brief a proposer reads: identity + window, the
    label slate with glosses (keep labels marked), the overlays already active
    over the window, the span output contract, then the numbered lines."""
    src = pack.get("source") or {}
    win = pack.get("window") or {}
    end_txt = _fmt_ts(win["end"]) if win.get("end") is not None else "end"
    lines: List[str] = [
        f"# Speech-overlay pack `{pack.get('pack_id')}`",
        "",
        f"Source: **{src.get('title') or src.get('source_id')}**  "
        f"(`{src.get('source_id')}`; spine `{(src.get('skeleton_hash') or 'legacy')[-12:]}`)",
        f"Window: {_fmt_ts(win.get('start') or 0.0)} – {end_txt}  ·  "
        f"{len(pack.get('segments') or [])} text lines  ·  digest `{pack.get('digest', '')[-12:]}`",
        "",
        "## Task",
        "",
        "Read the numbered transcript lines below and propose SPEECH OVERLAYS: the sub-line",
        "runs of words a clean read would subtract (filled pauses, discourse markers, stutters,",
        "abandoned false starts) — and, with the KEEP labels, the repeats that are content. Quote",
        "the exact words of ONE line per row. Everything you do not mark stays as spoken. Prefer",
        "precise spans over generous ones; when a run is content, leave it. A line that is",
        "ENTIRELY filler ('Um', 'Okay.') is not yours — the filler stratum pass takes whole lines;",
        "propose only spans INSIDE lines that also carry content.",
    ]
    if pack.get("context_only_strata"):
        lines += [
            "",
            "Lines marked `(▣" + " / ▣".join(pack["context_only_strata"]) + " · context only)` were already",
            "classified by the human as WHOLLY elidable: every later reader drops them. They stay here",
            "so you can follow a thought that runs through one, but a row on such a line is REFUSED —",
            "never propose over them.",
        ]
    roster = list(dict.fromkeys(r["speaker"] for r in pack.get("segments") or []
                                if r.get("speaker")))
    if roster:
        lines += ["", "Speakers (from the human's assignment pass): " + " · ".join(roster)]
    lines += ["", "## Labels", ""]
    for v in pack.get("overlay_vocabulary") or []:
        keep = "  **(KEEP — never filtered)**" if v.get("keep") else ""
        lines.append(f"- `{v['label']}`{keep} — {v.get('gloss') or ''}".rstrip(" —"))
    existing = pack.get("existing_overlays") or []
    lines += ["", "## Already annotated over this window", ""]
    if existing:
        for e in existing:
            lines.append(f"- line {e['i']}: “{e['text']}” — `{e['label']}` "
                         f"(by {e.get('actor') or '?'}) — do not re-propose")
    else:
        lines.append("- (none)")
    lines += ["", SPAN_OUTPUT_CONTRACT, "## Transcript", ""]
    ctx = pack.get("context") or {}
    if ctx.get("before"):
        lines += ["### Context BEFORE the window — read-only, never propose over `(ctx)` lines", ""]
        lines += _render_pack_lines(ctx["before"], numbered=False)
        lines += ["", "### The window — propose over these numbered lines", ""]
    lines += _render_pack_lines(pack.get("segments") or [], numbered=True)
    if ctx.get("after"):
        lines += ["", "### Context AFTER the window — read-only, never propose over `(ctx)` lines", ""]
        lines += _render_pack_lines(ctx["after"], numbered=False)
    return "\n".join(lines) + "\n"


def find_token_span(
    tokens: List[Tuple[int, int, str]],  # segment_word_tokens output for the line
    text: str,                           # The quoted words (verbatim-ish; whitespace-split)
    nth: Optional[int] = None,           # 1-based occurrence when the words repeat in the line
) -> Tuple[int, int]:  # (token index a, token index b) inclusive
    """Resolve quoted words to WHOLE word tokens of a line (pure): the quote is
    tokenized and normalized exactly like the line (case / punctuation-
    insensitive), every position where the line's normalized token sequence
    equals the quote's is a candidate; one candidate resolves, several need
    `nth`, none refuses. Raises ValueError with the reason — the caller
    prefixes the row number."""
    q = [_norm_token(t) for t in (text or "").split()]
    q = [t for t in q if t]
    if not q:
        raise ValueError(f"text {text!r} has no matchable words")
    norm = [_norm_token(t) for _, _, t in tokens]
    k = len(q)
    hits = [i for i in range(0, len(norm) - k + 1) if norm[i:i + k] == q]
    if not hits:
        raise ValueError(f"text {text!r} not found as whole words on the line")
    if nth is not None:
        if not (1 <= int(nth) <= len(hits)):
            raise ValueError(f"nth={nth} but text {text!r} occurs {len(hits)} time(s) on the line")
        a = hits[int(nth) - 1]
    elif len(hits) == 1:
        a = hits[0]
    else:
        raise ValueError(f"text {text!r} occurs {len(hits)} times on the line — give nth (1..{len(hits)})")
    return a, a + k - 1


def validate_span_rows(
    rows: List[Dict[str, Any]],  # Raw proposer output rows (parsed JSONL)
    pack: Dict[str, Any],        # The span pack the rows reference
) -> List[Dict[str, Any]]:  # Normalized + RESOLVED rows (label/i/text/nth/tier/confidence/rationale + token/char range)
    """Validate + resolve proposer rows against their span pack — loud on the
    first bad row (row number in the message). Enforces the contract: a label
    from the pack's slate, `i` in range, the quoted words found as whole tokens
    on that line (nth disambiguates), tier in {1, 2}, confidence in [0, 1], no
    duplicate (same label, same tokens). Overlaps between DIFFERENT labels are
    allowed — the human's practice nests a hesitation inside a repeat."""
    if not is_span_pack(pack):
        raise ValueError("not a span pack (row_kind != span) — filter-ingest reads stratum packs")
    segs = pack.get("segments") or []
    slate = {v["label"] for v in (pack.get("overlay_vocabulary") or [])}
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for k, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"row {k}: not an object")
        lab = str(raw.get("label") or raw.get("category") or "").strip()
        if lab not in slate:
            raise ValueError(f"row {k}: label {lab!r} is not in the pack's slate "
                             f"({', '.join(sorted(slate))})")
        try:
            i = int(raw.get("i"))
        except (TypeError, ValueError):
            raise ValueError(f"row {k}: i must be an integer pack line number")
        if not (0 <= i < len(segs)):
            raise ValueError(f"row {k}: line {i} outside the pack (0..{len(segs) - 1})")
        if segs[i].get("context_only"):
            raise ValueError(f"row {k}: line {i} is an accepted {segs[i]['context_only']} line "
                             "(context only) — the clean read drops it whole; propose spans only "
                             "on the lines it keeps")
        text = str(raw.get("text") or "").strip()
        nth = raw.get("nth")
        if nth is not None:
            try:
                nth = int(nth)
            except (TypeError, ValueError):
                raise ValueError(f"row {k}: nth must be an integer")
        line_text = str(segs[i].get("text") or "")
        tokens = segment_word_tokens(line_text)
        try:
            a, b = find_token_span(tokens, text, nth)
        except ValueError as e:
            raise ValueError(f"row {k}: line {i}: {e}")
        tier = int(raw.get("tier", 1) or 1)
        if tier not in (1, 2):
            raise ValueError(f"row {k}: tier must be 1 or 2, got {tier}")
        conf = raw.get("confidence")
        if conf is not None:
            conf = float(conf)
            if not (0.0 <= conf <= 1.0):
                raise ValueError(f"row {k}: confidence must be within [0, 1], got {conf}")
        key = (lab, i, a, b)
        if key in seen:
            raise ValueError(f"row {k}: duplicate {lab} span on line {i} ({text!r})")
        seen.add(key)
        cs, ce = tokens[a][0], tokens[b][1]
        out.append({"label": lab, "i": i, "text": text, "nth": nth, "tier": tier,
                    "confidence": conf,
                    "rationale": str(raw.get("rationale") or "").strip() or None,
                    "tok_a": a, "tok_b": b, "char_start": cs, "char_end": ce,
                    "text_snapshot": line_text[cs:ce]})
    return out


def span_proposals_from_rows(
    rows: List[Dict[str, Any]],  # validate_span_rows output
    pack: Dict[str, Any],        # The span pack the rows reference
) -> List[Dict[str, Any]]:  # Span proposal-set rows (time-ordered)
    """Resolve validated rows to span proposal-set rows: proposal id, label
    (also `category` for the propset walkers' generic key), the SPAN ANCHOR
    (segment id + char range + the line's own substring as text_snapshot),
    char-fraction time ESTIMATES (ordering only — accept snaps from the FA
    words; `snap` says so), tier, confidence, rationale, and the evidence
    read-trace (pack id + line + the quote + nth)."""
    segs = pack.get("segments") or []
    out: List[Dict[str, Any]] = []
    for r in rows:
        line = segs[r["i"]]
        st, en = line.get("start"), line.get("end")
        n = max(1, len(str(line.get("text") or "")))
        if st is not None and en is not None and float(en) > float(st):
            dur = float(en) - float(st)
            ps = float(st) + dur * (r["char_start"] / n)
            pe = float(st) + dur * (r["char_end"] / n)
            ps, pe = round(ps, 4), round(max(pe, ps + 0.01), 4)
        else:
            ps, pe = (float(st) if st is not None else None), (float(en) if en is not None else None)
        out.append({
            "proposal_id": str(uuid.uuid4()),
            "label": r["label"],
            "category": r["label"],          # the propset walkers' generic key
            "segment_id": line["id"],
            "index": line["index"],
            "anchor": {"kind": "span", "segment_id": line["id"],
                       "char_start": r["char_start"], "char_end": r["char_end"],
                       "text_snapshot": r["text_snapshot"]},
            "start_time": ps,
            "end_time": pe,
            "snap": "estimated",
            "tier": r["tier"],
            "confidence": r.get("confidence"),
            "score": r.get("confidence"),   # the propset walkers' generic key
            "rationale": r.get("rationale"),
            "evidence": {"pack_id": pack.get("pack_id"), "i": r["i"], "text": r["text"],
                         **({"nth": r["nth"]} if r.get("nth") is not None else {})},
        })
    out.sort(key=lambda p: (p["start_time"] if p["start_time"] is not None else 0.0,
                            p["evidence"]["i"], p["anchor"]["char_start"]))
    return out


def lexicon_span_rows(
    pack: Dict[str, Any],                       # The span pack
    lexicon: Sequence[str] = HESITATION_LEXICON,  # Bare tokens proposed as hesitation-marker
) -> List[Dict[str, Any]]:  # Raw proposer rows (tier 1) — feed them to validate_span_rows like any proposer's
    """The LEXICON TIER (design bbf8bafd (g)): a bare um / uh token is a
    hesitation marker with no judgment needed — one tier-1 row per token, nth
    counted among the line's occurrences of that token, provenance the
    capability actor. A line that is NOTHING but such tokens ('Um', 'Uh, um.')
    is skipped: a wholly elidable line is the filler stratum's class, and an
    overlay over a whole line is the limit case the design keeps separate.
    `like` / `you know` / `I mean` are NOT here: they need the agent's reading
    ('things like tensors' is content)."""
    lex = {_norm_token(t) for t in lexicon}
    out: List[Dict[str, Any]] = []
    for line in pack.get("segments") or []:
        if line.get("context_only"):
            continue   # an accepted filler line: elided whole by stratum, never an overlay's (ec370add)
        toks = segment_word_tokens(str(line.get("text") or ""))
        if toks and all(_norm_token(t) in lex for _a, _b, t in toks):
            continue   # a WHOLLY elidable line is the filler stratum's (design bbf8bafd (a)), not an overlay
        counts: Dict[str, int] = {}
        for _cs, _ce, tok in toks:
            n = _norm_token(tok)
            if n not in lex:
                continue
            counts[n] = counts.get(n, 0) + 1
            out.append({"label": "hesitation-marker", "i": line["i"], "text": tok,
                        "nth": counts[n], "tier": 1, "confidence": 1.0,
                        "rationale": f"lexicon: bare '{n}'"})
    return out


def write_span_propset(
    pack: Dict[str, Any],                 # The span pack the proposals came from
    proposals: List[Dict[str, Any]],      # span_proposals_from_rows output
    *,
    out_root: Path,                       # Proposal-set root (<workspace>/proposals)
    proposer: Dict[str, Any],             # Provenance: {"kind": ..., "name": ..., ...}
    ws: Any = None,                       # Resolved workspace (relativize_recorded) or None
    extra: Optional[Dict[str, Any]] = None,  # Additive manifest fields
) -> Dict[str, Any]:  # {"set_id","set_dir","manifest_path","classes","counts","tier2_counts"}
    """Write one SPAN proposal set — the filter set's layout under the span
    format string (design bbf8bafd (d)): the stratum lane's loader never lists
    it, the span lane's loader lists only it."""
    return write_filter_propset(
        pack, proposals, out_root=out_root, proposer=proposer, ws=ws, extra=extra,
        set_format=SPAN_PROPOSAL_SET_FORMAT, set_version=SPAN_PROPOSAL_SET_VERSION,
        lane=SPAN_LANE,
        vocabulary=[v["label"] for v in (pack.get("overlay_vocabulary") or [])])


def load_span_proposal_sets(
    ws_root: str,                         # Workspace root (sets live under <root>/proposals/)
    source_id: str,                       # Source node id to match
    skeleton_hash: Optional[str] = None,  # Restrict to sets bound to this spine (None = any)
) -> List[Dict[str, Any]]:  # [{"manifest","path","proposals"}] newest first
    """Every SPAN proposal set for a source (and optionally one spine), newest
    first — never a stratum set (format-gated)."""
    return load_filter_proposal_sets(ws_root, source_id, skeleton_hash=skeleton_hash,
                                     set_format=SPAN_PROPOSAL_SET_FORMAT)


def merge_span_proposals(
    sets: List[Dict[str, Any]],  # load_span_proposal_sets-shaped entries ({"manifest","proposals"}), any windows
    pack: Dict[str, Any],        # The TARGET span pack every row re-resolves against (normally the whole spine)
    *,
    iou: float = 1.0,            # Same-label, same-line character IoU at/above which two rows are ONE (1.0 = the same words)
) -> List[Dict[str, Any]]:  # Span proposal-set rows in target-pack coordinates, each carrying `origins`
    """Fold several SPAN sets over one spine into one walkable set — the
    merge_filter_proposals dual (design 6752db0a (7) + (8)): N window sets
    become one set, rival arms one walk. Each row re-resolves by SEGMENT ID
    into the target pack's line (loud when the line is not there, or its
    text differs from the anchor's snapshot at the recorded offsets — the
    spine drifted or the pack is the wrong one); same-label rows on the same
    line whose character ranges agree (IoU >= `iou`; the default 1.0 = the
    same words, since a span IS its words) collapse into one row whose
    `origins` name every contributor. Representative = the most-agreed
    range, ties to the highest confidence; tier = the most visible member.
    Overlapping-but-different rows stay separate: the human's accept decides,
    and each origin set still benches on its own by character overlap."""
    lines = {r["id"]: r for r in pack.get("segments") or []}
    flat: List[Dict[str, Any]] = []
    for entry in sets:
        m = entry.get("manifest") or {}
        if (m.get("source") or {}).get("source_id") != (pack.get("source") or {}).get("source_id"):
            raise ValueError(f"set {m.get('proposal_set_id')}: a different source than the target pack")
        for p in entry.get("proposals") or []:
            a = p.get("anchor") or {}
            line = lines.get(a.get("segment_id"))
            if line is None:
                raise ValueError(f"set {m.get('proposal_set_id')} proposal {p.get('proposal_id')}: "
                                 f"segment {a.get('segment_id')} is not in the target pack")
            cs, ce = int(a.get("char_start") or 0), int(a.get("char_end") or 0)
            if str(line.get("text") or "")[cs:ce] != str(a.get("text_snapshot") or ""):
                raise ValueError(f"set {m.get('proposal_set_id')} proposal {p.get('proposal_id')}: "
                                 f"line {line['i']} reads differently than the anchor's snapshot "
                                 f"({a.get('text_snapshot')!r} at {cs}:{ce})")
            flat.append({"p": p, "i": line["i"], "range": (cs, ce),
                         "origin": {"set_id": m.get("proposal_set_id"),
                                    "proposal_id": p.get("proposal_id"),
                                    "proposer": (m.get("model") or {}).get("name"),
                                    "window": m.get("window"),
                                    "i": line["i"], "char_start": cs, "char_end": ce,
                                    "tier": int(p.get("tier", 1)),
                                    "confidence": p.get("confidence")}})
    flat.sort(key=lambda f: (str(f["p"].get("label")), f["i"], f["range"]))
    clusters: List[List[Dict[str, Any]]] = []
    for f in flat:
        home = next((c for c in reversed(clusters)
                     if c[0]["p"].get("label") == f["p"].get("label") and c[0]["i"] == f["i"]
                     and _char_iou(*c[0]["range"], *f["range"]) >= float(iou)
                     and f["origin"]["set_id"] not in {x["origin"]["set_id"] for x in c}), None)
        if home is None:
            clusters.append([f])
        else:
            home.append(f)
    out: List[Dict[str, Any]] = []
    for c in clusters:
        votes: Dict[Tuple[int, int], int] = {}
        for f in c:
            votes[f["range"]] = votes.get(f["range"], 0) + 1
        rep = max(c, key=lambda f: (votes[f["range"]], float(f["p"].get("confidence") or 0.0)))
        p = rep["p"]
        row = validate_span_rows([{"label": p.get("label"), "i": rep["i"],
                                   "text": (p.get("anchor") or {}).get("text_snapshot"),
                                   "nth": _nth_of(lines_text=str((pack.get("segments") or [])[rep["i"]].get("text") or ""),
                                                  text=str((p.get("anchor") or {}).get("text_snapshot") or ""),
                                                  char_start=rep["range"][0]),
                                   "tier": min(f["origin"]["tier"] for f in c),
                                   "confidence": p.get("confidence"),
                                   "rationale": p.get("rationale")}], pack)
        merged = span_proposals_from_rows(row, pack)[0]
        merged["origins"] = [f["origin"] for f in c]
        out.append(merged)
    out.sort(key=lambda r: (r["start_time"] if r["start_time"] is not None else 0.0,
                            r["evidence"]["i"], r["anchor"]["char_start"]))
    return out


def _nth_of(lines_text: str, text: str, char_start: int) -> Optional[int]:  # Which occurrence starts at char_start
    """The 1-based occurrence of `text` (as whole tokens) whose first token
    starts at `char_start` — so a merged row re-states the SAME words a
    proposer quoted, disambiguated the way the contract disambiguates."""
    tokens = segment_word_tokens(lines_text)
    q = [_norm_token(t) for t in text.split() if _norm_token(t)]
    norm = [_norm_token(t) for _, _, t in tokens]
    k = len(q)
    hits = [i for i in range(0, len(norm) - k + 1) if norm[i:i + k] == q]
    for n, i in enumerate(hits, start=1):
        if tokens[i][0] == char_start:
            return n if len(hits) > 1 else None
    return None


def render_span_propset_markdown(
    manifest: Dict[str, Any],                 # The span set's manifest
    proposals: List[Dict[str, Any]],          # Its rows
    pack: Optional[Dict[str, Any]] = None,    # The pack the rows reference (None = quote-only)
) -> str:  # A human-readable projection of the set (markdown)
    """Project a span set for a human: one line per proposal — tier, label,
    the pack line with the span marked, spine index, time estimate."""
    src = manifest.get("source") or {}
    model = manifest.get("model") or {}
    win = manifest.get("window") or {}
    segs = (pack or {}).get("segments") or []
    tag = (src.get("skeleton_hash") or "legacy").split(":")[-1][:8]
    end_txt = _fmt_ts(win["end"]) if win.get("end") is not None else "end"
    counts = manifest.get("counts") or {}
    t2 = manifest.get("tier2_counts") or {}
    lines: List[str] = [
        f"# Speech-overlay proposals — {src.get('title') or src.get('source_id')}",
        "",
        f"Set `{manifest.get('proposal_set_id')}` · spine `{tag}` · window "
        f"{_fmt_ts(win.get('start') or 0.0)}–{end_txt} · proposer "
        f"{model.get('kind') or '?'}:{model.get('name') or '?'}"
        + (f" ({model.get('model')})" if model.get("model") else ""),
        f"Tier 1: {' · '.join(f'{k}×{v}' for k, v in sorted(counts.items())) or 'none'}  ·  "
        f"Tier 2: {' · '.join(f'{k}×{v}' for k, v in sorted(t2.items())) or 'none'}",
        "",
        "`?` = tier 1 (batch-acceptable), `??` = tier 2 (audition). Times are char-fraction "
        "estimates until accept snaps them to the forced alignment.",
        "",
    ]
    rows = sorted(proposals or [], key=lambda p: (float(p.get("start_time") or 0.0),
                                                  int((p.get("evidence") or {}).get("i") or 0),
                                                  int((p.get("anchor") or {}).get("char_start") or 0)))
    for n, p in enumerate(rows, start=1):
        ev = p.get("evidence") or {}
        a = p.get("anchor") or {}
        i = ev.get("i")
        line = segs[i] if (segs and i is not None and 0 <= i < len(segs)) else None
        tier = "??" if int(p.get("tier", 1)) == 2 else "?"
        conf = p.get("confidence")
        st = _fmt_ts(p["start_time"]) if p.get("start_time") is not None else "--:--"
        where = f"spine {line['index']}" if line else f"pack line {i}"
        lines.append(f"## {n}. `{tier}` **{p.get('label')}** · {st} · {where}"
                     + (f" · c={conf:.2f}" if isinstance(conf, (int, float)) else "")
                     + f" · id `…{str(p.get('proposal_id') or '')[-8:]}`")
        lines.append("")
        if line:
            t = str(line.get("text") or "")
            cs, ce = int(a.get("char_start") or 0), int(a.get("char_end") or 0)
            lines.append(f"> [{i}] {t[:cs]}**⟦{t[cs:ce]}⟧**{t[ce:]}")
        else:
            lines.append(f"> “{a.get('text_snapshot') or ev.get('text')}”")
        if p.get("rationale"):
            lines.append(f"> — {p['rationale']}")
        lines.append("")
    if not rows:
        lines.append("_(no proposals in this set)_")
        lines.append("")
    return "\n".join(lines)


def _char_overlap(a0: int, a1: int, b0: int, b1: int) -> int:  # Characters of overlap
    return max(0, min(a1, b1) - max(a0, b0))


def _char_iou(a0: int, a1: int, b0: int, b1: int) -> float:  # Character IoU, 0 on degenerate
    inter = _char_overlap(a0, a1, b0, b1)
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def _same_words(a: str, b: str) -> bool:  # Token-equal after normalization
    return ([_norm_token(t) for t in (a or "").split() if _norm_token(t)]
            == [_norm_token(t) for t in (b or "").split() if _norm_token(t)])


def _overlay_record(c: Dict[str, Any]) -> Optional[Dict[str, Any]]:  # A flat view of an active overlay
    p = c.get("payload") or {}
    a = p.get("anchor") or {}
    try:
        cs, ce = int(a.get("char_start")), int(a.get("char_end"))
    except (TypeError, ValueError):
        return None
    return {"overlay_id": c.get("id"), "label": p.get("label"),
            "segment_id": a.get("segment_id"), "char_start": cs, "char_end": ce,
            "text": str(p.get("text") or a.get("text_snapshot") or ""),
            "start": float(p.get("start_time") or 0.0), "end": float(p.get("end_time") or 0.0),
            "proposal_id": p.get("proposal_id"), "actor": c.get("actor")}


def pending_span_proposals(
    proposals: List[Dict[str, Any]],  # A span set's rows
    overlays: List[Dict[str, Any]],   # active_speech_overlays output
    show_tier2: bool = False,         # Include the audition tier
) -> List[Dict[str, Any]]:  # Proposals not yet materialized, time order
    """The headless worklist: proposals with NO active overlay carrying their id
    and NO same-label overlay overlapping them on the same segment (the human
    already annotated it, by hand or from another set). Tier 2 hides by
    default (dual-tier doctrine a475ccd6)."""
    recs = [r for r in (_overlay_record(c) for c in overlays) if r]
    by_pid = {r["proposal_id"] for r in recs if r["proposal_id"]}
    by_seg: Dict[Any, List[Dict[str, Any]]] = {}   # the qt lane re-derives per gesture over hundreds of rows
    for r in recs:
        by_seg.setdefault(r["segment_id"], []).append(r)
    out: List[Dict[str, Any]] = []
    for p in sorted(proposals, key=lambda d: (float(d.get("start_time") or 0.0),
                                              int((d.get("anchor") or {}).get("char_start") or 0))):
        if not show_tier2 and int(p.get("tier", 1)) == 2:
            continue
        if p.get("proposal_id") in by_pid:
            continue
        a = p.get("anchor") or {}
        if any(r["label"] == p.get("label")
               and _char_overlap(int(a.get("char_start") or 0), int(a.get("char_end") or 0),
                                 r["char_start"], r["char_end"]) > 0
               for r in by_seg.get(a.get("segment_id"), ())):
            continue
        out.append(p)
    return out


def bench_span_proposals(
    proposals: List[Dict[str, Any]],       # A span set's rows
    overlays: List[Dict[str, Any]],        # active_speech_overlays output (the live final state)
    window: Tuple[float, Optional[float]], # (start, end) the proposals covered; end None = unbounded
    *,
    watermark: Optional[float] = None,     # The lane's annotated_through (None = nothing visited)
) -> Dict[str, Any]:  # {"counts", "rates", "verdicts", "missed"}
    """Derive the span verdicts (design bbf8bafd (f), the bench_filter_proposals
    sibling) — pure, nothing stored.

    Per proposal: an active overlay carrying its proposal id, else the best
    same-segment character-overlap overlay, decides — same label with the SAME
    WORDS = ACCEPTED; same label, edges moved = EDITED; another label =
    RELABELED. No match below the watermark = REJECTED; above it = UNVISITED.
    Tier-2 rows read 'unaccepted' instead of rejected. MISSED = active overlays
    inside the window no proposal matched. Rates are the tier-1 operating-
    point contract.

    COVERED (finding 67c4af17): an unmatched proposal that a same-label
    overlay overlaps on its segment — the exact rule that closes it in
    `pending_span_proposals`, so the worklist never offered it (a nested row:
    the lexicon's 'uh' beside an agent's 'uh like', one overlay lands, it
    matches ONE proposal). Neither accepted nor rejected: it carries the
    covering overlay and stays OUT of the rates."""
    w0 = float(window[0])
    w1 = float(window[1]) if window[1] is not None else None
    recs = [r for r in (_overlay_record(c) for c in overlays) if r]
    live = [r for r in recs if r["start"] >= w0 and (w1 is None or r["start"] < w1)]
    cover_by_seg: Dict[Any, List[Dict[str, Any]]] = {}   # every active overlay, as pending sees them
    for r in recs:
        cover_by_seg.setdefault(r["segment_id"], []).append(r)
    unmatched = {id(r): r for r in live}
    rows = sorted(proposals or [], key=lambda d: float(d.get("start_time") or 0.0))
    tier1 = [p for p in rows if int(p.get("tier", 1)) == 1]
    tier2 = [p for p in rows if int(p.get("tier", 1)) == 2]

    def _span(p: Dict[str, Any]) -> Tuple[Optional[str], int, int]:
        a = p.get("anchor") or {}
        return a.get("segment_id"), int(a.get("char_start") or 0), int(a.get("char_end") or 0)

    def join(ordered: List[Dict[str, Any]], no_match: str) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        matches: Dict[int, Dict[str, Any]] = {}
        carried: Dict[Any, List[int]] = {}   # proposal id -> unmatched keys, overlay order
        seg_keys: Dict[Any, List[int]] = {}  # segment id -> unmatched keys, overlay order
        for key, r in unmatched.items():
            if r["proposal_id"]:
                carried.setdefault(r["proposal_id"], []).append(key)
            seg_keys.setdefault(r["segment_id"], []).append(key)
        for p in ordered:   # pass 0: exact proposal-id carry (the accept gesture records it)
            key = next((k for k in carried.get(p.get("proposal_id"), ()) if k in unmatched), None)
            if key is not None:
                matches[id(p)] = unmatched.pop(key)
        for same in (True, False):   # pass 1: same label best char IoU; pass 2: any label
            pairs: List[Tuple[float, int, int]] = []
            for p in ordered:
                if id(p) in matches:
                    continue
                sid, cs, ce = _span(p)
                for key in seg_keys.get(sid, ()):
                    r = unmatched.get(key)
                    if r is None or (same and r["label"] != p.get("label")):
                        continue
                    iou = _char_iou(cs, ce, r["char_start"], r["char_end"])
                    if iou > 0.0:
                        pairs.append((iou, id(p), key))
            for _v, pid, key in sorted(pairs, key=lambda t: -t[0]):
                if pid in matches or key not in unmatched:
                    continue
                matches[pid] = unmatched.pop(key)
        counts = {"accepted": 0, "edited": 0, "relabeled": 0, "covered": 0, no_match: 0,
                  "unvisited": 0}
        verdicts: List[Dict[str, Any]] = []
        for p in ordered:
            ps = float(p.get("start_time") or 0.0)
            m = matches.get(id(p))
            a = p.get("anchor") or {}
            cover = None
            if m is None:
                _sid, cs, ce = _span(p)
                cover = next((r for r in cover_by_seg.get(a.get("segment_id"), ())
                              if r["label"] == p.get("label")
                              and _char_overlap(cs, ce, r["char_start"], r["char_end"]) > 0), None)
            if cover is not None:
                verdict = "covered"
            elif m is None:
                verdict = (no_match if (watermark is not None and ps < float(watermark))
                           else "unvisited")
            elif m["label"] != p.get("label"):
                verdict = "relabeled"
            elif _same_words(m["text"], a.get("text_snapshot") or ""):
                verdict = "accepted"
            else:
                verdict = "edited"
            counts[verdict] += 1
            verdicts.append({"proposal_id": p.get("proposal_id"), "label": p.get("label"),
                             "segment_id": a.get("segment_id"), "text": a.get("text_snapshot"),
                             "start_time": ps, "tier": int(p.get("tier", 1)),
                             "confidence": p.get("confidence"), "verdict": verdict,
                             **({"overlay_id": m["overlay_id"], "overlay_label": m["label"],
                                 "overlay_text": m["text"]} if m else {}),
                             **({"covered_by": cover["overlay_id"], "covered_by_text": cover["text"],
                                 "covered_by_proposal": cover["proposal_id"]} if cover else {})})
        return counts, verdicts

    c1, v1 = join(tier1, "rejected")
    c2, v2 = join(tier2, "unaccepted")
    decided = c1["accepted"] + c1["edited"] + c1["relabeled"] + c1["rejected"]
    rates = ({k: round(c1[k] / decided, 3) for k in ("accepted", "edited", "relabeled", "rejected")}
             if decided else {})
    missed = sorted(unmatched.values(), key=lambda r: (r["start"], r["char_start"]))
    return {"counts": {"tier1": c1, "tier2": c2}, "rates": rates,
            "verdicts": v1 + v2, "missed": missed}


def locate_span_tokens(
    proposal: Dict[str, Any],  # A span set row (anchor + text_snapshot)
    text: str,                 # The segment's CURRENT effective text
    where: str = "the line",   # How refusals name the line (the snap passes "#<index>")
) -> Tuple[int, int, int, int]:  # (char_start, char_end, first_token, last_token) on the current line
    """Where a proposal's words sit on the CURRENT line, as a character range
    AND a whole-token range — the one resolution the accept snap and the qt
    lane's ARM step share (DEC d52d105f: a jump pre-loads the word selection
    with the proposal's tokens, so the hand gestures refine it). The anchor
    re-locates through `reanchor_span` (the snapshot is the truth, the offsets
    its hint). Raises ValueError when the words are gone from the line or the
    range no longer sits on token edges."""
    a = dict(proposal.get("anchor") or {})
    located = reanchor_span(a, text or "")
    if located is None:
        raise ValueError(f"words {a.get('text_snapshot')!r} are no longer on {where} ({text!r})")
    cs, ce = located
    tokens = segment_word_tokens(text or "")
    starts = [i for i, t in enumerate(tokens) if t[0] == cs]
    ends = [i for i, t in enumerate(tokens) if t[1] == ce]
    if not starts or not ends or ends[0] < starts[0]:
        raise ValueError(f"span {cs}:{ce} on {where} no longer sits on word edges")
    return cs, ce, starts[0], ends[0]


def snap_span_proposal(
    proposal: Dict[str, Any],                  # A span set row (anchor + text_snapshot)
    segment: SpineSegment,                     # The segment's CURRENT effective view
    fa_words: Optional[List[Dict[str, Any]]],  # Transcript FA words [{"s","e","text"}]; None = no cache
    label: Optional[str] = None,               # Override label (relabel at accept)
) -> Dict[str, Any]:  # {"anchor","label","start_time","end_time","text","words","snap"} — the overlay commit's kwargs
    """Resolve a proposal against the CURRENT line at accept (design bbf8bafd
    (e)): the anchor re-locates through `reanchor_span` (the snapshot is the
    truth, the offsets its hint), the char range maps back to whole word
    tokens, and the times SNAP from the FA words through `snap_word_span` —
    ingest's estimates never reach the graph. Raises ValueError when the
    words are gone from the line or the range no longer sits on token edges."""
    a = dict(proposal.get("anchor") or {})
    if a.get("segment_id") != segment.id:
        raise ValueError(f"proposal anchors segment {a.get('segment_id')}, not {segment.id}")
    text = segment.text or ""
    cs, ce, first, last = locate_span_tokens(proposal, text, where=f"#{segment.index}")
    tokens = segment_word_tokens(text)
    if segment.start_time is None or segment.end_time is None:
        raise ValueError(f"#{segment.index} has no audio times to snap against")
    snapped = _spine().snap_word_span(tokens, first, last, float(segment.start_time),
                             float(segment.end_time), len(text), fa_words)
    if snapped is None:
        raise ValueError(f"span on #{segment.index} refused by the snap (word range invalid)")
    start_s, end_s, snap, words = snapped
    return {"anchor": {"kind": "span", "segment_id": segment.id, "char_start": cs, "char_end": ce,
                       "text_snapshot": text[cs:ce]},
            "label": label or str(proposal.get("label")),
            "start_time": start_s, "end_time": end_s, "text": text[cs:ce],
            "words": words, "snap": snap}
