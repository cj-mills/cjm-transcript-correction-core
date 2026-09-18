"""The clean read — L1 of the reading ladder (design 6752db0a; the subtraction
shape from bbf8bafd (a)).

L0 is the spine as spoken. L1 is the same spine MINUS what a reader never
needs, as a MECHANICAL projection with VISIBLE elisions — it subtracts twice:

1. FILLER LINES first: every text segment under an excluded stratum (default
   `filler` — a wholly elidable line or run; the notes types' exclude_strata
   name more) drops as a line;
2. then the FILTERABLE OVERLAY SPANS inside the lines that remain — the
   annotate lane's speech overlays whose label a reader filters (hesitation-
   marker / discourse-marker / word-repeat / false-start); the KEEP labels
   (emphasis-repeat / coincidental-repeat) never subtract. A line emptied by
   its overlays drops too.

The projection is pure over the effective spine + the active strata + the
active overlays; nothing is stored. Every elision is visible: a subtracted
span leaves `marker` in the line (default `[…]`; "" for a clean-verbatim
consumer), a dropped line is recorded in the gap before the next kept line,
and each kept line carries what was cut. A wrong exclusion must never HIDE
content silently (ruling 6752db0a (9)) — the reader can always see that
something was elided and audit it against L0.

WORD-REPEAT KEEPS ITS LAST UNIT: the overlay quotes the whole run ('the the',
'we, we', 'and and and') because the run is the detector sample; the clean
read keeps the speaker's final instance — the span's tokens are split into
the shortest repeating unit and only the last unit survives; a run that is
not a clean repetition keeps its last token (a mechanical rule, auditable
through the marker; refine per label as drives show).
"""

import re
from dataclasses import replace
from typing import Any, Dict, List, Sequence, Tuple

from cjm_transcript_correction_core.graph import reanchor_span
from cjm_transcript_correction_core.models import SpineSegment
from cjm_transcript_correction_core.strata import strata_index

CLEAN_READ_EXCLUDE_STRATA = ("filler",)   # The default line-level subtraction (the notes types name more)
CLEAN_READ_FILTER_LABELS = ("hesitation-marker", "discourse-marker", "word-repeat", "false-start")
CLEAN_READ_KEEP_LAST_UNIT = ("word-repeat",)   # Labels whose span keeps its final repeated unit
ELISION_MARKER = "[…]"


def _norm(t: str) -> str:  # Case/punctuation-insensitive token form (the span lane's rule)
    return "".join(ch for ch in t.lower() if ch.isalnum())


def repeat_survivor(
    span_text: str,  # The overlay's words (a whole repeated run)
) -> Tuple[int, int]:  # (char offset, char end) INSIDE span_text of the surviving last unit
    """Locate the last repeating unit of a word-repeat span (pure): the
    shortest unit u with the span = u × n (normalized) wins; no clean
    repetition = the last token. Returned offsets are relative to the span."""
    toks = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", span_text or "")]
    if not toks:
        return 0, len(span_text or "")
    norm = [_norm(t) for _, _, t in toks]
    k = len(norm)
    for unit in range(1, k // 2 + 1):
        if k % unit == 0 and all(norm[i:i + unit] == norm[:unit] for i in range(0, k, unit)):
            return toks[k - unit][0], toks[-1][1]
    return toks[-1][0], toks[-1][1]


def _cut_ranges(
    text: str,                          # The line's effective text
    overlays: List[Dict[str, Any]],     # Active overlay correction dicts anchored on this line
    labels: Sequence[str],              # Filterable labels
    keep_last: Sequence[str],           # Labels that keep their last repeated unit
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:  # (merged char ranges to cut, the cuts made)
    cuts: List[Tuple[int, int]] = []
    made: List[Dict[str, Any]] = []
    for c in overlays:
        p = c.get("payload") or {}
        lab = p.get("label")
        if lab not in labels:
            continue
        a = p.get("anchor") or {}
        located = reanchor_span(a, text)
        if located is None:
            made.append({"overlay_id": c.get("id"), "label": lab, "text": a.get("text_snapshot"),
                         "cut": None, "note": "words no longer on the line"})
            continue
        cs, ce = located
        if lab in keep_last:
            ks, ke = repeat_survivor(text[cs:ce])
            rng = (cs, cs + ks)          # everything before the surviving unit
            tail = (cs + ke, ce)         # trailing punctuation after it, if any
            cuts.append(rng)
            if tail[1] > tail[0]:
                cuts.append(tail)
            made.append({"overlay_id": c.get("id"), "label": lab, "text": text[cs:ce],
                         "cut": text[cs:cs + ks], "kept": text[cs + ks:cs + ke]})
        else:
            cuts.append((cs, ce))
            made.append({"overlay_id": c.get("id"), "label": lab, "text": text[cs:ce],
                         "cut": text[cs:ce]})
    cuts.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in cuts:
        if e <= s:
            continue
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged, made


def _tidy(s: str) -> str:  # Whitespace + punctuation seams after a cut
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:?!])", r"\1", s)          # "larger, than" — a space before punctuation
    s = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", s)    # ", ," -> ","
    s = re.sub(r"^[,;:]\s*", "", s)                  # a leading comma left by a cut at the line start
    return s


def subtract_spans(
    text: str,                        # The line's effective text
    ranges: List[Tuple[int, int]],    # Merged char ranges to cut (from _cut_ranges)
    marker: str = ELISION_MARKER,     # What a cut leaves behind ("" = nothing)
) -> str:  # The clean text (tidied)
    """Cut the ranges out of the text, leaving `marker` at each seam (pure)."""
    if not ranges:
        return text
    out: List[str] = []
    at = 0
    for s, e in ranges:
        out.append(text[at:s])
        if marker:
            out.append(f" {marker} ")
        at = e
    out.append(text[at:])
    joined = "".join(out)
    if marker:   # never let the marker glue to punctuation-less neighbours: "we [...] have"
        joined = re.sub(r"\s*" + re.escape(marker) + r"\s*", f" {marker} ", joined)
    return _tidy(joined)


def clean_read(
    segments: Sequence[SpineSegment],     # The EFFECTIVE spine (corrections applied), index order
    strata: List[Dict[str, Any]],         # active_strata output
    overlays: List[Dict[str, Any]],       # active_speech_overlays output
    *,
    exclude_strata: Sequence[str] = CLEAN_READ_EXCLUDE_STRATA,  # Line-level subtraction classes
    filter_labels: Sequence[str] = CLEAN_READ_FILTER_LABELS,    # Overlay labels a reader filters
    keep_last_unit: Sequence[str] = CLEAN_READ_KEEP_LAST_UNIT,  # Labels whose span keeps its last unit
    marker: str = ELISION_MARKER,          # In-line elision marker ("" = clean-verbatim)
) -> List[Dict[str, Any]]:  # Kept lines: {id, index, start, end, text, raw_text, elisions, dropped_before}
    """L1 — the clean read (pure). Kept lines in spine order; each carries its
    clean `text`, the L0 `raw_text`, the `elisions` cut inside it, and
    `dropped_before`: the lines elided between the previous kept line and
    this one (segment id, index, why). Wordless segments are not lines and
    are never counted as dropped."""
    idx = strata_index(strata)
    drop = set(exclude_strata)
    by_seg: Dict[str, List[Dict[str, Any]]] = {}
    for c in overlays:
        sid = ((c.get("payload") or {}).get("anchor") or {}).get("segment_id")
        if sid:
            by_seg.setdefault(sid, []).append(c)
    out: List[Dict[str, Any]] = []
    pending_drops: List[Dict[str, Any]] = []
    for s in sorted(segments, key=lambda x: x.index):
        if s.is_empty:
            continue
        cats = drop & set(idx.get(s.id, []))
        if cats:
            pending_drops.append({"id": s.id, "index": s.index, "why": sorted(cats)})
            continue
        text = s.text or ""
        ranges, made = _cut_ranges(text, by_seg.get(s.id, []), filter_labels, keep_last_unit)
        clean = subtract_spans(text, ranges, marker) if ranges else text
        bare = clean.replace(marker, "").strip() if marker else clean.strip()
        if not bare:
            pending_drops.append({"id": s.id, "index": s.index,
                                  "why": ["emptied by " + ", ".join(sorted({m["label"] for m in made}))]})
            continue
        out.append({"id": s.id, "index": s.index,
                    "start": float(s.start_time) if s.start_time is not None else None,
                    "end": float(s.end_time) if s.end_time is not None else None,
                    "text": clean, "raw_text": text, "elisions": made,
                    "dropped_before": pending_drops})
        pending_drops = []
    if pending_drops and out:   # a trailing run of dropped lines hangs off the last kept line
        out[-1]["dropped_after"] = pending_drops
    return out


def clean_read_summary(lines: List[Dict[str, Any]]) -> Dict[str, int]:  # Counts for a pack / a report
    dropped = sum(len(ln.get("dropped_before") or []) + len(ln.get("dropped_after") or []) for ln in lines)
    cut = sum(1 for ln in lines for m in (ln.get("elisions") or []) if m.get("cut") is not None)
    return {"lines_kept": len(lines), "lines_dropped": dropped, "spans_cut": cut}


def clean_read_pack_read(
    lines: List[Dict[str, Any]],  # clean_read output
    marker: str = ELISION_MARKER,  # The marker the lines carry
) -> Dict[str, Any]:  # build_filter_pack's `read` argument for an L1 pack
    """The `read` record an L1 pack carries: the layer, the marker, the
    counts, and the gaps (segment id -> the lines dropped before it) the
    pack builder attaches to its rows so the brief shows every elision."""
    return {"layer": "clean", "marker": marker, **clean_read_summary(lines),
            "gaps": {ln["id"]: ln["dropped_before"] for ln in lines if ln.get("dropped_before")}}


def clean_read_segments(
    lines: List[Dict[str, Any]],  # clean_read output
    segments: Sequence[SpineSegment],  # The same effective spine (for the fields a line does not carry)
) -> List[SpineSegment]:  # SpineSegments carrying the CLEAN text — what a pack builder reads as L1
    """Project the clean read back onto SpineSegments (clean text, same
    identity and times), so every pack builder reads L1 unchanged."""
    by_id = {s.id: s for s in segments}
    return [replace(by_id[ln["id"]], text=ln["text"]) for ln in lines]
