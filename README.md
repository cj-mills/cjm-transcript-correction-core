# cjm-transcript-correction-core

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

A frontend-agnostic core for the transcript correction workflow — the first downstream graph-extension core; composes the graph-storage capability worker into a headless pipeline that applies unified, non-destructive corrections (text, punctuation, segmentation) as a supersede-able overlay on a committed decomposition spine, recomputes the review worklist from deterministic signals, and exposes a CLI as its first driver.

## Modules

- **`cjm_transcript_correction_core.cli`** — The CLI driver — the correction core's first (and currently only) frontend. run <decomp-manifest> corrects the committed spine in the decomp graph DB, pointing the graph worker at that shared DB via load-time config, with optional session resume/reopen; review runs the interactive text-correction loop (the cross-transcriber diff is intra-graph since stage 5).
- **`cjm_transcript_correction_core.graph`** — The correction overlay's graph I/O: targeted (scale-shaped) reads of a committed spine via the graph-storage query action, construction of Correction / CorrectionSession nodes + CORRECTS / SUPERSEDES / DERIVED_FROM / REVIEWED edges, the in-core effective-spine projection (layer-0 + applied corrections), and commit through the job queue. Hand-rolled (revolution-1) = direct CR-18 spec material; append-only on layer-0 (never update/delete a Segment).
- **`cjm_transcript_correction_core.journal`**
- **`cjm_transcript_correction_core.models`** — Overlay data shapes for the transcript-correction workflow: the Correction / CorrectionSession graph nodes + their relation registry, the read view of a committed spine segment, the worklist item, run configuration, and the correction run manifest (proto-bundle that chains decomp -> correction).
- **`cjm_transcript_correction_core.pipeline`** — The headless correction workflow: load a decomp run manifest, resolve the shared graph DB, start/resume/reopen a CorrectionSession, recompute the worklist from deterministic signals + persisted review state, run the D14 empty-segment prune (first operation), and record a chainable correction run manifest — with a cheapest-form HITL approval seam.
- **`cjm_transcript_correction_core.signals`** — Pure deterministic Tier-1 signal functions (no capability calls): empty-segment detection, bidirectional boundary punctuation/capitalization heuristics, forced-alignment coverage flags, positional cross-transcriber diff, and phonetic + edit-distance variant clustering. The worklist is recomputed from these each session; revolution-1 builds ZERO new capabilities.

## API

### `cjm_transcript_correction_core.cli`

- `build_parser` _function_ — Build the CLI parser (subcommands: run, review).
- `load_capabilities` _function_ — Discover manifests + load each capability, passing per-capability config (CR-2 caller-wins).
- `main` _function_ — CLI entry point (console script: `cjm-transcript-correction-core`).
- `review_command` _function_ — Execute the `review` subcommand: interactive text corrections over the flagged worklist.
- `run_command` _function_ — Execute the `run` subcommand: correct a decomp manifest's committed spine.

### `cjm_transcript_correction_core.graph`

- `active_corrections` _function_ — Filter to the effective correction set (the layer's resolve_active over a read superseded set).
- `build_boundary_shift_correction` _function_ — Build a grouping Correction that moves text across one segment boundary.
- `build_correction_node` _function_ — Construct a Correction overlay node (pure; commit happens separately).
- `build_prune_amendment` _function_ — Build a grouping Correction that supersedes a prune with a REDUCED set (unprune).
- `build_prune_correction` _function_ — Build one batch grouping Correction that prunes empty segments (D14).
- `build_reject_review` _function_ — Build a review Correction that REJECTS a prior correction (reject-as-supersede).
- `build_text_correction` _function_ — Build a text_content Correction + its CORRECTS (+ optional SUPERSEDES) edges.
- `commit_boundary_shift_correction` _function_ — Commit a boundary-shift correction (node + CORRECTS x2 [+ SUPERSEDES]) + REVIEWED markers on both segments.
- `commit_nodes_edges` _function_ — Commit overlay nodes/edges through the layer's idempotent extend_graph.
- `commit_prune_amendment` _function_ — Commit an unprune amendment (node + DERIVED_FROM edges + SUPERSEDES).
- `commit_text_correction` _function_ — Commit a text_content correction (node + CORRECTS [+ SUPERSEDES]) + a REVIEWED marker.
- `corrections_to_edits` _function_ — Map this core's Correction payloads onto the layer's spine-edit vocabulary.
- `count_source_segments` _function_ — Count a Source's segments server-side under its chosen rendition (typed count mode).
- `find_active_text_correction` _function_ — Single-segment convenience over the batch read (cross-session; latest wins).
- `find_active_text_corrections_batch` _function_ — Active text corrections for MANY segments in TWO round-trips (C17).
- `find_corrections_for_session` _function_ — List corrections recorded in a session (typed property filter).
- `find_prior_corrections_by_hash` _function_ — Cross-transcript correction-cache lookup (targeted; the graph IS the lexicon).
- `get_session` _function_ — Fetch a CorrectionSession node by id (resume/reopen) — typed get, dict shape preserved.
- `load_empty_segments` _function_ — Load ONLY a Source's empty-text segments under its chosen rendition (D14 prune).
- `load_review_markers` _function_ — Load a session's review markers (typed edge projection over REVIEWED edges).
- `load_source_corrections` _function_ — Load every Correction targeting a Source (across sessions) + the superseded-id set.
- `load_source_segments` _function_ — Load a Source's fine Segment spine under its chosen rendition (typed query surface).
- `load_variant_texts` _function_ — Resolve per-transcriber chunk texts from the segments' CharSlice refs.
- `project_effective_spine` _function_ — Project the effective spine = layer-0 + applied corrections.
- `record_review_markers` _function_ — Persist per-(session, segment) review markers as REVIEWED edges.
- `resolve_source_renditions` _function_ — Pick the AudioRendition set whose fine Segment spine correction operates on.
- `set_session_status` _function_ — Update a session's status + updated_at.
- `source_audio_segment_ids` _function_ — The Source's coarse spine (one small typed read; ordered by index).
- `start_session` _function_ — Create + commit a new CorrectionSession node.
- `submit_and_wait` _function_ — Submit one capability job, wait for it, return its result (raise on failure).

### `cjm_transcript_correction_core.journal`

- `correction_replay_handlers` _function_ — The correction core's registered replay vocabulary (replay stays DOMAIN-OWNED).
- `journal_correction_op` _function_ — Append one correction op — envelope + semantic args + the EXACT wires committed.
- `segment_anchor` _function_ — The run-independent anchor stamped on every correction op (DEC ccbab9f5 point 5).
- `sidecar_journal_path` _function_ — The db's sidecar journal path (DEC ccbab9f5 point 3: placement is per-workflow,

### `cjm_transcript_correction_core.models`

- `Correction` _class_ — A single non-destructive correction over the committed spine (overlay node).
- `CorrectionConfig` _class_ — Configuration for one correction run.
- `CorrectionManifest` _class_ — Durable record of one correction run (proto-bundle; chainable, CR-20).
- `CorrectionRelations` _class_ — Registry of edge types the correction overlay adds to the spine graph.
- `CorrectionSession` _class_ — A resumable, reopen-able correction review over one or more sources.
- `SpineSegment` _class_ — A committed layer-0 Segment loaded from the graph (read view).
- `WorklistItem` _class_ — One spine segment surfaced for review, with its deterministic Tier-1 flags.
- `new_run_id` _function_ — Generate a unique, sortable correction run id.

### `cjm_transcript_correction_core.pipeline`

- `collect_capability_info` _function_ — Record capability identity + data-DB pointers for the run manifest (provenance).
- `compute_worklist` _function_ — Recompute the worklist from layer-0 + signals + review state (only decisions persist).
- `confirm_seam` _function_ — HITL approval seam in its cheapest viable form (log + optional CLI prompt).
- `load_decomp_manifest` _function_ — Load + lightly validate a decomp-core run manifest (untyped JSON; CR-20 interchange).
- `prune_empty_segments` _function_ — First operation: prune empty (silence) segments as one grouping correction (D14).
- `resolve_graph_db_path` _function_ — Resolve the graph DB path: explicit override > the decomp manifest's recorded db_path.
- `review_worklist` _function_ — Interactive review loop -> text_content corrections (cheapest HITL seam).
- `run_correction` _function_ — Correct every source in a decomp run manifest (prune + worklist surfacing).
- `run_review` _function_ — Interactive review pass over a decomp manifest's flagged worklist (text corrections).

### `cjm_transcript_correction_core.signals`

- `boundary_punct_caps_flags` _function_ — Bidirectional boundary punctuation/capitalization heuristics (in-segment only).
- `cluster_variants` _function_ — Cluster word variants by phonetic key + edit distance (fix-one-fix-all).
- `compute_signal_flags` _function_ — Combine all deterministic Tier-1 signals into per-segment flags.
- `detect_empty_segments` _function_ — Find empty-text segments (silence VAD chunks with no aligned words; decomp D14).
- `fa_coverage_flags` _function_ — Flag segments whose forced-alignment coverage looks suspect (Tier-1).
- `levenshtein` _function_ — Levenshtein edit distance (pure, in-core; variant-clustering primitive).
- `phonetic_key` _function_ — Compute a coarse phonetic key for a word (groups like-sounding variants).
- `variant_divergence` _function_ — Within-segment cross-transcriber divergence (stage 5: intra-graph).

## Dependencies

**Depends on:** `cjm-context-graph-layer`, `cjm-context-graph-primitives`, `cjm-substrate`, `cjm-transcript-graph-schema`
**Used by:** `cjm-transcript-correction-tui`
