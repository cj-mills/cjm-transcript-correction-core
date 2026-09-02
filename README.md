# cjm-transcript-correction-core

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

A frontend-agnostic core for the transcript correction workflow — the first downstream graph-extension core; composes the graph-storage capability worker into a headless pipeline that applies unified, non-destructive corrections (text, punctuation, segmentation) as a supersede-able overlay on a committed decomposition spine, recomputes the review worklist from deterministic signals, and exposes a CLI as its first driver.

## Modules

- **`cjm_transcript_correction_core.__init__`**
- **`cjm_transcript_correction_core.cli`** — The CLI driver — the correction core's first (and currently only) frontend. run <decomp-manifest> corrects the committed spine in the decomp graph DB, pointing the graph worker at that shared DB via load-time config, with optional session resume/reopen; review runs the interactive text-correction loop (the cross-transcriber diff is intra-graph since stage 5).
- **`cjm_transcript_correction_core.graph`** — The correction overlay's graph I/O: targeted (scale-shaped) reads of a committed spine via the graph-storage query action, construction of Correction / CorrectionSession nodes + CORRECTS / SUPERSEDES / DERIVED_FROM / REVIEWED edges, the in-core effective-spine projection (layer-0 + applied corrections), and commit through the job queue. Hand-rolled (revolution-1) = direct CR-18 spec material; append-only on layer-0 (never update/delete a Segment).
- **`cjm_transcript_correction_core.journal`** — Live append-through for the correction verbs — the workflow journal's domain half.
- **`cjm_transcript_correction_core.launch`** — The shared launch surface every correction shell drives through: the
- **`cjm_transcript_correction_core.models`** — Overlay data shapes for the transcript-correction workflow: the Correction / CorrectionSession graph nodes + their relation registry, the read view of a committed spine segment, the worklist item, run configuration, and the correction run manifest (proto-bundle that chains decomp -> correction).
- **`cjm_transcript_correction_core.pipeline`** — The headless correction workflow: load a decomp run manifest, resolve the shared graph DB, start/resume/reopen a CorrectionSession, recompute the worklist from deterministic signals + persisted review state, run the D14 empty-segment prune (first operation), and record a chainable correction run manifest — with a cheapest-form HITL approval seam.
- **`cjm_transcript_correction_core.signals`** — Pure deterministic Tier-1 signal functions (no capability calls): empty-segment detection, bidirectional boundary punctuation/capitalization heuristics, forced-alignment coverage flags, positional cross-transcriber diff, phonetic + edit-distance variant clustering, and the event-proposal overlay (leg 4: the finetuned detector's spans anchored onto the spine). The worklist is recomputed from these each session; revolution-1 builds ZERO new capabilities.
- **`cjm_transcript_correction_core.spine`**
- **`cjm_transcript_correction_core.state`** — Sidecar view-state + spine-picker helpers — the correction TUI's pure
- **`cjm_transcript_correction_core.strata`** — Strata — the filtering lane's domain half (DECs 304fd984 + 9d4c0a38; work items

## API

### `cjm_transcript_correction_core.cli`

- `bench_command` _function_ — Execute the `bench` subcommand: the reserved-tail verdict join.
- `build_parser` _function_ — Build the CLI parser (subcommands: run, review).
- `commit_wordless_transfer` _function_ — COMMIT a planned transfer — the engine's second half. One
- `export_command` _function_ — Execute `export-wordless-propset`: write one spine's effective wordless
- `extract_command` _function_ — Execute the `extract` subcommand: fold the gated overlay into a manifested dataset.
- `filter_confirm_command` _function_ — Execute `filter-confirm`: the HEADLESS HITL worklist (bc8dbbdd pass-1
- `filter_ingest_command` _function_ — Execute `filter-ingest`: validate proposer rows against their pack and
- `filter_pack_command` _function_ — Execute `filter-pack`: write one spine window's effective text-bearing
- `gate_command` _function_ — Execute the `gate` subcommand: show or assert per-spine extraction gates.
- `load_capabilities` _function_ — Discover manifests + load each capability, passing per-capability config (CR-2 caller-wins).
- `main` _function_ — CLI entry point (console script: `cjm-transcript-correction-core`).
- `overlay_event_rows` _function_ — Overlay span records -> dataset event rows for ONE spine (pure).
- `plan_transfer_rows` _function_ — Place wordless donors on the destination (pure) — the event half of
- `plan_wordless_export` _function_ — PLAN an export (reads only) — the `export-wordless-propset` engine's
- `plan_wordless_transfer` _function_ — PLAN a wordless transfer (reads only) — the `transfer-wordless`
- `resolve_source_node` _function_ — Resolve the --source selector the respine verbs share: exactly ONE
- `review_command` _function_ — Execute the `review` subcommand: interactive text corrections over the flagged worklist.
- `run_command` _function_ — Execute the `run` subcommand: correct a decomp manifest's committed spine.
- `run_extract` _function_ — The extract fold on an ALREADY-OPEN graph seat (flywheel build leg 2,
- `scan_command` _function_ — Execute `scan-mishomed`: flag authoritative FA words stranded outside
- `stats_command` _function_ — Execute the `stats` subcommand: flywheel accounting over the shared graph.
- `transfer_command` _function_ — Execute `transfer-wordless`: replay wordless event inserts (and speaker-
- `wordless_donors` _function_ — The EFFECTIVE wordless layer of a spine: labeled, effectively wordless
- `write_wordless_propset` _function_ — WRITE a planned export as a proposal set — the engine's write half.

### `cjm_transcript_correction_core.graph`

- `active_corrections` _function_ — Filter to the effective correction set (the layer's resolve_active over a read superseded set).
- `active_speaker_assignments` _function_ — Project the ACTIVE speaker assignment per segment (latest-wins).
- `active_speech_overlays` _function_ — The surviving speech-overlay corrections (supersession applied; pure).
- `aggregate_session_purposes` _function_ — Fold sessions into a per-source purpose mix (pure; d915d545 picker rung).
- `apply_chunk_inserts` _function_ — Synthesize inserted chunks into the effective spine (DEC 3d3fa2a8).
- `apply_time_nudges` _function_ — Apply timing corrections onto segment times (latest-wins per edge).
- `bench_event_proposals` _function_ — The reserved-tail bench join (leg 4, DECs 8e05b87b + 8cf12c22) — pure.
- `build_boundary_shift_correction` _function_ — Build a grouping Correction that moves text across one segment boundary.
- `build_chunk_insert_correction` _function_ — Build an insertion Correction that adds a chunk the skeleton never cut (DEC 3d3fa2a8).
- `build_chunk_split_corrections` _function_ — Compose a chunk SPLIT from the EXISTING verbs (work item 99c1d2ba) — no
- `build_correction_node` _function_ — Construct a Correction overlay node (pure; commit happens separately).
- `build_extraction_gate_assertion` _function_ — Build one extraction-gate ASSERTION (DEC 8e05b87b — flywheel build leg 1).
- `build_mark_correction` _function_ — Build a NON-MUTATING mark Correction (DEC 2a231843: routed attention).
- `build_prune_amendment` _function_ — Build a grouping Correction that supersedes a prune with a REDUCED set (unprune).
- `build_prune_correction` _function_ — Build one batch grouping Correction that prunes empty segments (D14).
- `build_reject_review` _function_ — Build a review Correction that REJECTS a prior correction (reject-as-supersede).
- `build_speaker_assign_correction` _function_ — Build a speaker Correction — the assignment op envelope (DEC d6df3a8e).
- `build_speech_overlay_correction` _function_ — Build a NON-MUTATING speech-overlay Correction (check fc42614d, DEC 4e05a066).
- `build_stratum_correction` _function_ — Build a NON-MUTATING stratum Correction (DECs 304fd984 + 9d4c0a38: the
- `build_text_correction` _function_ — Build a text_content Correction + its CORRECTS (+ optional SUPERSEDES) edges.
- `build_time_nudge_correction` _function_ — Build a timing Correction that nudges segment boundary TIMES (node + CORRECTS edges).
- `commit_boundary_shift_correction` _function_ — Commit a boundary-shift correction (node + CORRECTS x2 [+ SUPERSEDES]) + REVIEWED markers on both segments.
- `commit_chunk_insert_correction` _function_ — Commit a chunk insertion (node + CORRECTS per flank).
- `commit_chunk_insert_removal` _function_ — Remove an inserted chunk WITHOUT touching layer-0 (reject-as-supersede).
- `commit_chunk_split_correction` _function_ — Commit a chunk split: three composed nodes in ONE atomic batch + ONE journal op.
- `commit_chunk_split_removal` _function_ — UNSPLIT: remove a split's right half AND its whole group (one review
- `commit_extraction_gate` _function_ — Commit one extraction-gate assertion (node + GATES edge) — the journaled
- `commit_mark_correction` _function_ — Commit a mark (node + CORRECTS per anchored segment [+ SUPERSEDES]).
- `commit_mark_dismissal` _function_ — Dismiss an open mark WITHOUT a correction (reject-as-supersede).
- `commit_nodes_edges` _function_ — Commit overlay nodes/edges through the layer's idempotent extend_graph.
- `commit_prune_amendment` _function_ — Commit an unprune amendment (node + DERIVED_FROM edges + SUPERSEDES).
- `commit_speaker_assign_correction` _function_ — Commit a speaker assignment (node + CORRECTS per segment + ASSIGNS).
- `commit_speaker_entity` _function_ — Mint a source-spanning Entity into the shared registry (DEC 4ec6a49c).
- `commit_speech_overlay_correction` _function_ — Commit a speech overlay (node + CORRECTS [+ SUPERSEDES]).
- `commit_speech_overlay_removal` _function_ — Remove a speech overlay WITHOUT a replacement (reject-as-supersede).
- `commit_stratum_correction` _function_ — Commit a stratum (node + CORRECTS per covered segment [+ SUPERSEDES]).
- `commit_stratum_retraction` _function_ — Retract a live stratum WITHOUT replacing it (reject-as-supersede, the
- `commit_text_correction` _function_ — Commit a text_content correction (node + CORRECTS [+ SUPERSEDES]) + a REVIEWED marker.
- `commit_time_nudge_correction` _function_ — Commit a time-nudge correction (node + CORRECTS per touched segment).
- `correction_stats` _function_ — Fold one Source's ACTIVE overlay into flywheel-accounting counts (pure).
- `corrections_to_edits` _function_ — Map this core's Correction payloads onto the layer's spine-edit vocabulary.
- `count_source_segments` _function_ — Count a Source's segments server-side under its chosen rendition + skeleton (typed count mode).
- `extract_spine_dataset` _function_ — Fold ONE spine's overlay into its v1 insert-span dataset slice (pure).
- `fa_words_for_transcript` _function_ — One transcript's FA words in source coordinates (the scan-mishomed join,
- `find_active_text_correction` _function_ — Single-segment convenience over the batch read (cross-session; latest wins).
- `find_active_text_corrections_batch` _function_ — Active text corrections for MANY segments in TWO round-trips (C17).
- `find_chunk_split_group` _function_ — Resolve a split right-half's GROUP — the ac84360a group marker cashed in.
- `find_corrections_for_session` _function_ — List corrections recorded in a session (typed property filter).
- `find_prior_corrections_by_hash` _function_ — Cross-transcript correction-cache lookup (targeted; the graph IS the lexicon).
- `get_session` _function_ — Fetch a CorrectionSession node by id (resume/reopen) — typed get, dict shape preserved.
- `labeled_insert_spans` _function_ — Fold the ACTIVE chunk inserts into span records with op-chain provenance (pure).
- `latest_extraction_gates` _function_ — Fold gate assertions into the live per-spine gate state (pure; latest-wins).
- `list_source_spines` _function_ — The SPINES coexisting under a Source's chosen rendition (DEC f1024568).
- `list_speaker_entities` _function_ — Read the source-spanning entity registry (the picker's registry tier).
- `load_empty_segments` _function_ — Load ONLY a Source's empty-text segments under its chosen rendition (D14 prune).
- `load_extraction_gates` _function_ — Read one Source's live per-spine extraction gates (typed property filter).
- `load_review_markers` _function_ — Load a session's review markers (typed edge projection over REVIEWED edges).
- `load_source_corrections` _function_ — Load every Correction targeting a Source (across sessions) + the superseded-id set.
- `load_source_segments` _function_ — Load a Source's fine Segment spine under its chosen rendition + skeleton (typed query surface).
- `load_variant_texts` _function_ — Resolve per-transcriber chunk texts from the segments' CharSlice refs.
- `mark_anchor_segments` _function_ — Validate a mark anchor and list the Segment ids it touches.
- `negative_regions` _function_ — Complement the accounted spans below the watermark (pure; DEC 8e05b87b).
- `open_marks` _function_ — Filter to the OPEN marks — the pass-2 worklist ('query open marks, walk them').
- `project_effective_spine` _function_ — Project the effective spine = layer-0 + applied corrections.
- `reanchor_span` _function_ — Re-locate a span anchor in text that may have been edited since mark time.
- `record_review_markers` _function_ — Persist per-(session, segment) review markers as REVIEWED edges.
- `rename_speaker_entity` _function_ — Rename an Entity on its STABLE id — identification IS a rename (DEC 484e2d74).
- `reorder_gap_inserts` _function_ — Restore TIME order among stacked inserts after the nudge stage
- `resolve_source_renditions` _function_ — Pick the AudioRendition set whose fine Segment spine correction operates on.
- `session_purposes_by_source` _function_ — Read every CorrectionSession once and fold the per-source purpose mix.
- `set_session_purpose` _function_ — Update a session's purpose + updated_at (the test-session hygiene tag, DEC c86714a4).
- `set_session_status` _function_ — Update a session's status + updated_at.
- `skeleton_hash_for` _function_ — Resolve a skeleton selector to the chosen spine's HASH (pure) — the gate's
- `source_audio_segment_ids` _function_ — The Source's coarse spine (one small typed read; ordered by index).
- `speech_overlay_spans` _function_ — Fold the ACTIVE speech overlays into extraction-facing span records (pure).
- `spine_where_for` _function_ — Resolve a skeleton selector against the observed spine set (pure).
- `start_session` _function_ — Create + commit a new CorrectionSession node.
- `submit_and_wait` _function_ — Submit one capability job, wait for it, return its result (raise on failure).

### `cjm_transcript_correction_core.journal`

- `correction_replay_handlers` _function_ — The correction core's registered replay vocabulary (replay stays DOMAIN-OWNED).
- `journal_correction_op` _function_ — Append one correction op — envelope + semantic args + the EXACT wires committed.
- `segment_anchor` _function_ — The run-independent anchor stamped on every correction op (DEC ccbab9f5 point 5).

### `cjm_transcript_correction_core.launch`

- `build_parser` _function_ — The TUI driver's argument surface (mirrors correction-core's run/review args).
- `resolve_settings` _function_ — Resolve the launch settings every shell shares (5daadfc4 workspace

### `cjm_transcript_correction_core.models`

- `Correction` _class_ — A single non-destructive correction over the committed spine (overlay node).
- `CorrectionConfig` _class_ — Configuration for one correction run.
- `CorrectionManifest` _class_ — Durable record of one correction run (proto-bundle; chainable, CR-20).
- `CorrectionRelations` _class_ — Registry of edge types the correction overlay adds to the spine graph.
- `CorrectionSession` _class_ — A resumable, reopen-able correction review over one or more sources.
- `DatasetManifest` _class_ — Durable record of one dataset extraction (chainable; DEC 16159e09).
- `Entity` _class_ — A source-spanning identity in the shared entity substrate (DEC 4ec6a49c).
- `ExtractionGate` _class_ — One per-spine extraction-gate ASSERTION (DEC 8e05b87b — flywheel build leg 1).
- `SpineSegment` _class_ — A committed layer-0 Segment loaded from the graph (read view).
- `WorklistItem` _class_ — One spine segment surfaced for review, with its deterministic Tier-1 flags.
- `new_dataset_id` _function_ — Generate a unique, sortable dataset id (the new_run_id pattern, dataset kind).
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
- `event_span_proposals` _function_ — Anchor pending event proposals onto the spine — the propose lane's paint.
- `fa_coverage_flags` _function_ — Flag segments whose forced-alignment coverage looks suspect (Tier-1).
- `levenshtein` _function_ — Levenshtein edit distance (pure, in-core; variant-clustering primitive).
- `load_event_proposal_set` _function_ — Find the latest proposal set for a source (the turns-artifact discovery
- `phonetic_key` _function_ — Compute a coarse phonetic key for a word (groups like-sounding variants).
- `speaker_turn_proposals` _function_ — Dominant diarization cluster per segment — the assign lane's proposal paint.
- `variant_divergence` _function_ — Within-segment cross-transcriber divergence (stage 5: intra-graph).

### `cjm_transcript_correction_core.spine`

- `ChunkRef` _class_ — Where one Segment's VAD-chunk audio lives: the model-input WAV + the chunk-local span.
- `SeamRef` _class_ — A source-coordinate audio span across one fine-spine boundary (the g/G
- `SpineView` _class_ — One Source's effective correction spine, cursor-windowed for the TUI.
- `list_sources` _function_ — Enumerate the graph's Source nodes (the discovery corpus, 2ce81638).
- `load_source_slice` _function_ — Decode a source-coordinate slice of the ORIGINAL media to playable samples.
- `match_sources` _function_ — The --source selector (pure; shared by direct open and the picker's seed).
- `neighbor_word_bound` _function_ — The adjacent word's FA boundary facing an overlay span (pure).
- `open_stack` _function_ — Bootstrap the graph capability stack, resolving the db path (2ce81638).
- `parse_entity_input` _function_ — Parse the new-speaker editor line (pure). A leading `?` marks the entity
- `parse_mark_input` _function_ — Parse the M-editor mark grammar (pure; the DEC 2a231843 TUI gesture).
- `plan_boundary_shift` _function_ — Plan a ONE-WORD boundary shift (the [ / ] gesture unit).
- `plan_chunk_insert` _function_ — Plan a chunk insertion into the seam after the cursor (the i gesture unit; pure).
- `plan_chunk_split` _function_ — Plan a chunk split at a caret position (the S gesture unit; pure).
- `plan_gate` _function_ — Plan an extraction-gate assertion (the F gesture unit; pure — DEC 8e05b87b).
- `plan_split_rows` _function_ — Place speaker-split boundaries on a destination spine (pure) — the
- `plan_time_nudge` _function_ — Plan a boundary-time nudge (the ,/. and </> gesture unit; pure).
- `resolve_mark_class_token` _function_ — Resolve a leading digit token to its menu class (the M picker; pure).
- `segment_word_tokens` _function_ — Tokenize a segment's text into words WITH character offsets (pure).
- `snap_word_span` _function_ — Derive a word range's TIME span from FA word timestamps (pure; fc42614d).
- `source_status` _function_ — Correction-status-at-a-glance for one Source (the picker's detail row).
- `split_donors` _function_ — The SPEAKER-SPLIT layer of a spine (pure) — the DoD rider 54aac7d3 on

### `cjm_transcript_correction_core.state`

- `load_tui_state` _function_ — Read the per-graph TUI sidecar state (last-focused positions).
- `save_tui_state` _function_ — Merge one source's view state into the sidecar state file.
- `selector_for_spine` _function_ — The selector value a picker choice persists (pure): the full skeleton
- `spine_label` _function_ — One picker row's config summary for a skeleton spine (pure).

### `cjm_transcript_correction_core.strata`

- `active_strata` _function_ — The live strata: stratum corrections neither superseded (reclassified /
- `bench_filter_proposals` _function_ — Derive the filtering verdicts (DEC 8e05b87b, the bench_event_proposals
- `build_filter_pack` _function_ — Build the proposer's input: one source window's text-bearing effective
- `exclude_strata` _function_ — The per-consumer filtered projection (DEC 9d4c0a38): "filtered" is a
- `load_filter_proposal_sets` _function_ — Every filtering proposal set for a source (and optionally one spine),
- `new_pack_id` _function_ — Generate a unique, sortable pack id.
- `pack_digest` _function_ — Digest the READ content (source binding + window + numbered segments) —
- `pending_filter_proposals` _function_ — The headless worklist: proposals with NO live stratum carrying their id and
- `proposals_from_rows` _function_ — Resolve validated rows to proposal-set rows: proposal id, category, source
- `render_filter_pack` _function_ — Render a pack as the brief a proposer reads: identity + window, the class
- `strata_index` _function_ — Segment-keyed view of the live strata (the consumer query's index).
- `validate_proposal_rows` _function_ — Validate + normalize proposer rows against their pack — loud on the first
- `write_filter_propset` _function_ — Write one filtering proposal set: `<out_root>/<set_id>/manifest.json` +

## Dependencies

**Depends on:** `cjm-capability-primitives`, `cjm-context-graph-layer`, `cjm-context-graph-primitives`, `cjm-substrate`, `cjm-transcript-graph-schema`, `numpy`
**Used by:** `cjm-transcript-correction-qt`, `cjm-transcript-correction-tui`, `cjm-transcription-core`, `cjm-workflow-hub-qt`, `cjm-workflow-hub-tui`
