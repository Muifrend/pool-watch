# AGENTS.md

## Project Goal
Detect drowning/distress in residential pool footage using a 3-stage pipeline with low false positives.

## Architecture
Stage 1 — MediaPipe pose + temporal distress heuristics.
Stage 2 — VLM confirmation (`nebius` or `openrouter` provider).
Stage 3 — Routing: ntfy alert for `YES`, uncertain queue for `UNSURE`, no-op for `NO`.

Alerting rule: send alert only when Stage 1 flags and Stage 2 verdict is `YES`.

## Current Implementation Truth
Stage 1 currently runs 4 heuristics in code (`h1`, `h2`, `h3`, `h5`), not 6.
Stage 1 default threshold is `min_heuristics = 3` (3-of-4).
Stage 2 has a 10-second debounce for repeated `YES`.
Stage 3 does not use confidence gating; verdict-only routing is intentional.

## Distress Heuristics
Implemented:
1. Spine vector 70–90 degrees from horizontal.
2. Wrist oscillation near shoulder level.
3. Head bobbing relative to shoulders.
4. Head tilt back proxy (`h5`).

Planned but not implemented in code:
1. Near-zero displacement over 3s.
2. Stroke rate 1–2/sec without progress.

## Pipeline Logic
1. Run Stage 1 per frame (video loops between `demo_loop_start` and `demo_loop_end`).
2. If Stage 1 does not flag, continue.
3. If Stage 1 flags, call Stage 2 VLM.
4. Route Stage 2 verdict:
`YES` -> send ntfy alert.
`UNSURE` -> save `.jpg + .json` sidecar to `uncertain_frames/`.
`NO` -> no-op/reset.

## Stage 3 Queue Contract
Filename format:
`YYYYMMDD_HHMMSS_t{timestamp_sec:.2f}_conf{confidence:.2f}.jpg`
`YYYYMMDD_HHMMSS_t{timestamp_sec:.2f}_conf{confidence:.2f}.json`

Sidecar JSON keys:
`frame_index`
`timestamp_sec`
`saved_at`
`stage2_verdict`
`stage2_confidence`
`provider_used`
`raw_response`
`stage1_active_count`
`stage1_fired_heuristics`
`stage1_scores` (mapped directly from `HeuristicResult.heuristic_scores`)

## Config Surface
Keep these available in `config.py`:
`video_path`
`pose_landmarker_model_path`
`visibility_threshold`
`heuristics_window_seconds`
`min_heuristics`
`nebius_api_key`
`nebius_model`
`nebius_base_url`
`openrouter_api_key`
`openrouter_model`
`openrouter_base_url`
`vlm_provider`
`vlm_frame_size`
`vlm_timeout_seconds`
`confidence_threshold` (unused by Stage 3 routing)
`ntfy_topic` (now sourced from `NTFY_TOPIC` env with fallback)
`demo_loop_start`
`demo_loop_end`

## Operational Notes
- Video is pre-cropped to pool area. Do not add `POOL_CROP`.
- Provider switching should remain one-variable (`vlm_provider`) behavior.
- If Stage 2 cannot reach provider (DNS/network/auth), verdict becomes `UNSURE` and frames are queued.
- If no phone alert arrives, check both Stage 2 verdicts and the ntfy topic being used at runtime.

## Key Files
`main.py` — pipeline runner and Stage 1->2->3 orchestration.
`detector.py` — Stage 1 heuristics + `HeuristicResult`.
`vlm.py` — Stage 2 provider calls and verdict parsing.
`stage3.py` — ntfy sender + uncertain queue writer.
`config.py` — all runtime knobs.
`tests/` — unit tests for stage behavior and contracts.
