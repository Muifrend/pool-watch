# Pool Watch

Pool Watch is a 3-stage drowning/distress monitoring pipeline for pre-cropped residential pool footage.

## Stages
1. Stage 1 (`detector.py`): MediaPipe pose + temporal heuristics.
2. Stage 2 (`vlm.py`): VLM confirmation (`nebius` or `openrouter`).
3. Stage 3 (`stage3.py`): verdict router.
`YES` -> ntfy alert.
`UNSURE` -> save uncertain frame + metadata sidecar.
`NO` -> no-op.

Stage 3 routing is verdict-only and intentionally does not use confidence gating.

## Current Heuristics
Implemented Stage 1 heuristics:
1. Spine verticality (`h1_spine_vertical`)
2. Wrist oscillation (`h2_wrist_oscillation`)
3. Head bobbing (`h3_head_bobbing`)
4. Head tilt back proxy (`h5_head_tilt_back`)

Default flag threshold is `min_heuristics = 3` (3-of-4).

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
`pip install mediapipe opencv-python numpy requests python-dotenv`
3. Download pose model:
`curl -L https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task -o pose_landmarker_lite.task`
4. Configure `.env`:
`NEBIUS_API_KEY=...`
`OPENROUTER_API_KEY=...`
`NTFY_TOPIC=your-topic`

## Configuration
Main runtime config is in `config.py`.

Important fields:
`vlm_provider` (`"nebius"` or `"openrouter"`)
`vlm_timeout_seconds`
`ntfy_topic` (resolved from `NTFY_TOPIC` env, fallback: `"pool-watch-alerts"`)
`demo_loop_start` / `demo_loop_end`
`min_heuristics`

## Run
Run pipeline:
`python main.py`

Run tests:
`python -m unittest discover -s tests -q`

## Uncertain Queue Output
When Stage 2 returns `UNSURE`, files are written to `uncertain_frames/`:

Filename format:
`YYYYMMDD_HHMMSS_t{timestamp_sec:.2f}_conf{confidence:.2f}.jpg`
`YYYYMMDD_HHMMSS_t{timestamp_sec:.2f}_conf{confidence:.2f}.json`

Sidecar JSON fields:
`frame_index`
`timestamp_sec`
`saved_at`
`stage2_verdict`
`stage2_confidence`
`provider_used`
`raw_response`
`stage1_active_count`
`stage1_fired_heuristics`
`stage1_scores`

## Troubleshooting
- No ntfy alert:
1. Confirm Stage 2 actually returns `YES` (alerts are sent only on `YES`).
2. Confirm runtime topic from `config.ntfy_topic` matches your intended `NTFY_TOPIC`.
3. Confirm you are subscribed to that exact ntfy topic on your phone.

- Lots of `UNSURE` + sidecar files:
1. Check network/DNS/auth for the selected VLM provider.
2. If provider call fails, Stage 2 returns `UNSURE` (`raw_response` may be `__exception__`), which is queued by design.

- Provider switching:
1. Change `vlm_provider` in `config.py`.
2. Ensure the matching API key is present in `.env`.
