# AGENTS.md

## Project Goal
Detect drowning/distress in residential pool footage using a 3-stage pipeline.

## Architecture

Stage 1 — MediaPipe heuristics
Stage 2 — Nebius vision confirmation
Stage 3 — Uncertain frame labeling queue + Alert via ntfy

Alert only when Stage 1 AND Stage 2 agree.

## Dependencies
mediapipe
opencv-python
numpy
requests

## Pipeline Logic

MediaPipe → heuristics (4/6 trigger) → flag

Flagged frame → Nebius

YES → alert via ntfy
UNSURE → save to uncertain_frames/
NO → reset

## Distress Heuristics

1. Spine vector 70–90° from horizontal (3s stable)
2. Wrist oscillation at shoulder level
3. Head bobbing relative to shoulders
4. Near-zero displacement over 3s
5. Head tilted back
6. Stroke rate 1–2/sec without progress

Trigger when ≥4 active.

## File Structure

project/
├── main.py
├── config.py
├── AGENTS.md
├── README.md
├── uncertain_frames/
└── pool_video.mp4

## Notes
- Video is pre-cropped to pool area. No POOL_CROP needed — do not add it.
- Nebius is the primary VLM provider. OpenRouter is fallback only.
- Switching providers requires changing VLM_PROVIDER in config.py only.

## Config Block

Expose in config.py:
video_path
visibility_threshold
heuristics_window_seconds
min_heuristics
nebius_api_key
nebius_model
nebius_base_url
openrouter_api_key        # fallback only
openrouter_model          # fallback only
vlm_provider              # "nebius" or "openrouter"
vlm_frame_size
vlm_timeout_seconds
confidence_threshold
ntfy_topic
demo_loop_start
demo_loop_end

## Definition of Done

- [ ] Video loops between demo_loop_start and demo_loop_end
- [ ] MediaPipe landmarks detected with visibility ≥ 0.6
- [ ] 4/6 heuristics trigger Stage 1 flag
- [ ] Flagged frame sent to Nebius for confirmation
- [ ] YES → ntfy alert fires on phone
- [ ] UNSURE → frame saved to uncertain_frames/ with timestamp + confidence in filename
- [ ] NO → pipeline resets
- [ ] VLM_PROVIDER swap requires changing one variable only