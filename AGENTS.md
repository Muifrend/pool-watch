# AGENTS.md

## Project Goal
Detect drowning/distress in residential pool footage using a 3-stage pipeline.

## Architecture

Stage 1 — MediaPipe heuristics  
Stage 2 — OpenRouter vision confirmation  
Stage 3 — Uncertain frame labeling queue  
Stage 4 — Alert via ntfy

Alert only when Stage 1 AND Stage 2 agree.

## Dependencies
mediapipe
opencv-python
numpy
requests

## Pipeline Logic

MediaPipe → heuristics (4/6 trigger) → flag

Flagged frame → OpenRouter

YES → alert
UNSURE → save to uncertain_frames
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
main.py
AGENTS.md
README.md
uncertain_frames/
assets/pool_clip.mp4

## Config Block

Expose:
video_path
visibility_threshold
heuristics_window
min_heuristics
openrouter_api_key
model_name
ntfy_topic

## Definition of Done

- Video loops
- MediaPipe landmarks detected
- 4/6 heuristics trigger Stage 1
- Frame sent to OpenRouter
- YES triggers ntfy
- UNSURE saves frame