"""Configuration for the pool-watch Stage 1/2/3 pipeline."""

video_path = "pool_video.mp4"
pose_landmarker_model_path = "pose_landmarker_lite.task"
visibility_threshold = 0.6
heuristics_window_seconds = 3.0
min_heuristics = 3

# Per-heuristic persistence (N = trailing frames, X = min passes in that N).
h1_persistence_window_frames = 45
h1_persistence_min_passes = 20
h2_persistence_window_frames = 60
h2_persistence_min_passes = 2
h3_persistence_window_frames = 45
h3_persistence_min_passes = 20
h5_persistence_window_frames = 30
h5_persistence_min_passes = 6

nebius_api_key = ""
nebius_model = "Qwen/Qwen2.5-VL-72B-Instruct"
nebius_base_url = "https://api.studio.nebius.com/v1/chat/completions"

openrouter_api_key = ""
openrouter_model = "openai/gpt-4o-mini"
vlm_provider = "nebius"

vlm_frame_size = 768
vlm_timeout_seconds = 20
confidence_threshold = 0.7
ntfy_topic = "pool-watch-alerts"

demo_loop_start = 0.0
demo_loop_end = 14.0
