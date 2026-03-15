# PoolGuard – Privacy‑First Pool Distress Detection

PoolGuard is a prototype computer‑vision system that detects possible drowning or distress in residential pool footage. It was designed for a hackathon setting and demonstrates how a lightweight on‑device detector can combine with a vision‑language model for higher confidence alerts.

The system uses a **three‑stage pipeline**:

1. **MediaPipe Pose heuristics** for fast local detection
2. **Vision model confirmation** through OpenRouter
3. **Human‑label queue** for uncertain cases

When both automated stages agree that distress is likely, the system sends an **instant phone alert via ntfy**.

---

# Key Features

### Privacy‑First Architecture

Stage 1 runs completely **on device**. Video never leaves the system unless a potential distress event is detected, and even then only **a single frame** is sent for confirmation.

### Multi‑Stage Detection Pipeline

Instead of relying on one model, the system combines:

* deterministic motion heuristics
* a vision‑language reasoning model

This significantly reduces false alarms while keeping compute costs low.

### Self‑Improving Dataset

When the vision model is uncertain, the frame is automatically saved to a local folder. Over time this creates a labeled dataset that can be used to train a dedicated drowning detection model.

### Realistic Pool Footage

The prototype is tuned for **real overhead pool CCTV angles** with water distortion rather than clean lab data.

---

# How the System Works

## Stage 1 — MediaPipe Pose Detection

MediaPipe Pose extracts body landmarks from each frame of the pool footage. The system only analyzes **upper‑body landmarks**:

* nose
* shoulders
* elbows
* wrists

Landmarks with visibility below **0.6** are ignored.

Six motion heuristics are evaluated:

1. Spine vector 70–90° from horizontal
2. Wrist oscillation at shoulder level
3. Head bobbing relative to shoulders
4. Near‑zero displacement over 3 seconds
5. Head tilted back
6. Stroke rate 1–2/sec with no forward progress

If **4 of 6 heuristics fire simultaneously**, the frame is flagged as potential distress.

---

## Stage 2 — Vision Model Confirmation

Flagged frames are sent to **OpenRouter** where a vision‑language model analyzes the image.

Default model:

```
Qwen2‑VL‑7B‑Instruct
```

The model returns one of three responses:

```
YES
NO
UNSURE
```

Decision logic:

| Stage 1 | Stage 2 | Action     |
| ------- | ------- | ---------- |
| Flagged | YES     | Send alert |
| Flagged | UNSURE  | Save frame |
| Flagged | NO      | Reset      |

If the default model performs poorly, the model string can be swapped to:

```
Llama 3.2 Vision
Gemini Flash 1.5
```

No other code changes are required.

---

## Stage 3 — Uncertain Case Queue

Frames that receive an **UNSURE** response are stored in:

```
uncertain_frames/
```

File names include:

* timestamp
* detection confidence

Example:

```
2026‑03‑15T14‑22‑03_conf0.63.png
```

This folder becomes a future dataset for:

* manual labeling
* model fine‑tuning

---

## Stage 4 — Phone Alerts

Confirmed detections trigger a push notification through **ntfy**.

How it works:

1. Parent installs the free **ntfy mobile app**
2. Parent subscribes to a topic
3. The system sends a POST request to that topic

No accounts or backend infrastructure are required.

---

# Demo Interface

During the demo the video feed displays:

* MediaPipe skeleton overlay
* colored border
* pipeline status text

Color indicators:

| Color  | Meaning              |
| ------ | -------------------- |
| Green  | No distress detected |
| Yellow | Stage 1 flagged      |
| Red    | Alert confirmed      |

The demo clip loops automatically to repeatedly show the detection event.

---

# Installation

Install required dependencies:

```
pip install mediapipe opencv-python numpy requests
```

Optional helper packages:

```
pip install tqdm
```

---

# Setup

## 1. Get an OpenRouter API Key

Create an account:

[https://openrouter.ai](https://openrouter.ai)

Copy your API key.

---

## 2. Configure the Script

At the top of `main.py`, set:

```
OPENROUTER_API_KEY = "your_key"
MODEL_NAME = "qwen2-vl-7b-instruct"
NTFY_TOPIC = "your_pool_alerts"
VIDEO_PATH = "assets/pool_clip.mp4"
```

You can also tune thresholds:

```
VISIBILITY_THRESHOLD
HEURISTIC_WINDOW
MIN_HEURISTICS_TRIGGER
```

---

# Running the System

```
python main.py
```

The system will:

1. Load the pool video
2. Run MediaPipe detection
3. Trigger heuristics
4. Confirm via OpenRouter
5. Send ntfy alerts

---

# Project Structure

```
project/

main.py
AGENTS.md
README.md

assets/
    pool_clip.mp4

uncertain_frames/

```

---

# Future Improvements

Potential next steps after the hackathon:

• Train a dedicated drowning detection model using saved frames

• Replace heuristics with a learned classifier

• Add multi‑person tracking

• Integrate multiple cameras

• Deploy on edge hardware (Jetson / Raspberry Pi)

---

# Why This Architecture Works

### Fast

Heuristic filtering avoids running expensive models on every frame.

### Cheap

Vision models only run on **rare flagged frames**.

### Private

Full video streams are never uploaded.

### Extensible

The heuristics stage can later be replaced by a fine‑tuned neural network without changing the rest of the pipeline.

---

# Hackathon Pitch

**Privacy‑first**
Video stays on device unless distress is detected.

**Real‑world footage**
Designed for overhead residential pool cameras with water distortion.

**Self‑improving**
Uncertain cases automatically build a training dataset.

**Production path**
Heuristics can be swapped for a trained model without changing system architecture.

---

# License

Prototype project created for a hackathon demonstration.
