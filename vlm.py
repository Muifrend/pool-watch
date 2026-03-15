from __future__ import annotations

import base64
from dataclasses import dataclass
import time
from typing import Literal, Optional

import cv2
import numpy as np
import requests

from detector import HeuristicResult

PROMPT = """You are a pool safety monitor reviewing a single frame from overhead CCTV footage.

Look at the person in the water. Based only on visible body position and posture, answer:

Is this person drowning or in physical distress?

Rules:
- Respond with exactly one word: YES, NO, or UNSURE
- Use UNSURE if the image is unclear, the person is partially visible, or you cannot determine posture confidently
- Do not explain your answer"""

DEBOUNCE_SECONDS = 10.0


@dataclass(frozen=True)
class Stage2Result:
    verdict: Literal["YES", "NO", "UNSURE"]
    confidence: float
    raw_response: str
    frame_index: int
    timestamp_sec: float
    provider_used: str


class Stage2Confirmer:
    def __init__(self, config_module, clock_fn=None, requester=None):
        self.config = config_module
        self._clock = clock_fn or time.monotonic
        self._requester = requester or requests.post
        self._last_yes_monotonic: Optional[float] = None

    def confirm(
        self, frame: np.ndarray, stage1_result: HeuristicResult
    ) -> Stage2Result:
        provider = self._normalize_provider(
            getattr(self.config, "vlm_provider", "nebius")
        )
        now = self._clock()
        if (
            self._last_yes_monotonic is not None
            and (now - self._last_yes_monotonic) < DEBOUNCE_SECONDS
        ):
            return self._result("NO", "__debounced__", stage1_result, provider)

        try:
            encoded_image = self._encode_frame(frame)
            url, api_key, model = self._provider_settings(provider)
            payload = self._payload(model=model, encoded_image=encoded_image)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            response = self._requester(
                url,
                headers=headers,
                json=payload,
                timeout=getattr(self.config, "vlm_timeout_seconds", 20),
            )
            response.raise_for_status()
            raw_text = self._extract_text(response.json())
            verdict, response_marker = self._parse_verdict(raw_text)

            result = self._result(verdict, response_marker, stage1_result, provider)
            if verdict == "YES":
                self._last_yes_monotonic = now
            return result
        except requests.Timeout as exc:
            print(f"[Stage2] Timeout during VLM call: {type(exc).__name__}: {exc}")
            return self._result("UNSURE", "__timeout__", stage1_result, provider)
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", "unknown")
            print(f"[Stage2] HTTP error {status}: {exc}")
            return self._result("UNSURE", "__exception__", stage1_result, provider)
        except requests.RequestException as exc:
            print(f"[Stage2] Request error: {type(exc).__name__}: {exc}")
            return self._result("UNSURE", "__exception__", stage1_result, provider)
        except Exception as exc:
            print(f"[Stage2] Unexpected error: {type(exc).__name__}: {exc}")
            return self._result("UNSURE", "__exception__", stage1_result, provider)

    def _result(
        self,
        verdict: Literal["YES", "NO", "UNSURE"],
        raw_response: str,
        stage1_result: HeuristicResult,
        provider: str,
    ) -> Stage2Result:
        confidence_map = getattr(self.config, "VERDICT_CONFIDENCE_MAP", {})
        confidence = float(confidence_map.get(verdict, 0.5))
        return Stage2Result(
            verdict=verdict,
            confidence=confidence,
            raw_response=raw_response,
            frame_index=int(getattr(stage1_result, "frame_index", -1)),
            timestamp_sec=float(getattr(stage1_result, "timestamp_sec", -1.0)),
            provider_used=provider,
        )

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        normalized = str(provider).strip().lower()
        if normalized not in {"nebius", "openrouter"}:
            return "nebius"
        return normalized

    def _provider_settings(self, provider: str) -> tuple[str, str, str]:
        if provider == "openrouter":
            return (
                getattr(self.config, "openrouter_base_url"),
                getattr(self.config, "openrouter_api_key"),
                getattr(self.config, "openrouter_model"),
            )
        return (
            getattr(self.config, "nebius_base_url"),
            getattr(self.config, "nebius_api_key"),
            getattr(self.config, "nebius_model"),
        )

    def _encode_frame(self, frame: np.ndarray) -> str:
        configured_size = getattr(self.config, "vlm_frame_size", None)
        image_to_encode = frame
        if configured_size is not None:
            size = int(configured_size)
            if size > 0:
                image_to_encode = cv2.resize(frame, (size, size))

        ok, encoded = cv2.imencode(".jpg", image_to_encode)
        if not ok:
            raise ValueError("Failed to JPEG encode frame")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def _payload(model: str, encoded_image: str) -> dict:
        return {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
        }

    @staticmethod
    def _extract_text(response_json: dict) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            return ""

        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            return " ".join(text_parts)
        return str(content)

    @staticmethod
    def _parse_verdict(raw_text: str) -> tuple[Literal["YES", "NO", "UNSURE"], str]:
        tokens = raw_text.strip().upper().split()
        if tokens:
            verdict = tokens[0]
            if verdict in {"YES", "NO", "UNSURE"}:
                return verdict, raw_text
        return "UNSURE", "__parse_failure__"
