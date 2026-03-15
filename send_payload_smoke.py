from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import requests

import config
from vlm import PROMPT


def resolve_provider_settings(provider: str) -> tuple[str, str, str]:
    provider = provider.strip().lower()
    if provider == "openrouter":
        return config.openrouter_base_url, config.openrouter_api_key, config.openrouter_model
    return config.nebius_base_url, config.nebius_api_key, config.nebius_model


def encode_image_for_vlm(image_path: Path, frame_size: int | None) -> str:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image_to_encode = image
    if frame_size is not None and int(frame_size) > 0:
        size = int(frame_size)
        image_to_encode = cv2.resize(image, (size, size))

    ok, encoded = cv2.imencode(".jpg", image_to_encode)
    if not ok:
        raise ValueError("Failed to encode image as JPEG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def build_payload(model: str, encoded_image: str) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        ],
        "temperature": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send one image payload to configured VLM provider and print response"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default="Screenshot from 2026-03-15 13-30-21.png",
        help="Path to image file (default: detected screenshot filename)",
    )
    parser.add_argument(
        "--provider",
        choices=["nebius", "openrouter"],
        default=config.vlm_provider,
        help="Override provider (default from config.vlm_provider)",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise SystemExit(f"Image file not found: {image_path}")

    url, api_key, model = resolve_provider_settings(args.provider)
    if not api_key:
        raise SystemExit(
            f"Missing API key for provider '{args.provider}'. Check your .env and config.py"
        )

    encoded_image = encode_image_for_vlm(
        image_path=image_path,
        frame_size=getattr(config, "vlm_frame_size", None),
    )
    payload = build_payload(model=model, encoded_image=encoded_image)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"Provider: {args.provider}")
    print(f"URL: {url}")
    print(f"Model: {model}")
    print(f"Image: {image_path}")
    print(f"Base64 length: {len(encoded_image)}")
    payload_preview = json.loads(json.dumps(payload))
    image_url = payload_preview["messages"][0]["content"][1]["image_url"]["url"]
    payload_preview["messages"][0]["content"][1]["image_url"]["url"] = (
        image_url[:96] + f"...<truncated,total_len={len(image_url)}>"
    )
    print("Payload being sent (preview):")
    print(json.dumps(payload_preview, indent=2))

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=int(config.vlm_timeout_seconds),
        )
        print(f"HTTP status: {response.status_code}")
        response.raise_for_status()

        data = response.json()
        print("Response JSON:")
        print(json.dumps(data, indent=2)[:4000])

        # Best-effort extraction for quick eyeballing.
        try:
            content = data["choices"][0]["message"]["content"]
            print("\nExtracted content:")
            print(content)
        except Exception:
            print("\nCould not extract choices[0].message.content; inspect full JSON above.")

    except requests.Timeout as exc:
        print(f"Timeout: {exc}")
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        print(f"HTTP error: {exc}")
        if body:
            print("Response body:")
            print(body[:4000])
    except requests.RequestException as exc:
        print(f"Request error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
