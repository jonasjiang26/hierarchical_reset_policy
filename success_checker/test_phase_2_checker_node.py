from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyzlc
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = (
    REPO_ROOT / "scene_graph" / "configs" / "success_checker_prompt_p2_no_img.yaml"
)
DEFAULT_NODE_IP = "141.3.53.25"
DEFAULT_GROUP_NAME = "robot_lab_robotiq_202"
DEFAULT_GROUP_PORT = 7725
DEFAULT_STATIC_CAM_TOPIC = "static_cam"
DEFAULT_LLM_URL = "https://ki-toolbox.scc.kit.edu/api/v1/chat/completions"
DEFAULT_MODEL = "kit.minimax-m2.7-229b"
DEFAULT_MAX_FRAME_AGE = 2.0
DEFAULT_CAMERA_CLOCK_SKEW = 0.25
DEFAULT_LATEST_FRAME_WINDOW = 0.15
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REQUEST_TIMEOUT = 300.0


# Replace this value with the chronological spatial-relation sequence that
# should be evaluated. Keep the entries ordered from oldest to newest.
SPATIAL_RELATION_SEQUENCE: list[dict[str, Any]] = [
    # Example:
        {
        
            "relations": [
                {"relation": "lemon on plate"},
                {"relation": "drawer on table"},
                {"relation": "plate on table"},
            ],
        
    },
        {
        
            "relations": [
                {"relation": "lemon on plate"},
                {"relation": "drawer on table"},
                {"relation": "plate on table"},
            ],
        
    },
            {
        
            "relations": [
                {"relation": "drawer on table"},
                {"relation": "plate on table"},
            ],
        
    },        {
        
            "relations": [
                {"relation": "lemon in drawer"},
                {"relation": "drawer on table"},
                {"relation": "plate on table"},
            ],
        
    },

{
        
            "relations": [
                {"relation": "lemon in drawer"},
                {"relation": "drawer on table"},
                {"relation": "plate on table"},
            ],
        
    },

{
        
            "relations": [
                {"relation": "drawer on table"},
                {"relation": "plate on table"},
            ],
        
    },



]


class StaticCameraFrameReceiver:
    """Keep only a recently captured frame from a ZeroLanCom camera topic."""

    def __init__(
        self,
        not_before: float,
        max_frame_age: float,
        camera_clock_skew: float,
    ) -> None:
        self.frame: Mapping[str, Any] | None = None
        self.frame_age_at_receive: float | None = None
        self.not_before = not_before
        self.max_frame_age = max_frame_age
        self.camera_clock_skew = camera_clock_skew
        self._event = threading.Event()
        self._lock = threading.Lock()

    def callback(self, frame: Mapping[str, Any]) -> None:
        received_at = time.time()
        capture_time = self._capture_time(frame)
        if capture_time is None:
            pyzlc.warning(
                "Ignoring static-camera frame with invalid timestamp: "
                f"{frame.get('timestamp')!r}"
            )
            return None

        frame_age = received_at - capture_time
        if frame_age < -self.camera_clock_skew:
            pyzlc.warning(
                "Ignoring static-camera frame timestamped too far in the future: "
                f"timestamp={capture_time:.6f}, age={frame_age:.3f}s"
            )
            return None
        if capture_time < self.not_before - self.camera_clock_skew:
            pyzlc.warning(
                "Ignoring pre-subscription static-camera frame: "
                f"timestamp={capture_time:.6f}, "
                f"subscription_timestamp={self.not_before:.6f}, "
                f"age={frame_age:.3f}s"
            )
            return None
        if frame_age > self.max_frame_age:
            pyzlc.warning(
                "Ignoring delayed static-camera frame: "
                f"timestamp={capture_time:.6f}, age={frame_age:.3f}s, "
                f"maximum_age={self.max_frame_age:.3f}s"
            )
            return None

        with self._lock:
            current_timestamp = (
                self._capture_time(self.frame) if self.frame is not None else None
            )
            if current_timestamp is not None and capture_time <= current_timestamp:
                return None
            self.frame = frame
            self.frame_age_at_receive = frame_age
            self._event.set()
        return None

    def wait(self, timeout: float, latest_frame_window: float) -> Mapping[str, Any]:
        wait_started = time.monotonic()
        if not self._event.wait(timeout=timeout):
            raise TimeoutError(
                f"No static-camera frame was received within {timeout:.1f}s."
            )

        # Keep receiving briefly after the first acceptable frame. The
        # callback continuously overwrites the slot with increasing camera
        # timestamps, so this returns the newest frame observed in the window.
        elapsed = time.monotonic() - wait_started
        remaining_timeout = max(0.0, timeout - elapsed)
        time.sleep(min(latest_frame_window, remaining_timeout))

        with self._lock:
            if self.frame is None:
                raise RuntimeError("The camera callback completed without a frame.")
            return self.frame

    @staticmethod
    def _capture_time(frame: Mapping[str, Any] | None) -> float | None:
        if frame is None:
            return None
        timestamp = frame.get("timestamp")
        if isinstance(timestamp, bool) or timestamp is None:
            return None
        try:
            raw_capture_time = float(timestamp)
        except (TypeError, ValueError):
            return None

        # Camera publishers may encode Unix time in seconds, milliseconds,
        # microseconds, or nanoseconds. Normalize all four to seconds so they
        # can be compared with time.time().
        for units_per_second in (1.0, 1_000.0, 1_000_000.0, 1_000_000_000.0):
            capture_time = raw_capture_time / units_per_second
            if 1_000_000_000.0 <= capture_time <= 10_000_000_000.0:
                return capture_time
        return None


def load_prompts(prompt_path: Path) -> dict[str, Any]:
    with prompt_path.open("r", encoding="utf-8") as prompt_file:
        prompts = yaml.safe_load(prompt_file)

    if not isinstance(prompts, dict):
        raise TypeError(f"Prompt file {prompt_path} must contain a YAML mapping.")

    required_keys = ("system", "roll-out_query", "reset_query")
    missing = [key for key in required_keys if key not in prompts]
    if missing:
        raise KeyError(f"Prompt file {prompt_path} is missing keys: {missing}")
    return prompts


def prompt_text(prompts: Mapping[str, Any], key: str) -> str:
    value = prompts[key]
    if isinstance(value, Mapping):
        value = value.get("user")
    if not isinstance(value, str):
        raise TypeError(
            f"Prompt entry {key!r} must be a string or contain a string "
            "'user' field."
        )
    return value.strip()


def frame_to_jpeg_data_url(
    frame: Mapping[str, Any],
    jpeg_quality: int,
    input_color_order: str,
) -> str:
    width = int(frame["width"])
    height = int(frame["height"])
    channels = int(frame.get("channels", 3))
    rgb_data = frame["rgb_data"]

    if not 1 <= jpeg_quality <= 100:
        raise ValueError(f"jpeg_quality must be between 1 and 100: {jpeg_quality}")

    try:
        raw_bytes = bytes(rgb_data)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"rgb_data must be bytes-like. Got: {type(rgb_data).__name__}"
        ) from exc

    expected_size = height * width * channels
    if len(raw_bytes) != expected_size:
        raise ValueError(
            f"Camera image has {len(raw_bytes)} bytes, expected {expected_size} "
            f"for shape ({height}, {width}, {channels})."
        )

    if channels == 1:
        pil_image = Image.frombytes("L", (width, height), raw_bytes)
    elif channels >= 3:
        if channels == 3:
            image = Image.frombytes("RGB", (width, height), raw_bytes)
        elif channels == 4:
            image = Image.frombytes("RGBA", (width, height), raw_bytes)
        else:
            image_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
                (height, width, channels)
            )
            image = Image.fromarray(
                np.array(image_array[:, :, :3], dtype=np.uint8, copy=True),
                mode="RGB",
            )

        if input_color_order == "bgr":
            red, green, blue = image.convert("RGB").split()
            pil_image = Image.merge("RGB", (blue, green, red))
        else:
            pil_image = image.convert("RGB")
    else:
        raise ValueError(f"Unsupported camera channel count: {channels}")

    encoded = io.BytesIO()
    pil_image.save(encoded, format="JPEG", quality=jpeg_quality)
    image_b64 = base64.b64encode(encoded.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{image_b64}"


def build_messages(
    prompts: Mapping[str, Any],
    state: str,
    spatial_relation_sequence: list[dict[str, Any]],
    image_url: str,
    reset_skill: str | None = None,
) -> list[dict[str, Any]]:
    query_key = "roll-out_query" if state == "task" else "reset_query"
    relation_text = json.dumps(
        spatial_relation_sequence,
        indent=2,
        ensure_ascii=False,
    )
    reset_skill_text = ""
    if state == "reset" and reset_skill:
        reset_skill_text = (
            "\n\nReset subskill that was just executed:\n"
            f"{reset_skill.strip()}"
        )
    user_text = (
        f"{prompt_text(prompts, query_key)}"
        f"{reset_skill_text}\n\n"
        "Chronological spatial-relation sequence "
        "(oldest entry first, newest entry last):\n"
        f"{relation_text}"
    )

    return [
        {
            "role": "system",
            "content": prompt_text(prompts, "system"),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        },
    ]


def send_chat_completion(
    llm_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: float,
    max_tokens: int,
    api_key: str,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    print(
        f"Sending LLM request to {llm_url!r} using model {model!r} "
        f"with max_tokens={max_tokens}..."
    )
    request = urllib.request.Request(
        llm_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM request failed with HTTP {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach LLM endpoint {llm_url!r}: {exc}") from exc

    try:
        choice = response_data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response: {response_data}") from exc
    if not isinstance(choice, Mapping) or not isinstance(message, Mapping):
        raise RuntimeError(f"Unexpected LLM response: {response_data}")

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    finish_reason = choice.get("finish_reason")
    reasoning = message.get("reasoning_content")
    reasoning_length = len(reasoning) if isinstance(reasoning, str) else 0
    if finish_reason == "length":
        raise RuntimeError(
            "The LLM exhausted its output budget before producing a final "
            f"answer (max_tokens={max_tokens}, reasoning_chars={reasoning_length}). "
            f"Increase --max-tokens (current default is {DEFAULT_MAX_TOKENS}; "
            "try --max-tokens 8192)."
        )
    raise RuntimeError(
        "The LLM returned no final message content "
        f"(finish_reason={finish_reason!r}, "
        f"reasoning_chars={reasoning_length}, "
        f"message_fields={sorted(map(str, message.keys()))})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Receive one static-camera frame and send it with an in-code "
            "spatial-relation sequence to the Phase-2 vision LLM."
        )
    )
    parser.add_argument("--node-ip", default=DEFAULT_NODE_IP)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--group-port", type=int, default=DEFAULT_GROUP_PORT)
    parser.add_argument("--static-cam-topic", default=DEFAULT_STATIC_CAM_TOPIC)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--state", choices=("task", "reset"), default="task")
    parser.add_argument(
        "--reset-skill",
        default=None,
        help=(
            "Reset subskill that just finished; included in the LLM prompt "
            "only when --state reset."
        ),
    )
    parser.add_argument("--frame-timeout", type=float, default=10.0)
    parser.add_argument(
        "--max-frame-age",
        type=float,
        default=DEFAULT_MAX_FRAME_AGE,
        help="Reject camera frames older than this many seconds.",
    )
    parser.add_argument(
        "--camera-clock-skew",
        type=float,
        default=DEFAULT_CAMERA_CLOCK_SKEW,
        help="Allowed camera/server clock difference in seconds.",
    )
    parser.add_argument(
        "--latest-frame-window",
        type=float,
        default=DEFAULT_LATEST_FRAME_WINDOW,
        help=(
            "After the first fresh frame, keep receiving for this many seconds "
            "and use the newest timestamp observed."
        ),
    )
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--input-color-order",
        choices=("bgr", "rgb"),
        default="bgr",
        help="Color order used by rgb_data. DepthAI static_cam publishes BGR.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        help=(
            "Bearer token for the LLM server; defaults to LLM_API_KEY, then "
            "OPENAI_API_KEY."
        ),
    )
    args = parser.parse_args()
    if args.frame_timeout <= 0:
        parser.error("--frame-timeout must be positive")
    if args.max_frame_age <= 0:
        parser.error("--max-frame-age must be positive")
    if args.camera_clock_skew < 0:
        parser.error("--camera-clock-skew cannot be negative")
    if args.latest_frame_window < 0:
        parser.error("--latest-frame-window cannot be negative")
    if args.reset_skill is not None and not args.reset_skill.strip():
        parser.error("--reset-skill cannot be empty")
    if not args.api_key or not args.api_key.strip():
        parser.error(
            "an API key is required; set LLM_API_KEY (or OPENAI_API_KEY), "
            "or pass --api-key"
        )
    api_key = args.api_key.strip()

    prompts = load_prompts(args.prompt_path)

    pyzlc.init(
        "phase_2_checker_llm_test",
        args.node_ip,
        args.group_name,
        group_port=args.group_port,
    )
    subscription_timestamp = time.time()
    receiver = StaticCameraFrameReceiver(
        not_before=subscription_timestamp,
        max_frame_age=args.max_frame_age,
        camera_clock_skew=args.camera_clock_skew,
    )
    pyzlc.info(f"Forcing {args.static_cam_topic} subscriber to use TCP transport.")
    pyzlc.get_node(args.group_name).subscriber_manager.local_ip = ""
    pyzlc.register_subscriber_handler(
        args.static_cam_topic,
        receiver.callback,
        args.group_name,
        buffer_size=1,
        conflate=True,
    )

    print(
        f"Waiting for a fresh frame from {args.static_cam_topic!r} "
        f"(maximum age {args.max_frame_age:.2f}s)..."
    )
    frame = receiver.wait(
        args.frame_timeout,
        latest_frame_window=args.latest_frame_window,
    )
    capture_time = StaticCameraFrameReceiver._capture_time(frame)
    current_age = time.time() - capture_time if capture_time is not None else float("nan")
    print(
        "Received frame: "
        f"timestamp={frame.get('timestamp')}, "
        f"size={frame.get('width')}x{frame.get('height')}, "
        f"age_at_receive={receiver.frame_age_at_receive:.3f}s, "
        f"current_age={current_age:.3f}s"
    )

    image_url = frame_to_jpeg_data_url(
        frame,
        jpeg_quality=args.jpeg_quality,
        input_color_order=args.input_color_order,
    )
    messages = build_messages(
        prompts,
        state=args.state,
        spatial_relation_sequence=SPATIAL_RELATION_SEQUENCE,
        image_url=image_url,
        reset_skill=args.reset_skill,
    )

    reset_skill_suffix = (
        f" for reset skill {args.reset_skill!r}"
        if args.state == "reset" and args.reset_skill
        else ""
    )
    print(
        f"Sending {args.state} evaluation{reset_skill_suffix} with "
        f"{len(SPATIAL_RELATION_SEQUENCE)} spatial-relation entries to "
        f"{args.llm_url!r} using model {args.model!r}..."
    )
    response = send_chat_completion(
        llm_url=args.llm_url,
        model=args.model,
        messages=messages,
        timeout=args.request_timeout,
        max_tokens=args.max_tokens,
        api_key=api_key,
    )
    print("\nLLM response:")
    print(response)


if __name__ == "__main__":
    main()
