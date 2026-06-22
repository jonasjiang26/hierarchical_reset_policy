from __future__ import annotations

import argparse
import base64
import json
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyzlc
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = "/home/jjiang/jing/hierarchical_reset_policy/scene_graph/configs/mp_success_checker_prompt.yaml"
DEFAULT_NODE_IP = "141.3.53.25"
DEFAULT_GROUP_NAME = "robot_lab_robotiq_202"
DEFAULT_GROUP_PORT = 7725
DEFAULT_STATIC_CAM_TOPIC = "static_cam"
DEFAULT_SERVICE_NAME = "phase_1_success_checker"
DEFAULT_SCENE_GRAPH_SERVICE_NAME = "scene_graph"
DEFAULT_LLM_URL = "http://141.3.54.19:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
_MISSING = object()


class Phase1SuccessChecker:
    """Check phase-1 rollout/reset status from the latest static-camera image."""

    def __init__(
        self,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        llm_url: str = DEFAULT_LLM_URL,
        model: str = DEFAULT_MODEL,
        node_ip: str = DEFAULT_NODE_IP,
        group_name: str = DEFAULT_GROUP_NAME,
        group_port: int = DEFAULT_GROUP_PORT,
        static_cam_topic: str = DEFAULT_STATIC_CAM_TOPIC,
        scene_graph_service_name: str = DEFAULT_SCENE_GRAPH_SERVICE_NAME,
        frame_timeout: float = 5.0,
        scene_graph_timeout: float = 60.0,
        scene_graph_poll_interval: float = 1.0,
        request_timeout: float = 60.0,
        jpeg_quality: int = 90,
        api_key: str | None = None,
        init_pyzlc: bool = True,
    ) -> None:
        self.prompt_path = Path(prompt_path)
        self.prompts = self._load_prompts(self.prompt_path)
        self.llm_url = llm_url
        self.model = model
        self.group_name = group_name
        self.static_cam_topic = static_cam_topic
        self.scene_graph_service_name = scene_graph_service_name
        self.frame_timeout = frame_timeout
        self.scene_graph_timeout = scene_graph_timeout
        self.scene_graph_poll_interval = scene_graph_poll_interval
        self.request_timeout = request_timeout
        self.jpeg_quality = jpeg_quality
        self.api_key = api_key

        self.latest_static_frame: Mapping[str, Any] | None = None
        self._frame_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._request_results: dict[str, dict[str, Any]] = {}
        self._request_threads: dict[str, threading.Thread] = {}

        if init_pyzlc:
            pyzlc.init(DEFAULT_SERVICE_NAME, node_ip, group_name, group_port=group_port)
            pyzlc.info(f"Forcing {static_cam_topic} subscriber to use TCP transport.")
            pyzlc.get_node(group_name).subscriber_manager.local_ip = ""
            pyzlc.register_subscriber_handler(static_cam_topic, self.static_cam_callback, group_name)
            pyzlc.info(f"{DEFAULT_SERVICE_NAME} subscribed to {static_cam_topic}.")

    def static_cam_callback(self, frame: Mapping[str, Any]) -> None:
        """Store the newest static-camera frame, matching the server node's pattern."""
        frame["rgb_data"] = np.ascontiguousarray(np.frombuffer(frame["rgb_data"], dtype=np.uint8).reshape((frame["height"], frame["width"], frame["channels"]))[:, :, ::-1]).tobytes()
        with self._frame_lock:
            self.latest_static_frame = frame
            self._frame_event.set()
        return None

    def check_rollout(self, wait_for_new_frame: bool = True, frame_timeout: float | None = None) -> str:
        """Return the LLM's rollout-state judgment."""
        return self._check(
            "roll-out_query",
            wait_for_new_frame=wait_for_new_frame,
            frame_timeout=frame_timeout,
        )

    def check_reset(self, wait_for_new_frame: bool = True, frame_timeout: float | None = None) -> str:
        """Return the LLM's reset-state judgment."""
        return self._check(
            "reset_query",
            wait_for_new_frame=wait_for_new_frame,
            frame_timeout=frame_timeout,
        )

    def check_state(self, request: Any) -> dict[str, Any]:
        """Service handler for task/reset state requests."""
        try:
            request_kind = self._request_kind(request)
            wait_for_new_frame = self._wait_for_new_frame(request)
            frame_timeout = self._request_frame_timeout(request)
            request_id = self._request_id(request)

            with self._request_lock:
                result = self._request_results.get(request_id)
                if result is not None:
                    return result

                thread = self._request_threads.get(request_id)
                if thread is None or not thread.is_alive():
                    thread = threading.Thread(
                        target=self._run_check_state_request,
                        args=(request_id, request_kind, wait_for_new_frame, frame_timeout),
                        daemon=True,
                    )
                    self._request_threads[request_id] = thread
                    thread.start()

            return {
                "success": False,
                "request_id": request_id,
                "complete": False,
                "state": "",
                "message": "processing",
            }
        except Exception as exc:
            pyzlc.error(f"{DEFAULT_SERVICE_NAME} request failed: {exc}")
            pyzlc.error(traceback.format_exc())
            return {
                "success": False,
                "complete": True,
                "state": "",
                "message": str(exc),
            }

    def _run_check_state_request(
        self,
        request_id: str,
        request_kind: str,
        wait_for_new_frame: bool,
        frame_timeout: float | None,
    ) -> None:
        try:
            if request_kind == "task":
                raw_response = self.check_rollout(
                    wait_for_new_frame=wait_for_new_frame,
                    frame_timeout=frame_timeout,
                )
            else:
                raw_response = self.check_reset(
                    wait_for_new_frame=wait_for_new_frame,
                    frame_timeout=frame_timeout,
                )

            state = self._normalize_state_response(raw_response, request_kind)
            result = {
                "success": True,
                "request_id": request_id,
                "complete": True,
                "state": state,
                "raw_response": raw_response,
            }
        except Exception as exc:
            pyzlc.error(f"{DEFAULT_SERVICE_NAME} background request failed: {exc}")
            pyzlc.error(traceback.format_exc())
            result = {
                "success": False,
                "request_id": request_id,
                "complete": True,
                "state": "",
                "message": str(exc),
            }

        with self._request_lock:
            self._request_results[request_id] = result
            self._request_threads.pop(request_id, None)

    def _check(
        self,
        query_key: str,
        wait_for_new_frame: bool,
        frame_timeout: float | None = None,
    ) -> str:
        frame = self._get_static_frame(
            wait_for_new_frame=wait_for_new_frame,
            frame_timeout=frame_timeout,
        )
        spatial_relation = self._get_current_spatial_relation()
        image_url = self._frame_to_jpeg_data_url(frame)
        messages = self._build_messages(query_key, image_url, spatial_relation)
        return self._send_chat_completion(messages)

    def _get_static_frame(
        self,
        wait_for_new_frame: bool,
        frame_timeout: float | None = None,
    ) -> Mapping[str, Any]:
        timeout = self.frame_timeout if frame_timeout is None else frame_timeout

        with self._frame_lock:
            frame = self.latest_static_frame

        if frame is not None and not wait_for_new_frame:
            return frame

        if wait_for_new_frame:
            self._frame_event.clear()

        if frame is None:
            pyzlc.info(f"Waiting for first frame from {self.static_cam_topic}.")
        else:
            pyzlc.info(f"Waiting for new frame from {self.static_cam_topic}.")

        if frame is None or wait_for_new_frame:
            if not self._frame_event.wait(timeout=timeout):
                raise TimeoutError(
                    f"No frame received from {self.static_cam_topic!r} within "
                    f"{timeout:.1f}s."
                )
            with self._frame_lock:
                frame = self.latest_static_frame

        if frame is None:
            raise RuntimeError(f"No frame is available from {self.static_cam_topic!r}.")
        return frame

    def _build_messages(
        self,
        query_key: str,
        image_url: str,
        spatial_relation: str,
    ) -> list[dict[str, Any]]:
        system_prompt = self._prompt_text("system")
        user_prompt = self._prompt_text(query_key)
        user_prompt = self._fill_current_spatial_relation(user_prompt, spatial_relation)

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            },
        ]

    def _send_chat_completion(self, messages: list[dict[str, Any]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 64,
            "temperature": 0,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            self.llm_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                response_body = response.read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach LLM endpoint {self.llm_url!r}: {exc}") from exc

        data = json.loads(response_body.decode("utf-8"))
        pyzlc.info(f"LLM response data: {data}")
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response: {data}") from exc

    def _get_current_spatial_relation(self) -> str:
        prompt = self._prompt_text("prompt")
        request_id = str(uuid.uuid4())
        request = {
            "request_id": request_id,
            "prompt": prompt,
        }
        request_fn = getattr(pyzlc, "call", None) or getattr(pyzlc, "zlc_request")
        deadline = time.monotonic() + self.scene_graph_timeout

        pyzlc.info(
            f"Requesting spatial relation from {self.scene_graph_service_name}: "
            f"request_id={request_id}, prompt={prompt!r}"
        )
        while True:
            response = request_fn(
                self.scene_graph_service_name,
                request,
                timeout=self.scene_graph_timeout,
                group_name=self.group_name,
            )
            if response and response.get("scene_graph_complete"):
                spatial_relation = self._format_spatial_relation(
                    response.get("spatial_relation", "")
                )
                pyzlc.info(f"Current spatial relation:\n{spatial_relation}")
                return spatial_relation

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Scene graph did not complete within {self.scene_graph_timeout:.1f}s."
                )
            time.sleep(self.scene_graph_poll_interval)

    def _format_spatial_relation(self, spatial_relation: Any) -> str:
        if spatial_relation is None:
            return ""
        if isinstance(spatial_relation, str):
            return spatial_relation.strip()
        return json.dumps(spatial_relation, ensure_ascii=False)

    def _fill_current_spatial_relation(self, prompt: str, spatial_relation: str) -> str:
        if "{current_spatial_relation}" in prompt:
            return prompt.format(current_spatial_relation=spatial_relation)

        return prompt.replace(
            "current spatial relation: ''",
            f"current spatial relation: '{spatial_relation}'",
        )

    def _frame_to_jpeg_data_url(self, frame: Any) -> str:
        image = self._frame_to_rgb_image(frame)
        if image.ndim == 2:
            encode_image = image
        else:
            encode_image = np.ascontiguousarray(image[:, :, ::-1])

        pyzlc.info(
            "Encoding static-camera frame as JPEG: "
            f"shape={encode_image.shape}, dtype={encode_image.dtype}, "
            f"contiguous={encode_image.flags.c_contiguous}"
        )

        ok, encoded = cv2.imencode(
            ".jpg",
            encode_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("Failed to encode static-camera frame as JPEG.")

        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{image_b64}"

    def _frame_to_rgb_image(self, frame: Any) -> np.ndarray:
        width = int(self._frame_get(frame, "width"))
        height = int(self._frame_get(frame, "height"))
        channels = int(self._frame_get(frame, "channels", 3))
        rgb_data = self._frame_get(frame, "rgb_data")

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid static-camera frame size: width={width}, height={height}")
        if channels <= 0:
            raise ValueError(f"Invalid static-camera channel count: {channels}")

        image = self._rgb_data_to_uint8_array(rgb_data, height, width, channels)
        if channels == 1:
            return np.ascontiguousarray(image[:, :, 0])
        if channels < 3:
            raise ValueError(f"Expected 1, 3, or 4 image channels. Got: {channels}")

        return np.ascontiguousarray(image[:, :, :3])

    def _frame_get(self, frame: Any, key: str, default: Any = _MISSING) -> Any:
        if isinstance(frame, Mapping):
            if default is _MISSING:
                return frame[key]
            return frame.get(key, default)

        try:
            return frame[key]
        except (KeyError, IndexError, TypeError):
            if default is not _MISSING:
                return default
            raise TypeError(
                "Static-camera frame must provide mapping-style access to "
                f"{key!r}. Got frame type: {type(frame).__name__}"
            )

    def _rgb_data_to_uint8_array(
        self,
        rgb_data: Any,
        height: int,
        width: int,
        channels: int,
    ) -> np.ndarray:
        expected_size = height * width * channels

        if isinstance(rgb_data, np.ndarray):
            array = np.asarray(rgb_data, dtype=np.uint8)
        else:
            try:
                array = np.frombuffer(rgb_data, dtype=np.uint8)
            except TypeError:
                array = np.asarray(rgb_data, dtype=np.uint8)

        pyzlc.info(
            "Received static-camera rgb_data: "
            f"type={type(rgb_data).__name__}, array_shape={array.shape}, "
            f"dtype={array.dtype}, size={array.size}, expected_size={expected_size}"
        )

        if array.size != expected_size:
            raise ValueError(
                "Static-camera rgb_data size does not match frame metadata: "
                f"got {array.size} bytes/elements, expected {expected_size} "
                f"for height={height}, width={width}, channels={channels}."
            )

        return np.ascontiguousarray(array.reshape((height, width, channels)), dtype=np.uint8)

    def _prompt_text(self, key: str) -> str:
        value = self.prompts[key]
        if isinstance(value, dict):
            value = value.get("user")
        if not isinstance(value, str):
            raise TypeError(f"Prompt entry {key!r} must be a string or contain a string 'user' field.")
        return value.strip()

    def _request_kind(self, request: Any) -> str:
        if isinstance(request, str):
            request_text = request
        elif isinstance(request, dict):
            request_text = (
                request.get("state")
                or request.get("request")
                or request.get("task_state")
                or request.get("query")
                or ""
            )
        else:
            request_text = ""

        request_text = str(request_text).strip().lower().replace("_", " ")
        if request_text in {"task", "task state", "rollout", "rollout state", "roll-out state"}:
            return "task"
        if request_text in {"reset", "reset state"}:
            return "reset"

        raise ValueError(
            "Request must ask for 'task state' or 'reset state'. "
            f"Got: {request_text!r}"
        )

    def _wait_for_new_frame(self, request: Any) -> bool:
        if isinstance(request, dict) and "wait_for_new_frame" in request:
            return bool(request["wait_for_new_frame"])
        return False

    def _request_frame_timeout(self, request: Any) -> float | None:
        if not isinstance(request, dict) or "frame_timeout" not in request:
            return None

        frame_timeout = float(request["frame_timeout"])
        if frame_timeout <= 0:
            raise ValueError(f"frame_timeout must be positive. Got: {frame_timeout}")
        return frame_timeout

    def _request_id(self, request: Any) -> str:
        if isinstance(request, dict) and request.get("request_id"):
            return str(request["request_id"])
        return str(uuid.uuid4())

    def _normalize_state_response(self, raw_response: str, request_kind: str) -> str:
        text = raw_response.strip().lower()

        if "ongoing" in text:
            status = "ongoing"
        elif "failed" in text or "failure" in text:
            status = "failed"
        elif "success" in text or "succeeded" in text:
            status = "success"
        else:
            raise ValueError(f"Could not parse LLM state response: {raw_response!r}")

        return f"{request_kind} {status}"

    def _load_prompts(self, prompt_path: Path) -> dict[str, Any]:
        with prompt_path.open("r", encoding="utf-8") as prompt_file:
            prompts = yaml.safe_load(prompt_file)

        if not isinstance(prompts, dict):
            raise TypeError(f"Prompt file {prompt_path} must contain a YAML mapping.")

        required_keys = ("prompt", "system", "roll-out_query", "reset_query")
        missing_keys = [key for key in required_keys if key not in prompts]
        if missing_keys:
            raise KeyError(f"Prompt file {prompt_path} is missing keys: {missing_keys}")

        return prompts


SuccessChecker = Phase1SuccessChecker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the phase-1 success checker service node.")
    parser.add_argument("--node-ip", default=DEFAULT_NODE_IP)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--group-port", type=int, default=DEFAULT_GROUP_PORT)
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--static-cam-topic", default=DEFAULT_STATIC_CAM_TOPIC)
    parser.add_argument("--scene-graph-service-name", default=DEFAULT_SCENE_GRAPH_SERVICE_NAME)
    parser.add_argument("--prompt-path", default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument("--scene-graph-timeout", type=float, default=60.0)
    parser.add_argument("--scene-graph-poll-interval", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    args = parser.parse_args()

    checker = Phase1SuccessChecker(
        prompt_path=args.prompt_path,
        llm_url=args.llm_url,
        model=args.model,
        node_ip=args.node_ip,
        group_name=args.group_name,
        group_port=args.group_port,
        static_cam_topic=args.static_cam_topic,
        scene_graph_service_name=args.scene_graph_service_name,
        frame_timeout=args.frame_timeout,
        scene_graph_timeout=args.scene_graph_timeout,
        scene_graph_poll_interval=args.scene_graph_poll_interval,
        request_timeout=args.request_timeout,
    )

    pyzlc.register_service_handler(args.service_name, checker.check_state, args.group_name)
    pyzlc.info(f"{args.service_name} initialized and ready to receive requests.")
    pyzlc.spin()


if __name__ == "__main__":
    main()
