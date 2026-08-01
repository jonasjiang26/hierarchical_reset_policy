from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import socket
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyzlc
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = (
    REPO_ROOT / "scene_graph" / "configs" / "success_checker_prompt_p2_berry.yaml"
)
DEFAULT_NODE_IP = "141.3.53.25"
DEFAULT_GROUP_NAME = "robot_lab_robotiq_202"
DEFAULT_GROUP_PORT = 7725
DEFAULT_NODE_NAME = "phase_2_success_checker"
DEFAULT_SCENE_GRAPH_SERVICE = "scene_graph"
DEFAULT_ROLLOUT_TOPIC = "roll-out state"
DEFAULT_RESET_TOPIC = "reset state"
DEFAULT_SPATIAL_RELATION_SEQUENCE_TOPIC = "spatial relation sequence"
DEFAULT_RESET_SEQUENCE_TOPIC = "reset sequence"
DEFAULT_RESET_FAILURE_TOPIC = "reset checker state"
DEFAULT_STATIC_CAM_TOPIC = "static_cam"
DEFAULT_SCENE_PROMPT = "strawberry. drawer. plate"
DEFAULT_LLM_URL = "https://ki-toolbox.scc.kit.edu/api/v1/chat/completions"
DEFAULT_MODEL = "kit.minimax-m2.7-229b"
DEFAULT_FRAME_TIMEOUT = 5.0
DEFAULT_MAX_FRAME_AGE = 2.0
DEFAULT_CAMERA_CLOCK_SKEW = 0.25
DEFAULT_LATEST_FRAME_WINDOW = 0.15
DEFAULT_MAX_TOKENS = 512
DEFAULT_LLM_TIMEOUT = 300.0
RESET_SUBSKILLS = (
     "open the lower drawer.",
     "put the strawberry from plate back in drawer.",
     "put the strawberry from table back in drawer.",
     "close the lower drawer."
)


@dataclass
class EvaluationSession:
    kind: str
    objective: str | None
    relations: list[dict[str, list[dict[str, Any]]]] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event)
    relation_event: threading.Event = field(default_factory=threading.Event)
    collector_thread: threading.Thread | None = None
    ending: bool = False


class Phase2SuccessChecker:
    """Evaluate rollout and reset sessions from sampled spatial relations."""

    def __init__(
        self,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        llm_url: str = DEFAULT_LLM_URL,
        model: str = DEFAULT_MODEL,
        group_name: str = DEFAULT_GROUP_NAME,
        static_cam_topic: str = DEFAULT_STATIC_CAM_TOPIC,
        scene_graph_service: str = DEFAULT_SCENE_GRAPH_SERVICE,
        scene_prompt: str = DEFAULT_SCENE_PROMPT,
        sample_interval: float = 1.0,
        service_call_timeout: float = 10.0,
        scene_request_timeout: float = 60.0,
        frame_timeout: float = DEFAULT_FRAME_TIMEOUT,
        max_frame_age: float = DEFAULT_MAX_FRAME_AGE,
        camera_clock_skew: float = DEFAULT_CAMERA_CLOCK_SKEW,
        latest_frame_window: float = DEFAULT_LATEST_FRAME_WINDOW,
        llm_timeout: float = DEFAULT_LLM_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        jpeg_quality: int = 90,
        input_color_order: str = "bgr",
        api_key: str | None = None,
        reset_sequence_publisher: Any | None = None,
        reset_failure_publisher: Any | None = None,
    ) -> None:
        self.prompts = self._load_prompts(Path(prompt_path))
        self.llm_url = llm_url
        self.model = model
        self.group_name = group_name
        self.static_cam_topic = static_cam_topic
        self.scene_graph_service = scene_graph_service
        self.scene_prompt = scene_prompt
        self.sample_interval = sample_interval
        self.service_call_timeout = service_call_timeout
        self.scene_request_timeout = scene_request_timeout
        self.frame_timeout = frame_timeout
        self.max_frame_age = max_frame_age
        self.camera_clock_skew = camera_clock_skew
        self.latest_frame_window = latest_frame_window
        self.llm_timeout = llm_timeout
        self.max_tokens = max_tokens
        self.jpeg_quality = jpeg_quality
        self.input_color_order = input_color_order
        if not api_key or not api_key.strip():
            raise ValueError("An API key is required for the LLM server.")
        self.api_key = api_key.strip()
        self.reset_sequence_publisher = reset_sequence_publisher
        self.reset_failure_publisher = reset_failure_publisher

        self._session: EvaluationSession | None = None
        self._session_lock = threading.Lock()
        self._subscription_started_at = time.time()
        self._latest_static_frame: Mapping[str, Any] | None = None
        self._latest_frame_age_at_receive: float | None = None
        self._frame_event = threading.Event()
        self._frame_lock = threading.Lock()

    def static_cam_callback(self, frame: Mapping[str, Any]) -> None:
        """Store the newest fresh static-camera frame for the next LLM request."""
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
        if capture_time < self._subscription_started_at - self.camera_clock_skew:
            pyzlc.warning(
                "Ignoring pre-subscription static-camera frame: "
                f"timestamp={capture_time:.6f}, "
                f"subscription_timestamp={self._subscription_started_at:.6f}, "
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

        with self._frame_lock:
            current_timestamp = self._capture_time(self._latest_static_frame)
            if current_timestamp is not None and capture_time <= current_timestamp:
                return None
            self._latest_static_frame = frame
            self._latest_frame_age_at_receive = frame_age
            self._frame_event.set()
        return None

    def rollout_state_callback(self, message: Any) -> None:
        print(f"roll-out state message: {message}", flush=True)
        signal = self._signal_text(message).lower()
        if signal in {"roll-out starts", "rollout starts"}:
            self._start_session("rollout", None)
        elif signal in {"roll-out ends", "rollout ends"}:
            self._end_session("rollout", None)
        else:
            pyzlc.warning(f"Ignoring unknown roll-out state signal: {message!r}")

    def reset_state_callback(self, message: Any) -> None:
        signal = self._reset_signal_text(message)
        if signal.strip().lower() == "reset subskill failed":
            return
        match = re.fullmatch(r"(.+?)\s+(starts|ends)", signal, flags=re.IGNORECASE)
        if not match:
            pyzlc.warning(f"Ignoring unknown reset state signal: {message!r}")
            return

        subskill = match.group(1).strip().strip("[]'\"").strip()
        if not subskill:
            pyzlc.warning(f"Ignoring reset signal without a subskill: {message!r}")
            return

        event = match.group(2).lower()
        if event == "starts":
            self._start_session("reset", subskill)
        else:
            self._end_session("reset", subskill)

    def _start_session(self, kind: str, objective: str | None) -> None:
        session = EvaluationSession(kind=kind, objective=objective)
        with self._session_lock:
            previous = self._session
            if previous is not None:
                previous.stop_event.set()
                pyzlc.warning(
                    "Replacing unfinished evaluation session "
                    f"{previous.kind!r} with {kind!r}."
                )
            self._session = session

        objective_suffix = f" for {objective!r}" if objective else ""
        pyzlc.info(
            f"Started {kind} relation collection{objective_suffix}; waiting for "
            "event-end spatial relation sequence."
        )

    def _end_session(self, kind: str, objective: str | None) -> None:
        with self._session_lock:
            session = self._session
            if session is None or session.kind != kind:
                pyzlc.warning(f"Ignoring {kind} end signal without a matching start.")
                return
            if kind == "reset" and not self._same_subskill(session.objective, objective):
                pyzlc.warning(
                    f"Ignoring reset end for {objective!r}; active subskill is "
                    f"{session.objective!r}."
                )
                return
            if session.ending:
                pyzlc.warning(f"Ignoring duplicate {kind} end signal.")
                return
            session.stop_event.set()
            session.ending = True

        threading.Thread(
            target=self._finish_session_after_relation_sequence,
            args=(session,),
            name=f"phase2-{kind}-evaluation",
            daemon=True,
        ).start()

    def spatial_relation_sequence_callback(self, message: Any) -> None:
        try:
            payload = self._spatial_relation_sequence_payload(message)
            payload_kind = self._relation_payload_kind(payload)
            relation_sequence = self._relation_sequence(payload)
        except Exception as exc:
            pyzlc.warning(f"Ignoring invalid spatial relation sequence: {exc}")
            return None

        with self._session_lock:
            session = self._session
            if session is None:
                pyzlc.warning(
                    "Received spatial relation sequence without an active "
                    "Phase-2 session."
                )
                return None
            if session.kind != payload_kind:
                pyzlc.warning(
                    f"Ignoring {payload_kind!r} relation sequence while active "
                    f"session is {session.kind!r}."
                )
                return None
            session.relations = relation_sequence
            session.relation_event.set()
        pyzlc.info(
            f"Received {payload_kind} spatial relation sequence with "
            f"{len(relation_sequence)} sample(s)."
        )
        return None

    def _finish_session_after_relation_sequence(self, session: EvaluationSession) -> None:
        if not session.relation_event.wait(timeout=self.scene_request_timeout):
            pyzlc.warning(
                f"Timed out waiting {self.scene_request_timeout:.1f}s for "
                f"{session.kind} spatial relation sequence; sending available "
                f"{len(session.relations)} sample(s) to the LLM."
            )
        with self._session_lock:
            if self._session is session:
                self._session = None
        self._finish_session(session)

    def _spatial_relation_sequence_payload(self, message: Any) -> Mapping[str, Any]:
        if isinstance(message, Mapping):
            return message
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        if isinstance(message, str):
            payload = json.loads(message)
            if isinstance(payload, Mapping):
                return payload
        raise TypeError(
            "spatial relation sequence message must be a mapping or JSON mapping"
        )

    def _relation_payload_kind(self, payload: Mapping[str, Any]) -> str:
        raw_kind = (
            payload.get("kind")
            or payload.get("state")
            or payload.get("event")
            or payload.get("session_kind")
        )
        kind = self._signal_text(raw_kind).lower().replace("-", "")
        if kind in {"rollout", "rolloutstate"}:
            return "rollout"
        if kind == "reset":
            return "reset"
        raise ValueError(f"Unknown relation sequence kind/state: {raw_kind!r}")

    def _relation_sequence(
        self,
        payload: Mapping[str, Any],
    ) -> list[dict[str, list[dict[str, Any]]]]:
        raw_sequence = (
            payload.get("relation_sequence")
            or payload.get("spatial_relation_sequence")
            or payload.get("relations_sequence")
        )
        if raw_sequence is None:
            raw_sequence = payload.get("relations")
        if not isinstance(raw_sequence, list):
            raise TypeError("relation sequence payload must contain a list")

        relation_sequence: list[dict[str, list[dict[str, Any]]]] = []
        for index, sample in enumerate(raw_sequence):
            if not isinstance(sample, Mapping):
                raise TypeError(f"relation sample {index} is not a mapping")
            relations = sample.get("relations")
            if not isinstance(relations, list):
                raise TypeError(f"relation sample {index} has no relations list")
            relation_sequence.append({"relations": relations})
        return relation_sequence

    def _collect_relations(self, session: EvaluationSession) -> None:
        request_id: str | None = None
        request_started = 0.0
        while not session.stop_event.is_set():
            cycle_started = time.monotonic()
            try:
                if request_id is None:
                    request_id = str(uuid.uuid4())
                    request_started = cycle_started
                relations = self._request_spatial_relations(request_id)
                if relations is not None:
                    with self._session_lock:
                        if self._session is session and not session.stop_event.is_set():
                            relation_sample = {"relations": relations}
                            session.relations.append(relation_sample)
                            pyzlc.info(
                                f"Stored relation sample {len(session.relations)}: "
                                f"{relation_sample}"
                            )
                    request_id = None
                elif cycle_started - request_started >= self.scene_request_timeout:
                    pyzlc.error(
                        f"Scene-graph request {request_id} did not complete within "
                        f"{self.scene_request_timeout:.1f}s. Starting a new request."
                    )
                    request_id = None
            except Exception as exc:
                pyzlc.error(f"Scene-graph sample failed: {exc}")
                pyzlc.error(traceback.format_exc())
                request_id = None

            remaining = self.sample_interval - (time.monotonic() - cycle_started)
            session.stop_event.wait(max(0.0, remaining))

    def _request_spatial_relations(
        self, request_id: str
    ) -> list[dict[str, Any]] | None:
        request = {"request_id": request_id, "prompt": self.scene_prompt}
        response = self._request_fn(
            self.scene_graph_service,
            request,
            timeout=self.service_call_timeout,
            group_name=self.group_name,
        )
        if not response or not response.get("scene_graph_complete"):
            return None
        spatial_relation = response.get("spatial_relation")
        if not isinstance(spatial_relation, Mapping):
            raise TypeError("Scene-graph response has no spatial_relation mapping.")
        relations = spatial_relation.get("relations")
        if not isinstance(relations, list):
            raise TypeError("Scene-graph spatial_relation has no relations list.")
        return relations

    def _finish_session(self, session: EvaluationSession) -> None:
        try:
            if session.collector_thread is not None:
                session.collector_thread.join()
            relation_sequence = list(session.relations)
            pyzlc.info(
                f"{session.kind.capitalize()} ended with "
                f"{len(relation_sequence)} relation samples. Sending them to the LLM."
            )
            image_url = self._build_image()
            messages = self._build_messages(session, relation_sequence, image_url)
            response = self._send_chat_completion(messages)
            print("\nLLM response:", flush=True)
            print(response, flush=True)
            pyzlc.info(f"Phase-2 {session.kind} LLM response: {response}")
            if session.kind == "rollout":
                self._publish_reset_sequence(response)
            else:
                reset_outcome = self._reset_outcome(response)
                if reset_outcome == "reset subskill failed":
                    self._publish_reset_failure()
                elif reset_outcome == "reset subskill succeeded":
                    self._publish_reset_success()
                else:
                    raise ValueError(
                        "The reset LLM response did not contain exactly one "
                        f"recognized reset outcome: {response!r}"
                    )
        except Exception as exc:
            pyzlc.error(f"Phase-2 {session.kind} evaluation failed: {exc}")
            pyzlc.error(traceback.format_exc())

    @staticmethod
    def _reset_outcome(llm_response: str) -> str | None:
        response_text = llm_response.strip().lower()
        outcomes = ("reset subskill succeeded", "reset subskill failed")
        matches = [outcome for outcome in outcomes if outcome in response_text]
        if len(matches) == 1:
            return matches[0]
        return None

    def _publish_reset_success(self) -> None:
        self._publish_reset_checker_state("reset subskill succeeded")

    def _publish_reset_failure(self) -> None:
        self._publish_reset_checker_state("reset subskill failed")

    def _publish_reset_checker_state(self, message: str) -> None:
        if self.reset_failure_publisher is None:
            raise RuntimeError("Reset-checker-state publisher is not configured.")
        self.reset_failure_publisher.publish(message)
        print(f"Published reset checker state: {message}", flush=True)
        pyzlc.info(f"Published reset checker state: {message}")

    def _publish_reset_sequence(self, llm_response: str) -> None:
        if self.reset_sequence_publisher is None:
            raise RuntimeError("Reset-sequence publisher is not configured.")
        sequence = self._extract_reset_sequence(llm_response)
        if not sequence:
            raise ValueError(
                "The rollout LLM response did not contain a recognized reset subskill."
            )
        self.reset_sequence_publisher.publish(sequence)
        print(f"Published reset sequence: {sequence}", flush=True)
        pyzlc.info(f"Published reset sequence: {sequence}")

    @staticmethod
    def _extract_reset_sequence(llm_response: str) -> list[str]:
        response_text = llm_response.lower()
        matches: list[tuple[int, str]] = []
        for subskill in RESET_SUBSKILLS:
            search_text = subskill.rstrip(".").lower()
            start = 0
            while True:
                index = response_text.find(search_text, start)
                if index < 0:
                    break
                matches.append((index, subskill))
                start = index + len(search_text)
        matches.sort(key=lambda match: match[0])
        return [subskill for _, subskill in matches]

    def _build_messages(
        self,
        session: EvaluationSession,
        relation_sequence: list[dict[str, list[dict[str, Any]]]],
        image_url: str,
    ) -> list[dict[str, Any]]:
        query_key = "roll-out_query" if session.kind == "rollout" else "reset_query"
        objective_text = ""
        if session.kind == "reset":
            objective_text = f"\n\nReset subskill just executed:\n{session.objective}"
        sequence_text = json.dumps(relation_sequence, indent=2, ensure_ascii=False)
        user_prompt = (
            f"{self._prompt_text(query_key)}{objective_text}\n\n"
            f"{sequence_text}"
        )
        return [
            {"role": "system", "content": self._prompt_text("system")},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    # {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

    def _build_image(self) -> str:
        frame = self._get_current_static_frame()
        capture_time = self._capture_time(frame)
        current_age = (
            time.time() - capture_time
            if capture_time is not None
            else float("nan")
        )
        pyzlc.info(
            "Using current static-camera frame for Phase-2 LLM request: "
            f"timestamp={frame.get('timestamp')!r}, "
            f"size={frame.get('width')}x{frame.get('height')}, "
            f"age_at_receive={self._latest_frame_age_at_receive}, "
            f"current_age={current_age:.3f}s"
        )
        return self._frame_to_jpeg_data_url(frame)

    def _get_current_static_frame(self) -> Mapping[str, Any]:
        self._frame_event.clear()
        pyzlc.info(f"Waiting for a current frame from {self.static_cam_topic!r}.")
        if not self._frame_event.wait(timeout=self.frame_timeout):
            raise TimeoutError(
                f"No fresh static-camera frame was received from "
                f"{self.static_cam_topic!r} within {self.frame_timeout:.1f}s."
            )

        # Keep receiving briefly after the first acceptable frame. The callback
        # overwrites the slot with increasing camera timestamps, so this returns
        # the newest frame observed in the window.
        time.sleep(min(self.latest_frame_window, self.frame_timeout))

        with self._frame_lock:
            frame = self._latest_static_frame

        if frame is None:
            raise RuntimeError("The static-camera callback completed without a frame.")
        return frame

    def _frame_to_jpeg_data_url(self, frame: Mapping[str, Any]) -> str:
        width = int(frame["width"])
        height = int(frame["height"])
        channels = int(frame.get("channels", 3))
        rgb_data = frame["rgb_data"]

        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(
                f"jpeg_quality must be between 1 and 100: {self.jpeg_quality}"
            )

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

            if self.input_color_order == "bgr":
                red, green, blue = image.convert("RGB").split()
                pil_image = Image.merge("RGB", (blue, green, red))
            else:
                pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported camera channel count: {channels}")

        encoded = io.BytesIO()
        pil_image.save(encoded, format="JPEG", quality=self.jpeg_quality)
        image_b64 = base64.b64encode(encoded.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{image_b64}"

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

    def _send_chat_completion(self, messages: list[dict[str, Any]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        pyzlc.info(
            f"Sending LLM request to {self.llm_url!r} using model "
            f"{self.model!r} with max_tokens={self.max_tokens}."
        )
        request = urllib.request.Request(
            self.llm_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.llm_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {body}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                f"Timed out after {self.llm_timeout:.1f}s waiting for the LLM "
                "response. Increase --llm-timeout or use a shorter prompt."
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach LLM endpoint {self.llm_url!r}: {exc}") from exc

        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response: {data}") from exc
        if not isinstance(choice, Mapping) or not isinstance(message, Mapping):
            raise RuntimeError(f"Unexpected LLM response: {data}")

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

        finish_reason = choice.get("finish_reason")
        reasoning = message.get("reasoning_content")
        reasoning_length = len(reasoning) if isinstance(reasoning, str) else 0
        if finish_reason == "length":
            raise RuntimeError(
                "The LLM exhausted its output budget before producing a final "
                f"answer (max_tokens={self.max_tokens}, "
                f"reasoning_chars={reasoning_length}). Increase --max-tokens "
                f"(current default is {DEFAULT_MAX_TOKENS}; try --max-tokens 8192)."
            )
        raise RuntimeError(
            "The LLM returned no final message content "
            f"(finish_reason={finish_reason!r}, "
            f"reasoning_chars={reasoning_length}, "
            f"message_fields={sorted(map(str, message.keys()))})."
        )

    @staticmethod
    def _signal_text(message: Any) -> str:
        if isinstance(message, Mapping):
            for key in ("state", "status", "message", "data", "signal"):
                if key in message:
                    return Phase2SuccessChecker._signal_text(message[key])
        if isinstance(message, (list, tuple)):
            message = " ".join(str(part) for part in message)
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        return str(message).strip().strip("[]").strip()

    @staticmethod
    def _reset_signal_text(message: Any) -> str:
        if isinstance(message, Mapping):
            subskill = message.get("subskill") or message.get("skill")
            event = message.get("state") or message.get("status") or message.get("event")
            if subskill is not None and event is not None:
                return f"{subskill} {event}".strip()
        return Phase2SuccessChecker._signal_text(message)

    @staticmethod
    def _same_subskill(left: str | None, right: str | None) -> bool:
        normalize = lambda value: (value or "").strip().rstrip(".").lower()
        return normalize(left) == normalize(right)

    @staticmethod
    def _load_prompts(prompt_path: Path) -> dict[str, Any]:
        with prompt_path.open("r", encoding="utf-8") as prompt_file:
            prompts = yaml.safe_load(prompt_file)
        if not isinstance(prompts, dict):
            raise TypeError(f"Prompt file {prompt_path} must contain a YAML mapping.")
        missing = [
            key for key in ("system", "roll-out_query", "reset_query")
            if key not in prompts
        ]
        if missing:
            raise KeyError(f"Prompt file {prompt_path} is missing keys: {missing}")
        return prompts

    def _prompt_text(self, key: str) -> str:
        value = self.prompts[key]
        if isinstance(value, Mapping):
            value = value.get("user")
        if not isinstance(value, str):
            raise TypeError(
                f"Prompt entry {key!r} must be a string or contain a string user field."
            )
        return value.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate rollouts and reset subskills from scene-graph sequences."
    )
    parser.add_argument("--node-ip", default=DEFAULT_NODE_IP)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--group-port", type=int, default=DEFAULT_GROUP_PORT)
    parser.add_argument("--node-name", default=DEFAULT_NODE_NAME)
    parser.add_argument("--rollout-topic", default=DEFAULT_ROLLOUT_TOPIC)
    parser.add_argument("--reset-topic", default=DEFAULT_RESET_TOPIC)
    parser.add_argument(
        "--spatial-relation-sequence-topic",
        default=DEFAULT_SPATIAL_RELATION_SEQUENCE_TOPIC,
    )
    parser.add_argument(
        "--reset-sequence-topic", default=DEFAULT_RESET_SEQUENCE_TOPIC
    )
    parser.add_argument(
        "--reset-failure-topic", default=DEFAULT_RESET_FAILURE_TOPIC
    )
    parser.add_argument("--static-cam-topic", default=DEFAULT_STATIC_CAM_TOPIC)
    parser.add_argument("--scene-graph-service", default=DEFAULT_SCENE_GRAPH_SERVICE)
    parser.add_argument("--scene-prompt", default=DEFAULT_SCENE_PROMPT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--service-call-timeout", type=float, default=10.0)
    parser.add_argument("--scene-request-timeout", type=float, default=60.0)
    parser.add_argument("--frame-timeout", type=float, default=DEFAULT_FRAME_TIMEOUT)
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
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-timeout", type=float, default=DEFAULT_LLM_TIMEOUT)
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

    for option in (
        "sample_interval",
        "service_call_timeout",
        "scene_request_timeout",
        "frame_timeout",
        "llm_timeout",
    ):
        if getattr(args, option) <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if args.max_frame_age <= 0:
        parser.error("--max-frame-age must be positive")
    if args.camera_clock_skew < 0:
        parser.error("--camera-clock-skew cannot be negative")
    if args.latest_frame_window < 0:
        parser.error("--latest-frame-window cannot be negative")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if not args.api_key or not args.api_key.strip():
        parser.error(
            "an API key is required; set LLM_API_KEY (or OPENAI_API_KEY), "
            "or pass --api-key"
        )
    api_key = args.api_key.strip()

    pyzlc.init(
        args.node_name,
        args.node_ip,
        args.group_name,
        group_port=args.group_port,
    )
    reset_sequence_publisher = pyzlc.Publisher(
        args.reset_sequence_topic, args.group_name
    )
    reset_failure_publisher = pyzlc.Publisher(
        args.reset_failure_topic, args.group_name
    )
    checker = Phase2SuccessChecker(
        prompt_path=args.prompt_path,
        llm_url=args.llm_url,
        model=args.model,
        group_name=args.group_name,
        static_cam_topic=args.static_cam_topic,
        scene_graph_service=args.scene_graph_service,
        scene_prompt=args.scene_prompt,
        sample_interval=args.sample_interval,
        service_call_timeout=args.service_call_timeout,
        scene_request_timeout=args.scene_request_timeout,
        frame_timeout=args.frame_timeout,
        max_frame_age=args.max_frame_age,
        camera_clock_skew=args.camera_clock_skew,
        latest_frame_window=args.latest_frame_window,
        llm_timeout=args.llm_timeout,
        max_tokens=args.max_tokens,
        jpeg_quality=args.jpeg_quality,
        input_color_order=args.input_color_order,
        api_key=api_key,
        reset_sequence_publisher=reset_sequence_publisher,
        reset_failure_publisher=reset_failure_publisher,
    )

    pyzlc.info(f"Forcing {args.static_cam_topic} subscriber to use TCP transport.")
    pyzlc.get_node(args.group_name).subscriber_manager.local_ip = ""
    pyzlc.register_subscriber_handler(
        args.rollout_topic, checker.rollout_state_callback, args.group_name
    )
    pyzlc.register_subscriber_handler(
        args.reset_topic, checker.reset_state_callback, args.group_name
    )
    pyzlc.register_subscriber_handler(
        args.spatial_relation_sequence_topic,
        checker.spatial_relation_sequence_callback,
        args.group_name,
    )
    pyzlc.register_subscriber_handler(
        args.static_cam_topic,
        checker.static_cam_callback,
        args.group_name,
        buffer_size=1,
        conflate=True,
    )
    pyzlc.info(
        f"{args.node_name} subscribed to {args.rollout_topic!r} and "
        f"{args.reset_topic!r}; receiving spatial relation sequences from "
        f"{args.spatial_relation_sequence_topic!r}; receiving images from "
        f"{args.static_cam_topic!r}; publishing rollout reset plans on "
        f"{args.reset_sequence_topic!r}."
    )
    pyzlc.spin()


if __name__ == "__main__":
    main()
