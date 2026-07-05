from __future__ import annotations

import argparse
import json
import os
import re
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

import pyzlc
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = (
    REPO_ROOT / "scene_graph" / "configs" / "success_checker_prompt_p2_no_img.yaml"
)
DEFAULT_NODE_IP = "141.3.53.25"
DEFAULT_GROUP_NAME = "robot_lab_robotiq_202"
DEFAULT_GROUP_PORT = 7725
DEFAULT_NODE_NAME = "phase_2_success_checker"
DEFAULT_SCENE_GRAPH_SERVICE = "scene_graph"
DEFAULT_ROLLOUT_TOPIC = "roll-out state"
DEFAULT_RESET_TOPIC = "reset state"
DEFAULT_RESET_SEQUENCE_TOPIC = "reset sequence"
DEFAULT_RESET_FAILURE_TOPIC = "reset checker state"
DEFAULT_SCENE_PROMPT = "drawer. plate. lemon."
DEFAULT_LLM_URL = "https://ki-toolbox.scc.kit.edu/api/v1/chat/completions"
DEFAULT_MODEL = "kit.gpt-oss-120b"
RESET_SUBSKILLS = (
    "open the lower drawer.",
    "put the lemon from lower drawer back on plate.",
    "put the lemon from table back on plate.",
    "close the lower drawer.",
)


@dataclass
class EvaluationSession:
    kind: str
    objective: str | None
    relations: list[dict[str, list[dict[str, Any]]]] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event)
    collector_thread: threading.Thread | None = None


class Phase2SuccessChecker:
    """Evaluate rollout and reset sessions from sampled spatial relations."""

    def __init__(
        self,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        llm_url: str = DEFAULT_LLM_URL,
        model: str = DEFAULT_MODEL,
        group_name: str = DEFAULT_GROUP_NAME,
        scene_graph_service: str = DEFAULT_SCENE_GRAPH_SERVICE,
        scene_prompt: str = DEFAULT_SCENE_PROMPT,
        sample_interval: float = 1.0,
        service_call_timeout: float = 10.0,
        scene_request_timeout: float = 60.0,
        llm_timeout: float = 60.0,
        max_tokens: int = 2048,
        api_key: str | None = None,
        reset_sequence_publisher: Any | None = None,
        reset_failure_publisher: Any | None = None,
    ) -> None:
        self.prompts = self._load_prompts(Path(prompt_path))
        self.llm_url = llm_url
        self.model = model
        self.group_name = group_name
        self.scene_graph_service = scene_graph_service
        self.scene_prompt = scene_prompt
        self.sample_interval = sample_interval
        self.service_call_timeout = service_call_timeout
        self.scene_request_timeout = scene_request_timeout
        self.llm_timeout = llm_timeout
        self.max_tokens = max_tokens
        if not api_key or not api_key.strip():
            raise ValueError("An API key is required for the LLM server.")
        self.api_key = api_key.strip()
        self.reset_sequence_publisher = reset_sequence_publisher
        self.reset_failure_publisher = reset_failure_publisher

        self._session: EvaluationSession | None = None
        self._session_lock = threading.Lock()
        self._request_fn = getattr(pyzlc, "call", None) or getattr(
            pyzlc, "zlc_request"
        )

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

        session.collector_thread = threading.Thread(
            target=self._collect_relations,
            args=(session,),
            name=f"phase2-{kind}-collector",
            daemon=True,
        )
        session.collector_thread.start()
        objective_suffix = f" for {objective!r}" if objective else ""
        pyzlc.info(f"Started {kind} relation collection{objective_suffix}.")

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
            self._session = None
            session.stop_event.set()

        threading.Thread(
            target=self._finish_session,
            args=(session,),
            name=f"phase2-{kind}-evaluation",
            daemon=True,
        ).start()

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
            messages = self._build_messages(session, relation_sequence)
            response = self._send_chat_completion(messages)
            print("\nLLM response:", flush=True)
            print(response, flush=True)
            pyzlc.info(f"Phase-2 {session.kind} LLM response: {response}")
            if session.kind == "rollout":
                self._publish_reset_sequence(response)
            elif response.strip().lower() == "reset subskill failed":
                self._publish_reset_failure()
        except Exception as exc:
            pyzlc.error(f"Phase-2 {session.kind} evaluation failed: {exc}")
            pyzlc.error(traceback.format_exc())

    def _publish_reset_failure(self) -> None:
        if self.reset_failure_publisher is None:
            raise RuntimeError("Reset-failure publisher is not configured.")
        message = "reset subskill failed"
        self.reset_failure_publisher.publish(message)
        print(f"Published reset failure: {message}", flush=True)
        pyzlc.info(f"Published reset failure: {message}")

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
    ) -> list[dict[str, str]]:
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
            {"role": "user", "content": user_prompt},
        ]

    def _send_chat_completion(self, messages: list[dict[str, str]]) -> str:
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
            f"{self.model!r}."
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
                f"reasoning_chars={reasoning_length}). Increase --max-tokens."
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
        "--reset-sequence-topic", default=DEFAULT_RESET_SEQUENCE_TOPIC
    )
    parser.add_argument(
        "--reset-failure-topic", default=DEFAULT_RESET_FAILURE_TOPIC
    )
    parser.add_argument("--scene-graph-service", default=DEFAULT_SCENE_GRAPH_SERVICE)
    parser.add_argument("--scene-prompt", default=DEFAULT_SCENE_PROMPT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--service-call-timeout", type=float, default=10.0)
    parser.add_argument("--scene-request-timeout", type=float, default=60.0)
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
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
        "llm_timeout",
    ):
        if getattr(args, option) <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
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
        scene_graph_service=args.scene_graph_service,
        scene_prompt=args.scene_prompt,
        sample_interval=args.sample_interval,
        service_call_timeout=args.service_call_timeout,
        scene_request_timeout=args.scene_request_timeout,
        llm_timeout=args.llm_timeout,
        max_tokens=args.max_tokens,
        api_key=api_key,
        reset_sequence_publisher=reset_sequence_publisher,
        reset_failure_publisher=reset_failure_publisher,
    )

    pyzlc.register_subscriber_handler(
        args.rollout_topic, checker.rollout_state_callback, args.group_name
    )
    pyzlc.register_subscriber_handler(
        args.reset_topic, checker.reset_state_callback, args.group_name
    )
    pyzlc.info(
        f"{args.node_name} subscribed to {args.rollout_topic!r} and "
        f"{args.reset_topic!r}; sampling {args.scene_graph_service!r} every "
        f"{args.sample_interval:.1f}s and publishing rollout reset plans on "
        f"{args.reset_sequence_topic!r}."
    )
    pyzlc.spin()


if __name__ == "__main__":
    main()
