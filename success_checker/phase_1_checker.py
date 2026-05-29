import base64
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pyzlc


DEFAULT_NODE_IP = "141.3.53.25"
DEFAULT_GROUP_NAME = "robot_lab_robotiq_202"
DEFAULT_SERVICE_NAME = "scene_graph"
DEFAULT_GROUP_PORT = 7725
API_KEY = "tgp_v1__DlFmaeunWsMyoCv5NY3pu34SxLOFh5hP0yEk0WzqaM"
DEFAULT_TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_TOGETHER_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

class ChatGPTSuccessChecker:
    """Checks robot roll-out and reset state using scene graph output plus an image."""

    SYSTEM_PROMPT = (
        "You are my robot research assistant. You are now in a robot policy "
        "evaluation loop. After one episode of roll-out, the table scene will "
        "be reset by the robot itself. Your task is to judge whether the "
        "evaluated policy succeeds and whether the reset process is successful. "
        "you will be given the task prompt, the current spatial relation of the "
        "relevant objects, the goal spatial relation, and one current image. "
        "You need to analyze those information and give your judgment. If you "
        "think the evaluated policy roll-out or the reset process is ongoing, "
        "simply reply \"reset ongoing\". If you think they failed, reply "
        "\"task failed\" when evaluating policy or \"reset failed\" when reset "
        "the scene. If you think they succeed, reply \"task succeeded\" or "
        "\"reset succeeded.\""
    )

    ROLL_OUT_STATES = ("reset ongoing", "task failed", "task succeeded")
    RESET_STATES = ("reset ongoing", "reset failed", "reset succeeded")

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        scene_graph_prompt: str = "",
        node_name: str = "success_checker",
        node_ip: str = DEFAULT_NODE_IP,
        group_name: str = DEFAULT_GROUP_NAME,
        group_port: int = DEFAULT_GROUP_PORT,
        service_name: str = DEFAULT_SERVICE_NAME,
        scene_graph_timeout: float = 5.0,
        scene_graph_wait_timeout: float = 20.0,
        scene_graph_poll_interval: float = 0.5,
        openai_timeout: float = 30.0,
        image_detail: str = "low",
        init_pyzlc: bool = True,
    ):
        self.scene_graph_prompt = scene_graph_prompt
        self.service_name = service_name
        self.group_name = group_name
        self.scene_graph_timeout = scene_graph_timeout
        self.scene_graph_wait_timeout = scene_graph_wait_timeout
        self.scene_graph_poll_interval = scene_graph_poll_interval
        self.openai_timeout = openai_timeout
        self.image_detail = image_detail
        env_together_key = os.getenv("TOGETHER_API_KEY")
        env_openai_key = os.getenv("OPENAI_API_KEY")
        candidate_key = api_key or env_together_key or env_openai_key
        inferred_provider = "together" if str(candidate_key or "").startswith("tgp_") else "openai"
        self.provider = (provider or os.getenv("SUCCESS_CHECKER_PROVIDER") or inferred_provider).lower()
        if self.provider == "together":
            self.api_key = api_key or env_together_key or API_KEY
            self.model = model or os.getenv("TOGETHER_SUCCESS_CHECKER_MODEL", DEFAULT_TOGETHER_MODEL)
            self.base_url = os.getenv("TOGETHER_BASE_URL", DEFAULT_TOGETHER_BASE_URL)
        else:
            self.api_key = api_key or env_openai_key
            self.model = model or os.getenv("OPENAI_SUCCESS_CHECKER_MODEL", DEFAULT_OPENAI_MODEL)
            self.base_url = None
        self.last_raw_response = ""

        if init_pyzlc:
            pyzlc.init(node_name, node_ip, group_name, group_port=group_port)

        if not self.api_key:
            raise ValueError(
                f"API key is required for success checker provider {self.provider!r}. "
                "Set TOGETHER_API_KEY or OPENAI_API_KEY."
            )

        self.client = None
        try:
            from openai import OpenAI
        except ModuleNotFoundError:
            pass
        else:
            if self.provider == "together":
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                self.client = OpenAI(api_key=self.api_key)

    def check_roll_out_state(
        self,
        task_prompt: str,
        goal_spatial_relation: str,
        scene_graph_prompt: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Return: reset ongoing, task failed, or task succeeded."""

        scene = self.get_current_scene(scene_graph_prompt, request_id)
        user_prompt = (
            "roll-out is begun. "
            f"Task: '{task_prompt}'\n"
            f"Goal spatial relation: '{goal_spatial_relation}'\n"
            f"current spatial relation: '{scene.get('spatial_relation', '')}'\n"
            "current image: attached\n"
            "tell me the roll-out state."
        )
        raw_response = self._ask_chatgpt(user_prompt, scene.get("static_image"))
        return self._normalize_state(raw_response, self.ROLL_OUT_STATES)

    def check_reset_state(
        self,
        goal_spatial_relation: str,
        scene_graph_prompt: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Return: reset ongoing, reset failed, or reset succeeded."""

        scene = self.get_current_scene(scene_graph_prompt, request_id)
        user_prompt = (
            "Reset is begun. "
            f"Goal spatial relation: '{goal_spatial_relation}'\n"
            f"current spatial relation: '{scene.get('spatial_relation', '')}'\n"
            "current image: attached\n"
            "tell me the reset state."
        )
        raw_response = self._ask_chatgpt(user_prompt, scene.get("static_image"))
        return self._normalize_state(raw_response, self.RESET_STATES)

    def get_current_scene(
        self,
        scene_graph_prompt: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt = scene_graph_prompt or self.scene_graph_prompt
        if not prompt:
            raise ValueError(
                "scene_graph_prompt is required. Example: 'potato. pan. table.'"
            )

        if not pyzlc.wait_for_service(
            self.service_name,
            timeout=self.scene_graph_timeout,
            group_name=self.group_name,
        ):
            raise TimeoutError(f"Service not available: {self.service_name}")

        request_id = request_id or f"success_checker_{uuid.uuid4().hex}"
        deadline = time.time() + self.scene_graph_wait_timeout
        response: Dict[str, Any] = {}

        while time.time() <= deadline:
            response = self._call_scene_graph(prompt, request_id)
            if response.get("static_image") and (
                response.get("scene_graph_complete") or response.get("spatial_relation")
            ):
                return response
            time.sleep(self.scene_graph_poll_interval)

        return response

    def _call_scene_graph(self, prompt: str, request_id: str) -> Dict[str, Any]:
        request_fn = getattr(pyzlc, "call", None) or getattr(pyzlc, "zlc_request")
        response = request_fn(
            self.service_name,
            {
                "request_id": request_id,
                "prompt": prompt,
                "include_static_image": True,
            },
            timeout=self.scene_graph_timeout,
            group_name=self.group_name,
        )
        return response or {}

    def _ask_chatgpt(
        self,
        user_prompt: str,
        static_image: Optional[Dict[str, Any]],
    ) -> str:
        content = [{"type": "input_text", "text": user_prompt}]
        if static_image:
            content.append(
                {
                    "type": "input_image",
                    "image_url": self._frame_to_data_url(static_image),
                    "detail": self.image_detail,
                }
            )

        if self.provider == "together":
            response = self._ask_together_chat(content)
        elif self.client is None:
            response = self._ask_chatgpt_http(content)
        else:
            response = self.client.responses.create(
                model=self.model,
                instructions=self.SYSTEM_PROMPT,
                input=[{"role": "user", "content": content}],
                max_output_tokens=32,
            )
        self.last_raw_response = self._extract_response_text(response)
        return self.last_raw_response

    def _to_chat_content(self, content: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        chat_content = []
        for item in content:
            if item.get("type") == "input_text":
                chat_content.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "input_image":
                chat_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": item.get("image_url", "")},
                    }
                )
        return chat_content

    def _ask_together_chat(self, content: list[Dict[str, Any]]) -> Any:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._to_chat_content(content)},
        ]
        if self.client is not None:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=64,
                temperature=0,
                extra_body={"reasoning": {"enabled": False}},
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 64,
            "temperature": 0,
            "reasoning": {"enabled": False},
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.openai_timeout,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Together API request failed: {exc.code} {body}") from exc

    def _ask_chatgpt_http(self, content: list[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "instructions": self.SYSTEM_PROMPT,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": 32,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.openai_timeout,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API request failed: {exc.code} {body}") from exc

    def _frame_to_data_url(self, frame: Dict[str, Any]) -> str:
        width = int(frame["width"])
        height = int(frame["height"])
        channels = int(frame.get("channels") or 3)
        rgb_data = frame["rgb_data"]

        rgb = np.frombuffer(rgb_data, dtype=np.uint8).reshape(
            (height, width, channels)
        )
        if channels == 1:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        else:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        success, encoded = cv2.imencode(".jpg", bgr)
        if not success:
            raise ValueError("Failed to encode static camera frame as JPEG.")

        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{image_b64}"

    def _extract_response_text(self, response: Any) -> str:
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if isinstance(content, list):
                    return " ".join(
                        item.get("text", "") for item in content if isinstance(item, dict)
                    ).strip()
                return str(content).strip()

            output_text = response.get("output_text")
            if output_text:
                return output_text.strip()

            chunks = []
            for item in response.get("output", []) or []:
                for content in item.get("content", []) or []:
                    text = content.get("text")
                    if text:
                        chunks.append(text)
            return " ".join(chunks).strip()

        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", "") if message is not None else ""
            if isinstance(content, list):
                return " ".join(
                    getattr(item, "text", "") if not isinstance(item, dict) else item.get("text", "")
                    for item in content
                ).strip()
            return str(content).strip()

        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()

        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        return " ".join(chunks).strip()

    def _normalize_state(self, response: str, allowed_states: tuple[str, ...]) -> str:
        normalized = response.strip().lower().strip("\"'.")
        for state in allowed_states:
            if state in normalized:
                return state
        raise ValueError(f"Unexpected ChatGPT response: {response!r}")


SuccessChecker = ChatGPTSuccessChecker


if __name__ == "__main__":
    checker = ChatGPTSuccessChecker(scene_graph_prompt="potato. pan. table.")
    print(
        checker.check_roll_out_state(
            task_prompt="put potato in the pan.",
            goal_spatial_relation="potato in the pan. pan on the table",
        )
    )
