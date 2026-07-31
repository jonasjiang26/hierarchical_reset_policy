from __future__ import annotations

import argparse
import json
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


API_KEY = "sk-6a072a97155d41b48e81bc3a179b6ff7"
LLM_URL = "https://ki-toolbox.scc.kit.edu/api/v1/chat/completions"
MODEL = "kit.minimax-m2.7-229b"
QUICK_CHECK_MAX_TOKENS = 16
FULL_PROMPT_MAX_TOKENS = 8194
TIMEOUT_SECONDS = 180
PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scene_graph"
    / "configs"
    / "success_checker_prompt_p2_pot.yaml"
)

SPATIAL_RELATION_SEQUENCE: list[dict[str, list[dict[str, Any]]]] = [
    {
        "relations": [
            {"relation": "pot on table"},
            {"relation": "carrot in stove"},
            {"relation": "stove on table"},
            {"relation": "lid on table"},
        ]
    },
    {
        "relations": [
            {"relation": "pot on stove"},
            {"relation": "carrot in pot"},
            {"relation": "stove on table"},
            {"relation": "lid on table"},
        ]
    },
]


def load_prompts(prompt_path: Path) -> dict[str, Any]:
    with prompt_path.open("r", encoding="utf-8") as prompt_file:
        prompts = yaml.safe_load(prompt_file)
    if not isinstance(prompts, dict):
        raise TypeError(f"Prompt file must contain a mapping: {prompt_path}")
    return prompts


def prompt_text(prompts: dict[str, Any], key: str) -> str:
    text = prompts.get(key)
    if isinstance(text, dict):
        text = text.get("user")
    if not isinstance(text, str) or not text.strip():
        raise KeyError(f"Prompt file has no non-empty {key!r} entry.")
    return text.strip()


def build_full_prompt_messages(
    prompts: dict[str, Any],
    no_think: bool = False,
) -> list[dict[str, Any]]:
    sequence_text = json.dumps(
        SPATIAL_RELATION_SEQUENCE,
        indent=2,
        ensure_ascii=False,
    )
    user_prompt = f"{prompt_text(prompts, 'roll-out_query')}\n\n{sequence_text}"
    if no_think:
        user_prompt = f"/no_think\n{user_prompt}"
    return [
        {"role": "system", "content": prompt_text(prompts, "system")},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def build_compact_phase2_messages(no_think: bool = False) -> list[dict[str, Any]]:
    sequence_text = json.dumps(
        SPATIAL_RELATION_SEQUENCE,
        indent=2,
        ensure_ascii=False,
    )
    user_prompt = (
        "The robot has finished the roll-out.\n"
        "Task: put the pot on the stove then cook the carrot.\n\n"
        "Use the chronological spatial-relation sequence below. Success requires "
        "evidence that the pot is on the stove and the carrot is in the pot in "
        "the final scene.\n\n"
        "Return only this format, with no explanation:\n"
        "task succeeded|task failed\n"
        "[zero or more reset subskills, one per line, chosen only from:\n"
        "put lid back in place.\n"
        "put carrot back in sink.\n"
        "put pot back in place.]\n\n"
        f"{sequence_text}"
    )
    if no_think:
        user_prompt = f"/no_think\n{user_prompt}"

    return [
        {
            "role": "system",
            "content": "You are a robot-policy evaluation assistant. Answer concisely.",
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        },
    ]


def build_quick_check_messages(no_think: bool = False) -> list[dict[str, str]]:
    content = "Accessibility check. Reply exactly: ok"
    if no_think:
        content = f"/no_think\n{content}"
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def send_chat_completion(
    messages: list[dict[str, Any]],
    llm_url: str,
    model: str,
    max_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    api_key = API_KEY.strip()
    if not api_key:
        raise ValueError("Fill API_KEY at the top of this script before running it.")

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    request = urllib.request.Request(
        llm_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Timed out after {timeout_seconds}s waiting for the LLM response. "
            "Increase --timeout or try a different --model. For the full Phase-2 "
            "prompt, prefer --compact-phase2 when checking basic behavior."
        ) from exc
    except socket.timeout as exc:
        raise RuntimeError(
            f"Timed out after {timeout_seconds}s waiting for the LLM response. "
            "Increase --timeout or try a different --model. For the full Phase-2 "
            "prompt, prefer --compact-phase2 when checking basic behavior."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {llm_url!r}: {exc}") from exc


def response_content(response_data: dict[str, Any], max_tokens: int) -> str:
    choice = response_data["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    finish_reason = choice.get("finish_reason")

    print(f"finish_reason={finish_reason!r}")
    print(f"reasoning_chars={len(reasoning)}")

    if isinstance(content, str) and content.strip():
        return content.strip()
    if finish_reason == "length":
        raise RuntimeError(
            "The LLM exhausted its output budget before producing final content "
            f"(max_tokens={max_tokens}, reasoning_chars={len(reasoning)}). "
            "Increase --max-tokens, or use --no-think with Qwen-style reasoning "
            "models. For the full prompt, try --max-tokens 8192."
        )
    raise RuntimeError(
        "The LLM returned no final content "
        f"(finish_reason={finish_reason!r}, message_fields={sorted(message.keys())})."
    )


def print_messages(messages: list[dict[str, Any]]) -> None:
    print("messages sent to LLM:")
    print(json.dumps(messages, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a small or full Phase-2 test request to the KIT LLM endpoint."
    )
    parser.add_argument(
        "--full-prompt",
        action="store_true",
        help="Send the full Phase-2 rollout prompt instead of a tiny accessibility ping.",
    )
    parser.add_argument(
        "--compact-phase2",
        action="store_true",
        help="Send a short Phase-2-equivalent prompt without the long YAML examples.",
    )
    parser.add_argument("--llm-url", default=LLM_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Prefix the user message with /no_think for Qwen reasoning models.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the exact messages payload before sending it to the LLM.",
    )
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.full_prompt and args.compact_phase2:
        parser.error("--full-prompt and --compact-phase2 are mutually exclusive")

    if args.full_prompt:
        prompts = load_prompts(PROMPT_PATH)
        messages = build_full_prompt_messages(prompts, no_think=args.no_think)
        max_tokens = args.max_tokens or FULL_PROMPT_MAX_TOKENS
        mode = "full Phase-2 prompt"
    elif args.compact_phase2:
        messages = build_compact_phase2_messages(no_think=args.no_think)
        max_tokens = args.max_tokens or 256
        mode = "compact Phase-2 prompt"
    else:
        messages = build_quick_check_messages(no_think=args.no_think)
        max_tokens = args.max_tokens or QUICK_CHECK_MAX_TOKENS
        mode = "quick accessibility ping"

    if max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    print(f"Sending request to {args.llm_url!r}")
    print(
        f"mode={mode!r}, model={args.model!r}, max_tokens={max_tokens}, "
        f"timeout={args.timeout}s, no_think={args.no_think}"
    )
    if args.print_prompt:
        print_messages(messages)
    response_data = send_chat_completion(
        messages,
        llm_url=args.llm_url,
        model=args.model,
        max_tokens=max_tokens,
        timeout_seconds=args.timeout,
    )

    content = response_content(response_data, max_tokens)
    print("content:")
    print(content)


if __name__ == "__main__":
    main()
