from __future__ import annotations

import argparse
import time
from typing import Any

import pyzlc

from phase_2_checker import (
    DEFAULT_GROUP_NAME,
    DEFAULT_GROUP_PORT,
    DEFAULT_NODE_IP,
    DEFAULT_RESET_FAILURE_TOPIC,
    DEFAULT_RESET_SEQUENCE_TOPIC,
    DEFAULT_RESET_TOPIC,
    DEFAULT_ROLLOUT_TOPIC,
    RESET_SUBSKILLS,
)


DEFAULT_NODE_NAME = "mock_phase_2_topic_sender"


def publish_repeated(
    publisher: Any,
    topic: str,
    message: Any,
    repeat: int,
    interval: float,
) -> None:
    for index in range(repeat):
        publisher.publish(message)
        print(f"Published to {topic!r}: {message!r}", flush=True)
        if index + 1 < repeat:
            time.sleep(interval)


def reset_state_message(subskill: str, event: str, as_dict: bool) -> Any:
    if as_dict:
        return {"subskill": subskill, "state": event}
    return f"{subskill} {event}"


def add_common_topic_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--node-ip", default=DEFAULT_NODE_IP)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--group-port", type=int, default=DEFAULT_GROUP_PORT)
    parser.add_argument("--node-name", default=DEFAULT_NODE_NAME)
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=0.2,
        help="Seconds to wait after initializing pyzlc before publishing.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.2,
        help="Seconds to wait after the final publish before exiting.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Publish the same message this many times.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds between repeated publishes.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish mock Phase-2 task/reset messages on the same pyzlc topics "
            "used by phase_2_checker.py."
        )
    )
    add_common_topic_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    task_parser = subparsers.add_parser(
        "task-status",
        help="Publish a rollout/task lifecycle event to the roll-out state topic.",
    )
    task_parser.add_argument("--topic", default=DEFAULT_ROLLOUT_TOPIC)
    task_parser.add_argument(
        "--event",
        choices=("starts", "ends"),
        default="starts",
        help="Task event to publish as 'roll-out <event>'.",
    )

    reset_parser = subparsers.add_parser(
        "reset-subskill",
        help="Publish a reset subskill lifecycle event to the reset state topic.",
    )
    reset_parser.add_argument("--topic", default=DEFAULT_RESET_TOPIC)
    reset_parser.add_argument(
        "--subskill",
        default=RESET_SUBSKILLS[0],
        help="Reset subskill name, with or without a trailing period.",
    )
    reset_parser.add_argument(
        "--event",
        choices=("starts", "ends"),
        default="starts",
        help="Reset event to publish.",
    )
    reset_parser.add_argument(
        "--as-dict",
        action="store_true",
        help="Publish {'subskill': ..., 'state': ...} instead of a string.",
    )

    sequence_parser = subparsers.add_parser(
        "reset-sequence",
        help="Publish a mock reset sequence exactly like phase_2_checker output.",
    )
    sequence_parser.add_argument("--topic", default=DEFAULT_RESET_SEQUENCE_TOPIC)
    sequence_parser.add_argument(
        "--subskill",
        action="append",
        dest="subskills",
        help=(
            "Reset subskill to include. Repeat for multiple entries. "
            "Defaults to all phase_2_checker RESET_SUBSKILLS."
        ),
    )

    checker_parser = subparsers.add_parser(
        "reset-checker-state",
        help="Publish a mock reset checker outcome like phase_2_checker output.",
    )
    checker_parser.add_argument("--topic", default=DEFAULT_RESET_FAILURE_TOPIC)
    checker_parser.add_argument(
        "--outcome",
        choices=("succeeded", "failed"),
        default="succeeded",
        help="Publishes 'reset subskill <outcome>'.",
    )

    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.group_port <= 0:
        parser.error("--group-port must be positive")
    if args.startup_delay < 0:
        parser.error("--startup-delay cannot be negative")
    if args.settle_time < 0:
        parser.error("--settle-time cannot be negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.interval < 0:
        parser.error("--interval cannot be negative")
    if hasattr(args, "subskill") and not args.subskill.strip():
        parser.error("--subskill cannot be empty")
    if hasattr(args, "subskills") and args.subskills is not None:
        stripped = [subskill.strip() for subskill in args.subskills]
        if any(not subskill for subskill in stripped):
            parser.error("--subskill entries cannot be empty")
        args.subskills = stripped


def message_for_args(args: argparse.Namespace) -> tuple[str, Any]:
    if args.command == "task-status":
        return args.topic, f"roll-out {args.event}"
    if args.command == "reset-subskill":
        subskill = args.subskill.strip()
        return args.topic, reset_state_message(subskill, args.event, args.as_dict)
    if args.command == "reset-sequence":
        sequence = args.subskills if args.subskills is not None else list(RESET_SUBSKILLS)
        return args.topic, sequence
    if args.command == "reset-checker-state":
        return args.topic, f"reset subskill {args.outcome}"
    raise ValueError(f"Unsupported command: {args.command!r}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    topic, message = message_for_args(args)

    pyzlc.init(
        args.node_name,
        args.node_ip,
        args.group_name,
        group_port=args.group_port,
    )
    publisher = pyzlc.Publisher(topic, args.group_name)
    time.sleep(args.startup_delay)

    publish_repeated(
        publisher,
        topic,
        message,
        repeat=args.repeat,
        interval=args.interval,
    )
    time.sleep(args.settle_time)


if __name__ == "__main__":
    main()
