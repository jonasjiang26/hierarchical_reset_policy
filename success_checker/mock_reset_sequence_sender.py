from __future__ import annotations

import argparse
import json
import time
from typing import Any

import pyzlc


DEFAULT_NODE_IP = "141.3.53.25"
DEFAULT_GROUP_NAME = "robot_lab_robotiq_202"
DEFAULT_GROUP_PORT = 7725
DEFAULT_NODE_NAME = "mock_reset_sequence_sender"
DEFAULT_RESET_SEQUENCE_TOPIC = "reset sequence"
DEFAULT_STARTUP_DELAY = 0.5
DEFAULT_SETTLE_TIME = 0.5

PRESET_SEQUENCES = {
    "strawberry-from-plate": [
        # "open the lower drawer.",
        "put strawberry from plate back in the lower drawer.",
        # "close the lower drawer.",
    ],
    "strawberry-from-table": [
        "open the lower drawer.",
        "put the strawberry from table back in drawer.",
        "close the lower drawer.",
    ],
    "lemon-from-plate": [
        "open the lower drawer.",
        "put the lemon from plate back in drawer.",
        "close the lower drawer.",
    ],
    "lemon-from-table": [
        "open the lower drawer.",
        "put the lemon from table back in drawer.",
        "close the lower drawer.",
    ],
    "open-close": [
        "open the lower drawer.",
        "close the lower drawer.",
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish a mock reset subskill sequence for reset-policy nodes that "
            "listen on the 'reset sequence' pyzlc topic."
        )
    )
    parser.add_argument("--node-ip", default=DEFAULT_NODE_IP)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--group-port", type=int, default=DEFAULT_GROUP_PORT)
    parser.add_argument("--node-name", default=DEFAULT_NODE_NAME)
    parser.add_argument("--topic", default=DEFAULT_RESET_SEQUENCE_TOPIC)
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_SEQUENCES),
        default="strawberry-from-plate",
        help="Named mock sequence to publish when --subskill and --json are omitted.",
    )
    parser.add_argument(
        "--subskill",
        action="append",
        default=[],
        help=(
            "Subskill to include in order. Repeat this flag to build a sequence; "
            "overrides --preset."
        ),
    )
    parser.add_argument(
        "--json",
        help=(
            "JSON list of subskill strings to publish; overrides --preset and "
            "--subskill."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to publish the sequence.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between repeated publishes.",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=DEFAULT_STARTUP_DELAY,
        help="Seconds to wait after initializing pyzlc before publishing.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=DEFAULT_SETTLE_TIME,
        help="Seconds to wait after the final publish before exiting.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.group_port <= 0:
        parser.error("--group-port must be positive")
    if not args.topic.strip():
        parser.error("--topic cannot be empty")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.interval < 0:
        parser.error("--interval cannot be negative")
    if args.startup_delay < 0:
        parser.error("--startup-delay cannot be negative")
    if args.settle_time < 0:
        parser.error("--settle-time cannot be negative")


def sequence_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> list[str]:
    if args.json is not None:
        try:
            sequence = json.loads(args.json)
        except json.JSONDecodeError as exc:
            parser.error(f"--json must be a JSON list of strings: {exc}")
        if not isinstance(sequence, list) or not all(
            isinstance(subskill, str) for subskill in sequence
        ):
            parser.error("--json must be a JSON list of strings")
        return normalize_sequence(parser, sequence)

    if args.subskill:
        return normalize_sequence(parser, args.subskill)

    return list(PRESET_SEQUENCES[args.preset])


def normalize_sequence(
    parser: argparse.ArgumentParser,
    sequence: list[str],
) -> list[str]:
    normalized = [subskill.strip() for subskill in sequence if subskill.strip()]
    if not normalized:
        parser.error("reset sequence cannot be empty")
    return [
        subskill if subskill.endswith(".") else f"{subskill}."
        for subskill in normalized
    ]


def publish_sequence(
    publisher: Any,
    topic: str,
    sequence: list[str],
    repeat: int,
    interval: float,
) -> None:
    for index in range(repeat):
        publisher.publish(sequence)
        print(f"Published to {topic!r}: {sequence}", flush=True)
        if index + 1 < repeat:
            time.sleep(interval)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    sequence = sequence_from_args(parser, args)

    pyzlc.init(
        args.node_name,
        args.node_ip,
        args.group_name,
        group_port=args.group_port,
    )
    publisher = pyzlc.Publisher(args.topic, args.group_name)
    time.sleep(args.startup_delay)
    publish_sequence(
        publisher,
        args.topic,
        sequence,
        repeat=args.repeat,
        interval=args.interval,
    )
    time.sleep(args.settle_time)


if __name__ == "__main__":
    main()
