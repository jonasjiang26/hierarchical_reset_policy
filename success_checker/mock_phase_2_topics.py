from __future__ import annotations

import argparse
import time
from typing import Any

import pyzlc


DEFAULT_NODE_IP = "141.3.53.25"
DEFAULT_GROUP_NAME = "robot_lab_robotiq_202"
DEFAULT_GROUP_PORT = 7725
DEFAULT_NODE_NAME = "mock_rollout_state_sender"
DEFAULT_ROLLOUT_TOPIC = "roll-out state"
DEFAULT_RESET_TOPIC = "reset state"
DEFAULT_RESET_LABEL = "mock reset"
ROLLOUT_START_MESSAGE = "roll-out starts"
ROLLOUT_END_MESSAGE = "roll-out ends"


def publish_once(publisher: Any, topic: str, message: str) -> None:
    publisher.publish(message)
    print(f"Published to {topic!r}: {message!r}", flush=True)


def publish_pulse(
    publisher: Any,
    topic: str,
    message: str,
    count: int,
    interval: float,
) -> None:
    for index in range(count):
        publish_once(publisher, topic, message)
        if index + 1 < count:
            time.sleep(interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish mock roll-out lifecycle messages for nodes that listen on "
            "the 'roll-out state' pyzlc topic."
        )
    )
    parser.add_argument("--node-ip", default=DEFAULT_NODE_IP)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--group-port", type=int, default=DEFAULT_GROUP_PORT)
    parser.add_argument("--node-name", default=DEFAULT_NODE_NAME)
    parser.add_argument(
        "--target",
        choices=("rollout", "reset", "both"),
        default="both",
        help="Which state topic to publish mock lifecycle messages to.",
    )
    parser.add_argument("--rollout-topic", default=DEFAULT_ROLLOUT_TOPIC)
    parser.add_argument("--reset-topic", default=DEFAULT_RESET_TOPIC)
    parser.add_argument(
        "--reset-label",
        default=DEFAULT_RESET_LABEL,
        help="Prefix for reset messages; publishes '<label> starts/ends'.",
    )
    parser.add_argument(
        "--mode",
        choices=("start", "end", "cycle"),
        default="cycle",
        help=(
            "start publishes only 'roll-out starts'; end publishes only "
            "'roll-out ends'; cycle publishes start, waits, then publishes end."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Seconds between start and end when --mode cycle is used.",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=0.5,
        help="Seconds to wait after initializing pyzlc before publishing.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.5,
        help="Seconds to wait after the final publish before exiting.",
    )
    parser.add_argument(
        "--pulse-count",
        type=int,
        default=1,
        help="Publish each lifecycle message this many times.",
    )
    parser.add_argument(
        "--pulse-interval",
        type=float,
        default=0.1,
        help="Seconds between repeated lifecycle message publishes.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.group_port <= 0:
        parser.error("--group-port must be positive")
    if args.duration < 0:
        parser.error("--duration cannot be negative")
    if args.startup_delay < 0:
        parser.error("--startup-delay cannot be negative")
    if args.settle_time < 0:
        parser.error("--settle-time cannot be negative")
    if args.pulse_count <= 0:
        parser.error("--pulse-count must be positive")
    if args.pulse_interval < 0:
        parser.error("--pulse-interval cannot be negative")
    if not args.rollout_topic.strip():
        parser.error("--rollout-topic cannot be empty")
    if not args.reset_topic.strip():
        parser.error("--reset-topic cannot be empty")
    if not args.reset_label.strip():
        parser.error("--reset-label cannot be empty")


def lifecycle_messages(args: argparse.Namespace, event: str) -> list[tuple[str, str]]:
    messages = []
    if args.target in {"rollout", "both"}:
        rollout_message = (
            ROLLOUT_START_MESSAGE if event == "starts" else ROLLOUT_END_MESSAGE
        )
        messages.append((args.rollout_topic, rollout_message))
    if args.target in {"reset", "both"}:
        messages.append((args.reset_topic, f"{args.reset_label.strip()} {event}"))
    return messages


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    pyzlc.init(
        args.node_name,
        args.node_ip,
        args.group_name,
        group_port=args.group_port,
    )
    publishers = {}
    for topic, _ in lifecycle_messages(args, "starts") + lifecycle_messages(args, "ends"):
        publishers.setdefault(topic, pyzlc.Publisher(topic, args.group_name))
    time.sleep(args.startup_delay)

    if args.mode in {"start", "cycle"}:
        for topic, message in lifecycle_messages(args, "starts"):
            publish_pulse(
                publishers[topic],
                topic,
                message,
                count=args.pulse_count,
                interval=args.pulse_interval,
            )

    if args.mode == "cycle":
        print(f"Waiting {args.duration:.3f}s before publishing end signal.", flush=True)
        time.sleep(args.duration)

    if args.mode in {"end", "cycle"}:
        for topic, message in lifecycle_messages(args, "ends"):
            publish_pulse(
                publishers[topic],
                topic,
                message,
                count=args.pulse_count,
                interval=args.pulse_interval,
            )

    time.sleep(args.settle_time)


if __name__ == "__main__":
    main()
