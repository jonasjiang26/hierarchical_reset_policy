import argparse
import pprint

import pyzlc

from mp_phase_1_checker import (
    DEFAULT_GROUP_NAME,
    DEFAULT_GROUP_PORT,
    DEFAULT_NODE_IP,
    DEFAULT_SCENE_GRAPH_SERVICE_NAME,
    DEFAULT_SERVICE_NAME,
)


def call_success_checker(service_name, request, timeout, group_name):
    request_fn = getattr(pyzlc, "call", None) or getattr(pyzlc, "zlc_request")
    return request_fn(service_name, request, timeout=timeout, group_name=group_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a request to the MP phase-1 success checker server.")
    parser.add_argument("--node-ip", default=DEFAULT_NODE_IP)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--group-port", type=int, default=DEFAULT_GROUP_PORT)
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--scene-graph-service-name", default=DEFAULT_SCENE_GRAPH_SERVICE_NAME)
    parser.add_argument("--state", choices=("rollout", "reset"), default="rollout")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--frame-timeout",
        type=float,
        default=10.0,
        help="How long the server should wait for a static_cam frame if it needs one.",
    )
    parser.add_argument(
        "--wait-for-new-frame",
        action="store_true",
        help="Ask the server to wait for a new static_cam frame before checking state.",
    )
    args = parser.parse_args()

    request = {
        "state": "roll-out state" if args.state == "rollout" else "reset state",
        "wait_for_new_frame": args.wait_for_new_frame,
        "frame_timeout": args.frame_timeout,
    }

    pyzlc.init("mp_phase_1_success_checker_test_node", args.node_ip, args.group_name, group_port=args.group_port)

    for service_name in (args.scene_graph_service_name, args.service_name):
        pyzlc.info(f"Waiting for service: {service_name}")
        if not pyzlc.wait_for_service(service_name, timeout=args.timeout, group_name=args.group_name):
            raise RuntimeError(f"Service not available: {service_name}")

    print(f"Sending request to {args.service_name}:")
    pprint.pp(request)

    response = call_success_checker(args.service_name, request, args.timeout, args.group_name)
    print("\nResponse:")
    pprint.pp(response)


if __name__ == "__main__":
    main()
