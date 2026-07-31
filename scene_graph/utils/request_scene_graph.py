import argparse
import json
import time
import uuid

import pyzlc


DEFAULT_NODE_IP = "141.3.53.25"
DEFAULT_GROUP_NAME = "robot_lab_robotiq_202"
DEFAULT_SERVICE_NAME = "scene_graph"
DEFAULT_GROUP_PORT = 7725
# DEFAULT_PROMPT = "lid. stove. blue pot. carrot."/"stove. blue pan. carrot."
DEFAULT_PROMPT = "lemon. drawer. plate."


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a request to the scene graph server.")
    parser.add_argument("--node-ip", default=DEFAULT_NODE_IP)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--group-port", type=int, default=DEFAULT_GROUP_PORT)
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--max-polls", type=int, default=60)
    args = parser.parse_args()

    pyzlc.init("scene_graph_requester", args.node_ip, args.group_name, group_port=args.group_port)
    pyzlc.info(f"Waiting for service: {args.service_name}")

    if not pyzlc.wait_for_service(args.service_name, timeout=args.timeout, group_name=args.group_name):
        pyzlc.error(f"Service not available: {args.service_name}")
        return

    request_id = args.request_id or str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "prompt": args.prompt,
    }
    pyzlc.info(f"Sending request: {request}")

    request_fn = getattr(pyzlc, "call", None) or getattr(pyzlc, "zlc_request")

    for poll_index in range(args.max_polls + 1):
        response = request_fn(
            args.service_name,
            request,
            timeout=args.timeout,
            group_name=args.group_name,
        )
        pyzlc.info(f"Received response {poll_index}: {response}")

        if response and response.get("scene_graph_complete"):
            spatial_relation = response.get("spatial_relation", {})
            print(json.dumps(spatial_relation, indent=2))
            return

        time.sleep(args.poll_interval)

    pyzlc.error(
        f"Timed out waiting for spatial relation after "
        f"{args.max_polls} polls for request_id={request_id}"
    )


if __name__ == "__main__":
    main()
