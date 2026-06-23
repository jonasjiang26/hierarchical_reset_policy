import argparse
import pprint
import time
import uuid

import pyzlc


DEFAULT_NODE_IP = "141.3.53.25"
DEFAULT_GROUP_NAME = "robot_lab_robotiq_202"
DEFAULT_GROUP_PORT = 7725
DEFAULT_SERVICE_NAME = "scene_graph"


def compact_payload(value):
    if isinstance(value, (bytes, bytearray)):
        return f"<{type(value).__name__}: {len(value)} bytes>"
    if isinstance(value, dict):
        return {key: compact_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact_payload(item) for item in value]
    return value


def call_scene_graph(service_name, request, timeout, group_name):
    request_fn = getattr(pyzlc, "call", None) or getattr(pyzlc, "zlc_request")
    return request_fn(service_name, request, timeout=timeout, group_name=group_name)


def main():
    parser = argparse.ArgumentParser(description="Test node for the scene graph server.")
    parser.add_argument("--node-ip", default=DEFAULT_NODE_IP)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--group-port", type=int, default=DEFAULT_GROUP_PORT)
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--prompt", default="lemon. drawer. plate.")
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--max-polls", type=int, default=60)
    parser.add_argument("--include-static-image", action="store_true")
    args = parser.parse_args()

    request_id = args.request_id or str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "prompt": args.prompt,
    }
    if args.include_static_image:
        request["include_static_image"] = True

    pyzlc.init("scene_graph_test_node", args.node_ip, args.group_name, group_port=args.group_port)
    pyzlc.info(f"Waiting for service: {args.service_name}")
    if not pyzlc.wait_for_service(args.service_name, timeout=args.timeout, group_name=args.group_name):
        raise RuntimeError(f"Service not available: {args.service_name}")

    print(f"Sending scene graph request_id={request_id}")
    print(f"Prompt: {args.prompt}")

    for poll_index in range(args.max_polls + 1):
        response = call_scene_graph(args.service_name, request, args.timeout, args.group_name)
        print(f"\nResponse {poll_index}:")
        pprint.pp(compact_payload(response))

        if response and response.get("scene_graph_complete"):
            print("\nScene graph complete.")
            return

        time.sleep(args.poll_interval)

    print("\nTimed out waiting for scene graph completion.")


if __name__ == "__main__":
    main()
