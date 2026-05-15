import argparse

import pyzlc


DEFAULT_NODE_IP = "141.3.53.25"
DEFAULT_GROUP_NAME = "robot_lab_robotiq_202"
DEFAULT_SERVICE_NAME = "scene_graph"
DEFAULT_GROUP_PORT = 7725


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a request to the scene graph server.")
    parser.add_argument("--node-ip", default=DEFAULT_NODE_IP)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--group-port", type=int, default=DEFAULT_GROUP_PORT)
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    pyzlc.init("scene_graph_requester", args.node_ip, args.group_name, group_port=args.group_port)
    pyzlc.info(f"Waiting for service: {args.service_name}")

    if not pyzlc.wait_for_service(args.service_name, timeout=args.timeout, group_name=args.group_name):
        pyzlc.error(f"Service not available: {args.service_name}")
        return

    request = {
        "prompt": "sponge. bowl.",
        # "goal_key": "wash sponge.",
    }
    pyzlc.info(f"Sending request: {request}")

    pyzlc.sleep(15)  # Ensure the server is ready to receive the request
    request_fn = getattr(pyzlc, "call", None) or getattr(pyzlc, "zlc_request")
    response = request_fn(
        args.service_name,
        request,
        timeout=args.timeout,
        group_name=args.group_name,
    )
    pyzlc.info(f"Received response: {response}")


if __name__ == "__main__":
    main()
