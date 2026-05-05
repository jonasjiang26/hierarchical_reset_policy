from typing import Any

import cv2
import numpy as np
import pyzlc


def image_callback(frame: dict[Any, Any]):
    pyzlc.info(f"Received static_cam frame keys: {list(frame.keys())}")
    width = frame["width"]
    height = frame["height"]
    channels = frame["channels"]
    rgb = np.frombuffer(frame["rgb_data"], dtype=np.uint8).reshape((height, width, channels))
    saved = cv2.imwrite("/tmp/static_cam_debug_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    pyzlc.info(f"Saved /tmp/static_cam_debug_rgb.png: {saved}")


if __name__ == "__main__":
    pyzlc.init("static_cam_debug_listener", "141.3.53.25", "robot_lab_robotiq_202", group_port=7725)
    pyzlc.get_node("robot_lab_robotiq_202").subscriber_manager.local_ip = ""
    pyzlc.register_subscriber_handler("static_cam", image_callback, "robot_lab_robotiq_202")
    pyzlc.info("Listening to static_cam...")
    pyzlc.spin()
