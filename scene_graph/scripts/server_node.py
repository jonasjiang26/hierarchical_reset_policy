import pyzlc
import numpy as np
from typing import Any
from grounded_sam import GroundedSAM
import cv2
import traceback

class SceneGraphServer:
    """A simple server node that provides a service to add two integers."""
    def __init__(self):
        pyzlc.init("scene_graph", "141.3.53.25", "robot_lab_robotiq_202", group_port=7725)

        pyzlc.info("scene_graph initialized and ready to receive requests.")
        self.mask_requested = False
        self.latest_frame = None
        self.frame_count = 0
        pyzlc.info("Forcing static_cam subscriber to use TCP transport.")
        pyzlc.get_node("robot_lab_robotiq_202").subscriber_manager.local_ip = ""
        pyzlc.register_subscriber_handler("static_cam", self.image_callback, "robot_lab_robotiq_202")
        self.grounded_sam = GroundedSAM()

    def image_callback(self, frame):
        # """Example callback for image data."""
        try:
            self.latest_frame = frame
            self.frame_count += 1

            if not self.mask_requested:
                if self.frame_count == 1 or self.frame_count % 100 == 0:
                    pyzlc.info(f"Cached static_cam frame #{self.frame_count}")
                return None

            self.mask_requested = False
            return self.process_frame(frame)
        except Exception as exc:
            pyzlc.error(f"image_callback failed: {exc}")
            pyzlc.error(traceback.format_exc())
            return {"success": False, "message": str(exc)}

    def process_frame(self, frame):
        pyzlc.info(f"Processing static_cam frame #{self.frame_count}")
        pyzlc.info(f"Received image frame keys: {list(frame.keys())}")
        pyzlc.info(f"Received image frame with timestamp: {frame['timestamp']}")
        width = frame['width']
        height = frame['height']
        channels = frame['channels']
        rgb_data = frame['rgb_data']
        pyzlc.info(f"Image shape: width={width}, height={height}, channels={channels}")

        rgb = np.frombuffer(rgb_data, dtype=np.uint8).reshape((height, width, channels))
        rgb_saved = cv2.imwrite("/tmp/scene_graph_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        pyzlc.info(f"Saved latest RGB frame to /tmp/scene_graph_rgb.png: {rgb_saved}")

        masks, phrases = self.grounded_sam.segment(self.grounded_sam.model,
                                                    rgb,
                                                    "banana. bowl.",
                                                    0.3,
                                                    0.3,
                                                    "cuda:0")
        pyzlc.info(f"Detected phrases: {phrases}")

        if masks.shape[0] == 0:
            pyzlc.warning("No masks detected for prompt: banana. bowl.")
            return {"success": False, "message": "no masks detected"}

        mask = masks[0, 0].detach().cpu().numpy().astype(np.uint8) * 255
        mask_saved = cv2.imwrite("/tmp/scene_graph_mask.png", mask)
        pyzlc.info(f"Saved first mask to /tmp/scene_graph_mask.png: {mask_saved}")
        cv2.imshow("masks", mask)
        cv2.waitKey(1)
        return {"success": True, "phrases": phrases}

    def send_scene_graph(self, request):

        pyzlc.info(f"Received request: {request}")
        if self.latest_frame is None:
            self.mask_requested = True
            return {
                "success": False,
                "message": "no static_cam frame received yet; will process the next frame",
                "static_cam_frames_seen": self.frame_count,
            }

        return self.process_frame(self.latest_frame)

if __name__ == "__main__":
    server = SceneGraphServer()

    # Register the service
    pyzlc.register_service_handler("scene_graph", server.send_scene_graph, "robot_lab_robotiq_202")
    
    pyzlc.spin()
