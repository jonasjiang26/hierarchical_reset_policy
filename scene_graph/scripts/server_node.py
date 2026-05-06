import pyzlc
import numpy as np
from typing import Any, Dict, List
from grounded_sam import GroundedSAM
from instance import TableInstance
import cv2
import traceback

class SceneGraphServer:
    """A simple server node that provides a service to add two integers."""
    def __init__(self):
        pyzlc.init("scene_graph", "141.3.53.25", "robot_lab_robotiq_202", group_port=7725)

        pyzlc.info("scene_graph initialized and ready to receive requests.")
        self.requested = False
        self.latest_frame = None
        self.frame_count = 0
        self.prompt = "banana. table. gripper. pen."
        self.wrist_instances: List[TableInstance] = []
        self.static_instances: List[TableInstance] = []
        pyzlc.info("Forcing static_cam subscriber to use TCP transport.")
        pyzlc.get_node("robot_lab_robotiq_202").subscriber_manager.local_ip = ""
        # pyzlc.register_subscriber_handler("wrist_cam", self.image_callback, "robot_lab_robotiq_202")
        pyzlc.register_subscriber_handler("static_cam", self.static_cam_callback, "robot_lab_robotiq_202")

        self.grounded_sam = GroundedSAM()

    def static_cam_callback(self, frame):
        # """Example callback for image data."""
        try:
            # self.latest_frame = frame

            if not self.requested:
                return None

            self.requested = False
            masks, phrases = self.process_frame(frame)
            pyzlc.info(f"Processed static_cam frame")
            if masks is None:
                return {"success": False, "message": "no masks detected"}

            for mask, phrase in zip(masks, phrases):
                if phrase:
                    instance = TableInstance(
                        phrase,
                        mask,
                        rgb=frame['rgb_data'],
                        depth=frame['depth_data'],
                        width=frame['width'],
                        height=frame['height'],
                        channels=frame['channels'],
                    )
                    self.static_instances.append(instance)
            pyzlc.info(f"created {len(self.static_instances)} instances.")
            for instance in self.static_instances:
                if instance.name == "pen":
                    pyzlc.info(f"Visualizing segmented point cloud for instance: {instance.name}")
                    instance.segmented_point_cloud_in_base(visualize=True, depth_trunc=10.0)
            # rgb_example = self.static_instances[0].segment_rgb()
            # self.latest_frame = rgb_example
            # cv2.imshow("Static Instance RGB", rgb_example)
            # cv2.waitKey(0)

            return {
                "success": True,
                "phrases": phrases,
                "num_masks": len(phrases),
            }
        except Exception as exc:
            pyzlc.error(f"static_cam_callback failed: {exc}")
            pyzlc.error(traceback.format_exc())
            return {"success": False, "message": str(exc)}
            
    def wrist_cam_callback(self, frame):
        # """Example callback for image data."""
        try:
            self.latest_frame = frame
            # self.frame_count += 1

            if not self.mask_requested:
            #     if self.frame_count == 1 or self.frame_count % 100 == 0:
            #         pyzlc.info(f"Cached static_cam frame #{self.frame_count}")
                return None

            self.mask_requested = False
            masks, phrases = self.process_frame(frame) 
            if masks is None:
                return {"success": False, "message": "no masks detected"}

            for mask, phrase in zip(masks, phrases):
                instance = TableInstance(
                    phrase,
                    mask,
                    rgb=frame['rgb_data'],
                    depth=frame['depth_data'],
                    width=frame['width'],
                    height=frame['height'],
                    channels=frame['channels'],
                )
                self.wrist_instances.append(instance)
            return None
        except Exception as exc:
            pyzlc.error(f"wrist_cam_callback failed: {exc}")
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

        masks, phrases = self.grounded_sam.segment(self.grounded_sam.model,
                                                    rgb,
                                                    self.prompt,
                                                    0.2,
                                                    0.3,
                                                    "cuda:0",
                                                    with_logits=False)
        pyzlc.info(f"Detected phrases: {phrases}")
        pyzlc.info(f"Number of masks detected: {masks.shape[0]}")
        if masks.shape[0] == 0:
            pyzlc.warning(f"No masks detected for prompt: {self.prompt}")
            return None, []

        #filter out empty phrases and corresponding masks
        valid_indices = [i for i, phrase in enumerate(phrases) if phrase.strip()]
        phrases = [phrases[i] for i in valid_indices]
        masks = masks[valid_indices]
        
        return masks, phrases

    def send_scene_graph(self, request):

        pyzlc.info(f"Received request: {request}")
        if self.latest_frame is None:
            self.requested = True 
            return {
                "success": False,
                "message": "no static_cam frame received yet; will process the next frame",
                "static_cam_frames_seen": self.frame_count,
            }

        cv2.imshow("Latest Frame", self.latest_frame)
        cv2.waitKey(1)

        return {
            "success": True
        }

if __name__ == "__main__":
    server = SceneGraphServer()

    # Register the service
    pyzlc.register_service_handler("scene_graph", server.send_scene_graph, "robot_lab_robotiq_202")
    
    pyzlc.spin()
