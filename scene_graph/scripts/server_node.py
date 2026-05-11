import pyzlc
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List
from grounded_sam import GroundedSAM
from instance import TableInstance
import traceback

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
WRIST_CAM_CONFIG = CONFIG_DIR / "wrist_cam_hand_eye.yaml"

class SceneGraphServer:
    """A simple server node that provides a service to add two integers."""
    def __init__(self):
        pyzlc.init("scene_graph", "141.3.53.25", "robot_lab_robotiq_202", group_port=7725)

        pyzlc.info("scene_graph initialized and ready to receive requests.")
        self.requested = False
        self.latest_frame = None
        self.latest_static_frame = None
        self.latest_wrist_frame = None
        self.latest_arm_state = None
        self.latest_T_base_hand = None
        self.static_processed_for_request = False
        self.wrist_processed_for_request = False
        self.fused_point_cloud = None
        self.prompt = "yellow. banana. red bowl."
        self.wrist_instances: List[TableInstance] = []
        self.static_instances: List[TableInstance] = []
        self.fused_instances: List[TableInstance] = []
        pyzlc.info("Forcing static_cam subscriber to use TCP transport.")
        pyzlc.get_node("robot_lab_robotiq_202").subscriber_manager.local_ip = ""
        pyzlc.register_subscriber_handler("static_cam", self.static_cam_callback, "robot_lab_robotiq_202")
        pyzlc.register_subscriber_handler("wrist_cam", self.wrist_cam_callback, "robot_lab_robotiq_202")
        pyzlc.register_subscriber_handler("FrankaPanda/franka_arm_state", self.panda_arm_state_callback, "robot_lab_robotiq_202")
        self.grounded_sam = GroundedSAM()

    def static_cam_callback(self, frame):
        # """Example callback for image data."""
        try:
            self.latest_frame = frame
            self.latest_static_frame = frame

            if not self.requested or self.static_processed_for_request:
                return None

            self.static_processed_for_request = True
            masks, phrases = self.process_frame(frame)
            pyzlc.info(f"Processed static_cam frame")
            if masks is None:
                self._try_fuse_instances()
                return {"success": False, "message": "no masks detected"}

            self.static_instances = []
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
            self._project_instances_in_base(
                self.static_instances,
                config_path=None,
                T_base_hand=None,
                camera_name="static_cam",
            )
            self._try_fuse_instances()

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
            self.latest_wrist_frame = frame

            if not self.requested or self.wrist_processed_for_request:
                return None

            if self.latest_T_base_hand is None:
                pyzlc.warning("Cannot process wrist_cam yet: no FrankaPanda/franka_arm_state received.")
                return {"success": False, "message": "no arm_state received yet"}

            T_base_hand = self.latest_T_base_hand.copy()
            self.wrist_processed_for_request = True
            masks, phrases = self.process_frame(frame) 
            if masks is None:
                self._try_fuse_instances()
                return {"success": False, "message": "no masks detected"}

            self.wrist_instances = []
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
            pyzlc.info(f"created {len(self.wrist_instances)} wrist_cam instances.")
            self._project_instances_in_base(
                self.wrist_instances,
                config_path=WRIST_CAM_CONFIG,
                T_base_hand=T_base_hand,
                camera_name="wrist_cam",
            )
            self._try_fuse_instances()

            return None
        except Exception as exc:
            pyzlc.error(f"wrist_cam_callback failed: {exc}")
            pyzlc.error(traceback.format_exc())
            return {"success": False, "message": str(exc)}

    def panda_arm_state_callback(self, arm_state):
        try:
            self.latest_arm_state = arm_state
            self.latest_T_base_hand = TableInstance.arm_state_to_base_hand_transform(arm_state)
        except Exception as exc:
            pyzlc.error(f"Failed to parse FrankaPanda/franka_arm_state: {exc}")
            pyzlc.error(traceback.format_exc())
        return None


    def process_frame(self, frame):
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
                                                    0.1,
                                                    0.1,
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
        self.requested = True
        self.static_processed_for_request = False
        self.wrist_processed_for_request = False
        self.static_instances = []
        self.wrist_instances = []
        self.fused_instances = []
        self.fused_point_cloud = None

        return {
            "success": True,
            "message": "will process the next static_cam and wrist_cam frames, then visualize fused point clouds",
            "has_arm_state": self.latest_T_base_hand is not None,
        }

    def _project_instances_in_base(self, instances, config_path, T_base_hand, camera_name):
        for instance in instances:
            pyzlc.info(f"Projecting {camera_name} segmented point cloud for instance: {instance.name}")
            instance.segmented_point_cloud = instance.segmented_point_cloud_in_base(
                config_path=config_path,
                T_base_hand=T_base_hand,
                visualize=False,
                depth_trunc=10.0,
            )

    def _try_fuse_instances(self):
        if not self.static_processed_for_request or not self.wrist_processed_for_request:
            return

        self.fused_instances = []
        wrist_instances_by_name = {
            self._instance_key(instance.name): instance
            for instance in self.wrist_instances
            if hasattr(instance, "segmented_point_cloud")
        }

        for static_instance in self.static_instances:
            if not hasattr(static_instance, "segmented_point_cloud"):
                continue

            wrist_instance = wrist_instances_by_name.get(self._instance_key(static_instance.name))
            if wrist_instance is None:
                continue

            pyzlc.info(f"Fusing segmented point cloud for instance: {static_instance.name}")
            static_instance.segmented_point_cloud = static_instance.fuse_projected_point_clouds(
                static_instance.segmented_point_cloud,
                wrist_instance.segmented_point_cloud,
                visualize=False,
            )
            self.fused_instances.append(static_instance)

        if len(self.fused_instances) == 0:
            pyzlc.warning("No matching static_cam and wrist_cam instances were available to fuse.")
        else:
            self._visualize_all_fused_point_clouds()

        self.requested = False

    def _instance_key(self, name):
        return name.strip().lower().rstrip(".")

    def _visualize_all_fused_point_clouds(self):
        self.fused_point_cloud = o3d.geometry.PointCloud()
        for instance in self.fused_instances:
            if hasattr(instance, "segmented_point_cloud"):
                self.fused_point_cloud += instance.segmented_point_cloud

        if len(self.fused_point_cloud.points) == 0:
            pyzlc.warning("Combined fused point cloud is empty.")
            return

        pyzlc.info(f"Visualizing all fused point clouds together: count={len(self.fused_point_cloud.points)}")
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1,
            origin=[0.0, 0.0, 0.0],
        )
        o3d.visualization.draw_geometries(
            [self.fused_point_cloud, frame],
            window_name="all fused segmented point clouds",
        )

if __name__ == "__main__":
    server = SceneGraphServer()

    # Register the service
    pyzlc.register_service_handler("scene_graph", server.send_scene_graph, "robot_lab_robotiq_202")
    
    pyzlc.spin()
