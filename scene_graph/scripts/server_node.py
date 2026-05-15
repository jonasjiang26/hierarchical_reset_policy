import pyzlc
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List
from grounded_sam import GroundedSAM
from instance import TableInstance
import traceback
from heuristics import TableSceneHeuristics

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
        self.prompt = str
        self.active_request_id = ""
        self.completed_request_id = ""
        self.spatial_relation = ""
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
                                                    0.3,
                                                    0.35,
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
        # self.visualize_masks(rgb, masks, phrases, frame.get("timestamp", "latest"))
        
        return masks, phrases

    # def visualize_masks(self, rgb, masks, phrases, timestamp):
        overlay = rgb.copy()
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        for index, (mask, phrase) in enumerate(zip(masks, phrases)):
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            mask = np.asarray(mask)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            mask = mask.astype(bool)

            color = np.asarray(colors[index % len(colors)], dtype=np.uint8)
            overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)

            ys, xs = np.where(mask)
            if len(xs) > 0:
                cv2.putText(
                    overlay,
                    phrase,
                    (int(xs.min()), int(ys.min()) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    tuple(int(c) for c in color.tolist()),
                    2,
                    cv2.LINE_AA,
                )

        try:
            cv2.imshow("GroundedSAM masks", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
        except Exception as exc:
            pyzlc.warning(f"Could not show mask debug window: {exc}")

    def send_scene_graph(self, request):

        pyzlc.info(f"Received request: {request}")
        request_id = str(request.get("request_id", ""))
        if request_id != self.active_request_id:
            self.active_request_id = request_id
            self.completed_request_id = ""
            self.spatial_relation = ""
            self.prompt = request.get("prompt", "")
            self.goal_key = request.get("goal_key", "")
            self.requested = True
            self.static_processed_for_request = False
            self.wrist_processed_for_request = False
            self.fused_instances = []
            self.static_instances = []
            self.wrist_instances = []
            self.fused_point_cloud = None
        else:
            pyzlc.info(f"Polling existing request_id: {request_id}")
        if self.goal_key:
            pyzlc.info(f"sending goal pcd: {self.goal_key}")
            for instance in self.fused_instances:
                if self._instance_key(instance.name) == self._instance_key(self.goal_key):
                    goal_point_cloud = instance.segmented_point_cloud
                    pcd_data = np.asarray(goal_point_cloud.points, dtype=np.float32).tobytes()
                    pyzlc.info(f"Found goal instance in fused_instances: {instance.name}.")
                    return {
                        "success": True,
                        "request_id": request_id,
                        "goal_point_cloud": pcd_data,
                        "num_points": len(goal_point_cloud.points),
                    }
            return {"success": False, "request_id": request_id, "message": f"goal instance not found: {self.goal_key}"}
        else:
            return {
                "request_id": request_id,
                "spatial_relation": self.spatial_relation
                if self.completed_request_id == request_id
                else "",
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
            instance.segmented_point_cloud = instance.segmented_point_cloud

    def _try_fuse_instances(self):
        if not self.static_processed_for_request or not self.wrist_processed_for_request:
            return

        self.fused_instances = []
        instances_by_key = self._group_projected_instances_by_key(
            self.static_instances + self.wrist_instances
        )

        for key, instances in sorted(instances_by_key.items()):
            fused_instance = instances[0]
            fused_pcd = None

            for instance in instances:
                fused_pcd = fused_instance.fuse_projected_point_clouds(
                    fused_pcd,
                    instance.segmented_point_cloud,
                    visualize=False,
                )

            fused_instance.segmented_point_cloud = fused_pcd
            self.fused_instances.append(fused_instance)
            pyzlc.info(f"Fused {len(instances)} point cloud(s) for instance name: {key}")

        if len(self.fused_instances) == 0:
            pyzlc.warning("No projected static_cam or wrist_cam instances were available to visualize.")
        else:
            heuristics = TableSceneHeuristics()
            spatial_relations = []
            seen_relations = set()
            contained_instance_keys = set()

            for instance in self.fused_instances:
                for chosen_instance in self.fused_instances:
                    if chosen_instance.name == instance.name:
                        continue
                    if not heuristics.is_in(instance, chosen_instance):
                        continue

                    relation = f"{instance.name} is in {chosen_instance.name}"
                    if relation and relation not in seen_relations:
                        spatial_relations.append(relation)
                        seen_relations.add(relation)
                    contained_instance_keys.add(self._instance_key(instance.name))

            for instance in self.fused_instances:
                if self._instance_key(instance.name) in contained_instance_keys:
                    continue
                for chosen_instance in self.fused_instances:
                    if self._instance_key(chosen_instance.name) in contained_instance_keys:
                        continue
                    if chosen_instance.name != instance.name and heuristics.is_in(
                        instance,
                        chosen_instance,
                    ):
                        continue

                    relation = heuristics.get_spatial_relation(instance, chosen_instance)
                    if relation and relation not in seen_relations:
                        spatial_relations.append(relation)
                        seen_relations.add(relation)

            self.spatial_relation = "\n".join(spatial_relations)
            self.completed_request_id = self.active_request_id
            pyzlc.info(f"Spatial relations:\n{self.spatial_relation}")

                # self._visualize_all_fused_point_clouds()

        self.requested = False

    def _group_projected_instances_by_key(self, instances):
        instances_by_key = {}
        for instance in instances:
            if not hasattr(instance, "segmented_point_cloud"):
                continue
            key = self._instance_key(instance.name)
            instances_by_key.setdefault(key, []).append(instance)
        return instances_by_key

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
