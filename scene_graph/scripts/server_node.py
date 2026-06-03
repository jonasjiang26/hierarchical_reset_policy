import pyzlc
import cv2
import gc
import numpy as np
import open3d as o3d
import threading
from pathlib import Path
from typing import List
from grounded_sam import GroundedSAM
from instance import TableInstance
import traceback
from heuristics import TableSceneHeuristics

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
ZED_STATIC_CAM_TOPIC = "zed_depth"
ZED_STATIC_CAM_CONFIG = CONFIG_DIR / "eye_to_hand_zed.yaml"
DEPTHAI_STATIC_CAM_TOPIC = "static_cam"
DEPTHAI_STATIC_CAM_CONFIG = CONFIG_DIR / "eye_to_hand.yaml"
WRIST_CAM_CONFIG = CONFIG_DIR / "wrist_cam_hand_eye.yaml"

class SceneGraphServer:
    """A simple server node that provides a service to add two integers."""
    def __init__(self):
        pyzlc.init("scene_graph", "141.3.53.25", "robot_lab_robotiq_202", group_port=7725)

        pyzlc.info("scene_graph initialized and ready to receive requests.")
        self.requested = False
        self.latest_frame = None
        self.latest_static_frame = None
        self.latest_zed_static_frame = None
        self.latest_depthai_static_frame = None
        self.latest_wrist_frame = None
        self.latest_arm_state = None
        self.latest_T_base_hand = None
        self.zed_static_processed_for_request = False
        self.depthai_static_processed_for_request = False
        self.wrist_processed_for_request = False
        self.fused_point_cloud = None
        self.prompt = str
        self.active_request_id = ""
        self.completed_request_id = ""
        self.spatial_relation = ""
        self.wrist_instances: List[TableInstance] = []
        self.zed_static_instances: List[TableInstance] = []
        self.depthai_static_instances: List[TableInstance] = []
        self.static_instances: List[TableInstance] = []
        self.fused_instances: List[TableInstance] = []
        pyzlc.info(f"Forcing {ZED_STATIC_CAM_TOPIC} subscriber to use TCP transport.")
        pyzlc.get_node("robot_lab_robotiq_202").subscriber_manager.local_ip = ""
        pyzlc.register_subscriber_handler(ZED_STATIC_CAM_TOPIC, self.zed_static_cam_callback, "robot_lab_robotiq_202")
        # pyzlc.register_subscriber_handler(DEPTHAI_STATIC_CAM_TOPIC, self.depthai_static_cam_callback, "robot_lab_robotiq_202")
        pyzlc.register_subscriber_handler("wrist_cam", self.wrist_cam_callback, "robot_lab_robotiq_202") 
        pyzlc.register_subscriber_handler("FrankaPanda/franka_arm_state", self.panda_arm_state_callback, "robot_lab_robotiq_202")
        self.grounded_sam = None
        self.grounded_sam_lock = threading.Lock()

    def zed_static_cam_callback(self, frame):
        # """Example callback for image data."""
        try:
            frame = self._normalize_depth_frame(frame)
            self.latest_frame = frame
            self.latest_static_frame = frame
            self.latest_zed_static_frame = frame

            if not self.requested or self.zed_static_processed_for_request:
                return None

            self.zed_static_processed_for_request = True
            masks, phrases = self.process_frame(frame, visualize_masks=False)
            pyzlc.info(f"Processed {ZED_STATIC_CAM_TOPIC} frame")
            if masks is None:
                self._try_fuse_instances()
                return {"success": False, "message": "no masks detected"}

            self.zed_static_instances = []
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
                    self.zed_static_instances.append(instance)
            self._refresh_static_instances()
            pyzlc.info(f"created {len(self.zed_static_instances)} {ZED_STATIC_CAM_TOPIC} instances.")
            self._project_instances_in_base(
                self.zed_static_instances,
                config_path=ZED_STATIC_CAM_CONFIG,
                T_base_hand=None,
                camera_name=ZED_STATIC_CAM_TOPIC,
            )
            self._try_fuse_instances()

            return {
                "success": True,
                "phrases": phrases,
                "num_masks": len(phrases),
            }
        except Exception as exc:
            pyzlc.error(f"{ZED_STATIC_CAM_TOPIC} callback failed: {exc}")
            pyzlc.error(traceback.format_exc())
            return {"success": False, "message": str(exc)}

    def depthai_static_cam_callback(self, frame):
        try:
            frame = self._normalize_depth_frame(frame)
            self.latest_frame = frame
            self.latest_depthai_static_frame = frame

            if not self.requested or self.depthai_static_processed_for_request:
                return None

            self.depthai_static_processed_for_request = True
            masks, phrases = self.process_frame(frame, visualize_masks=False, convert_rgb_to_bgr=False)
            pyzlc.info(f"Processed {DEPTHAI_STATIC_CAM_TOPIC} frame")
            if masks is None:
                self._try_fuse_instances()
                return {"success": False, "message": "no masks detected"}

            self.depthai_static_instances = []
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
                    self.depthai_static_instances.append(instance)
            self._refresh_static_instances()
            pyzlc.info(f"created {len(self.depthai_static_instances)} {DEPTHAI_STATIC_CAM_TOPIC} instances.")
            self._project_instances_in_base(
                self.depthai_static_instances,
                config_path=DEPTHAI_STATIC_CAM_CONFIG,
                T_base_hand=None,
                camera_name=DEPTHAI_STATIC_CAM_TOPIC,
            )
            self._try_fuse_instances()

            return {
                "success": True,
                "phrases": phrases,
                "num_masks": len(phrases),
            }
        except Exception as exc:
            pyzlc.error(f"{DEPTHAI_STATIC_CAM_TOPIC} callback failed: {exc}")
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
            masks, phrases = self.process_frame(frame, visualize_masks=False, convert_rgb_to_bgr=False)
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


    def process_frame(self, frame, visualize_masks=False, convert_rgb_to_bgr=False):
        pyzlc.info(f"Received image frame keys: {list(frame.keys())}")
        pyzlc.info(f"Received image frame with timestamp: {frame['timestamp']}")
        width = frame['width']
        height = frame['height']
        channels = frame['channels']
        rgb_data = frame['rgb_data']
        pyzlc.info(f"Image shape: width={width}, height={height}, channels={channels}")

        rgb = np.frombuffer(rgb_data, dtype=np.uint8).reshape((height, width, channels))
        segmentation_image = rgb
        if convert_rgb_to_bgr:
            if channels < 3:
                pyzlc.warning("Cannot convert RGB to BGR: image has fewer than 3 channels.")
            else:
                segmentation_image = np.ascontiguousarray(rgb[:, :, :3][:, :, ::-1])

        grounded_sam = self._ensure_grounded_sam_loaded()
        masks, phrases = grounded_sam.segment(grounded_sam.model,
                                                    segmentation_image,
                                                    self.prompt,
                                                    0.3,
                                                    0.335,
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
        if visualize_masks:
            self.visualize_masks(segmentation_image, masks, phrases, frame.get("timestamp", "latest"))
        
        return masks, phrases

    def visualize_masks(self, rgb, masks, phrases, timestamp):
        try:
            overlay = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8).copy())
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
                    overlay = self._draw_mask_label(
                        overlay,
                        str(phrase),
                        (int(xs.min()), max(0, int(ys.min()) - 18)),
                        tuple(int(c) for c in color.tolist()),
                    )

            self._show_mask_overlay(overlay, timestamp)
        except Exception as exc:
            pyzlc.warning(f"Could not show mask debug window: {exc}")

    def _show_mask_overlay(self, overlay, timestamp):
        try:
            bgr_overlay = np.ascontiguousarray(overlay[:, :, ::-1].copy())
            cv2.imshow("GroundedSAM masks", bgr_overlay)
            
            cv2.waitKey(0)
        except Exception as exc:
            output_path = Path(f"/home/jjiang/jing/grounded_sam_masks_{timestamp}.png")
            try:
                from PIL import Image

                safe_name = "".join(
                    char if char.isalnum() or char in "._-" else "_"
                    for char in str(timestamp)
                )
                output_path = Path(f"/home/jjiang/jing/grounded_sam_masks_{safe_name}.png")
                Image.fromarray(overlay).save(output_path)
                pyzlc.warning(f"Could not open mask debug window: {exc}. Saved {output_path}")
            except Exception as save_exc:
                pyzlc.warning(f"Could not show or save mask debug overlay: {exc}; save failed: {save_exc}")

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
            self.zed_static_processed_for_request = False
            self.depthai_static_processed_for_request = False
            self.wrist_processed_for_request = False
            self.fused_instances = []
            self.static_instances = []
            self.zed_static_instances = []
            self.depthai_static_instances = []
            self.wrist_instances = []
            self.fused_point_cloud = None
            if not self.goal_key:
                self._ensure_grounded_sam_loaded()
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
            response = {
                "request_id": request_id,
                "scene_graph_complete": self.completed_request_id == request_id,
                "spatial_relation": self.spatial_relation
                if self.completed_request_id == request_id
                else "",
            }
            if request.get("include_static_image", False):
                response["static_image"] = self._static_frame_payload()
            return response
        

            

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
        if (
            not self.zed_static_processed_for_request
            # or not self.depthai_static_processed_for_request
            or not self.wrist_processed_for_request
        ):
            return

        self.fused_instances = []
        instances_by_key = self._group_projected_instances_by_key(
            self.zed_static_instances + self.wrist_instances # + self.depthai_static_instances
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
            pyzlc.warning(
                f"No projected {ZED_STATIC_CAM_TOPIC}, {DEPTHAI_STATIC_CAM_TOPIC}, or wrist_cam instances were available."
            )
            self.spatial_relation = ""
            self.completed_request_id = self.active_request_id
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

            self._visualize_all_fused_point_clouds()

        self.requested = False
        self._release_grounded_sam()

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

    def _refresh_static_instances(self):
        self.static_instances = self.zed_static_instances + self.depthai_static_instances

    def _ensure_grounded_sam_loaded(self):
        with self.grounded_sam_lock:
            if self.grounded_sam is None:
                pyzlc.info("Loading GroundedSAM after receiving request.")
                self.grounded_sam = GroundedSAM()
            return self.grounded_sam

    def _release_grounded_sam(self):
        with self.grounded_sam_lock:
            if self.grounded_sam is None:
                return
            pyzlc.info("Releasing GroundedSAM to free GPU memory.")
            self.grounded_sam = None

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            pyzlc.warning(f"Could not clear CUDA cache after releasing GroundedSAM: {exc}")

    def _normalize_depth_frame(self, frame):
        depth_data = frame.get("depth_data")
        if depth_data is None:
            return frame

        width = frame["width"]
        height = frame["height"]
        uint16_size = width * height * np.dtype(np.uint16).itemsize
        if len(depth_data) == uint16_size:
            return frame

        depth = np.frombuffer(depth_data, dtype=np.float32).reshape((height, width))
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.clip(depth, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        frame = dict(frame)
        frame["depth_data"] = depth.tobytes()
        return frame

    def _normalize_zed_depth_frame(self, frame):
        return self._normalize_depth_frame(frame)

    def _static_frame_payload(self):
        if self.latest_static_frame is None:
            return None

        frame = self.latest_static_frame
        return {
            "timestamp": frame.get("timestamp"),
            "width": frame.get("width"),
            "height": frame.get("height"),
            "channels": frame.get("channels"),
            "rgb_data": frame.get("rgb_data"),
        }

    def _draw_mask_label(self, image, text, position, color):
        try:
            from PIL import Image, ImageDraw

            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)
            draw.text(position, text, fill=color)
            return np.array(pil_image, dtype=np.uint8, copy=True)
        except Exception as exc:
            pyzlc.warning(f"Could not draw mask label: {exc}")
            return image


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
