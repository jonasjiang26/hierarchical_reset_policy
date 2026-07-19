import asyncio
import pyzlc
import cv2
import numpy as np
import open3d as o3d
import threading
import time
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
CAMERA_TOPICS = (
    ZED_STATIC_CAM_TOPIC,
    DEPTHAI_STATIC_CAM_TOPIC,
    "wrist_cam",
)
# Both camera publishers stamp frames with time.time(), i.e. Unix epoch
# seconds as a float. Allow a small NTP/clock offset, but never process a
# frame that has spent several seconds in the transport queue.
CAMERA_CLOCK_SKEW_TOLERANCE_SECONDS = 0.25
MAX_ACCEPTABLE_FRAME_AGE_SECONDS = 2.0

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
        self.goal_key = ""
        self.active_request_id = ""
        self.completed_request_id = ""
        self.spatial_relation = self._empty_spatial_relation()
        self.wrist_instances: List[TableInstance] = []
        self.zed_static_instances: List[TableInstance] = []
        self.depthai_static_instances: List[TableInstance] = []
        self.static_instances: List[TableInstance] = []
        self.fused_instances: List[TableInstance] = []
        self._frame_condition = threading.Condition()
        self._frame_sequences = {
            ZED_STATIC_CAM_TOPIC: 0,
            DEPTHAI_STATIC_CAM_TOPIC: 0,
            "wrist_cam": 0,
        }
        self._latest_frame_slots = {}
        self._request_frame_cutoffs = dict(self._frame_sequences)
        self._request_started_monotonic = 0.0
        self._request_started_wall_time = 0.0
        self._request_generation = 0
        self._camera_stream_ready_generation = 0
        self._processed_frame_metadata = {}
        self._frame_worker_running = True
        pyzlc.info(f"Forcing {ZED_STATIC_CAM_TOPIC} subscriber to use TCP transport.")
        self._pyzlc_node = pyzlc.get_node("robot_lab_robotiq_202")
        self._pyzlc_node.subscriber_manager.local_ip = ""
        pyzlc.register_subscriber_handler(
            ZED_STATIC_CAM_TOPIC,
            self.zed_static_cam_callback,
            "robot_lab_robotiq_202",
            conflate=True,
        )
        pyzlc.register_subscriber_handler(
            DEPTHAI_STATIC_CAM_TOPIC,
            self.depthai_static_cam_callback,
            "robot_lab_robotiq_202",
            conflate=True,
        )
        pyzlc.register_subscriber_handler(
            "wrist_cam",
            self.wrist_cam_callback,
            "robot_lab_robotiq_202",
            conflate=True,
        )
        pyzlc.register_subscriber_handler(
            "FrankaPanda/franka_arm_state",
            self.panda_arm_state_callback,
            "robot_lab_robotiq_202",
            conflate=True,
        )
        pyzlc.info("Loading GroundedSAM during scene_graph initialization.")
        self.grounded_sam = GroundedSAM()
        self._frame_worker = threading.Thread(
            target=self._frame_worker_loop,
            name="scene-graph-frame-worker",
            daemon=True,
        )
        self._frame_worker.start()

    def zed_static_cam_callback(self, frame):
        self._store_latest_camera_frame(ZED_STATIC_CAM_TOPIC, frame)
        return None

    def depthai_static_cam_callback(self, frame):
        self._store_latest_camera_frame(DEPTHAI_STATIC_CAM_TOPIC, frame)
        return None

    def wrist_cam_callback(self, frame):
        self._store_latest_camera_frame("wrist_cam", frame)
        return None

    def panda_arm_state_callback(self, arm_state):
        try:
            T_base_hand = TableInstance.arm_state_to_base_hand_transform(arm_state)
            with self._frame_condition:
                self.latest_arm_state = arm_state
                self.latest_T_base_hand = T_base_hand
                self._frame_condition.notify_all()
        except Exception as exc:
            pyzlc.error(f"Failed to parse FrankaPanda/franka_arm_state: {exc}")
            pyzlc.error(traceback.format_exc())
        return None

    def _store_latest_camera_frame(self, topic, frame):
        received_at = time.monotonic()
        received_wall_time = time.time()
        capture_time = self._camera_capture_time(frame)
        with self._frame_condition:
            sequence = self._frame_sequences[topic] + 1
            self._frame_sequences[topic] = sequence
            self._latest_frame_slots[topic] = {
                "frame": frame,
                "sequence": sequence,
                "received_at": received_at,
                "received_wall_time": received_wall_time,
                "capture_time": capture_time,
            }
            self.latest_frame = frame
            if topic == ZED_STATIC_CAM_TOPIC:
                self.latest_static_frame = frame
                self.latest_zed_static_frame = frame
            elif topic == DEPTHAI_STATIC_CAM_TOPIC:
                self.latest_depthai_static_frame = frame
            elif topic == "wrist_cam":
                self.latest_wrist_frame = frame
            self._frame_condition.notify_all()

    def _frame_worker_loop(self):
        while True:
            with self._frame_condition:
                job = self._claim_next_frame_job_locked()
                while self._frame_worker_running and job is None:
                    self._frame_condition.wait()
                    job = self._claim_next_frame_job_locked()
                if not self._frame_worker_running:
                    return

            self._process_frame_job(job)

    def _claim_next_frame_job_locked(self):
        if not self.requested:
            return None
        if self._camera_stream_ready_generation != self._request_generation:
            return None

        topic_settings = (
            (
                ZED_STATIC_CAM_TOPIC,
                "zed_static_processed_for_request",
                False,
                ZED_STATIC_CAM_CONFIG,
                "zed_static_instances",
            ),
            (
                DEPTHAI_STATIC_CAM_TOPIC,
                "depthai_static_processed_for_request",
                True,
                DEPTHAI_STATIC_CAM_CONFIG,
                "depthai_static_instances",
            ),
            (
                "wrist_cam",
                "wrist_processed_for_request",
                True,
                WRIST_CAM_CONFIG,
                "wrist_instances",
            ),
        )

        for topic, processed_attr, convert_rgb_to_bgr, config_path, instances_attr in topic_settings:
            if getattr(self, processed_attr):
                continue

            slot = self._latest_frame_slots.get(topic)
            if slot is None:
                continue
            if slot["sequence"] <= self._request_frame_cutoffs.get(topic, 0):
                continue
            if slot["received_at"] <= self._request_started_monotonic:
                continue
            capture_time = slot["capture_time"]
            if capture_time is None:
                pyzlc.warning(
                    f"Rejecting {topic} frame without a valid time.time() "
                    f"timestamp: {slot['frame'].get('timestamp')!r}"
                )
                continue

            frame_age = time.time() - capture_time
            if capture_time < (
                self._request_started_wall_time
                - CAMERA_CLOCK_SKEW_TOLERANCE_SECONDS
            ):
                pyzlc.warning(
                    f"Rejecting pre-request {topic} frame: timestamp="
                    f"{slot['frame'].get('timestamp')}, "
                    f"request_timestamp={self._request_started_wall_time:.6f}, "
                    f"frame_age={frame_age:.3f}s"
                )
                continue
            if frame_age > MAX_ACCEPTABLE_FRAME_AGE_SECONDS:
                pyzlc.warning(
                    f"Rejecting delayed {topic} frame: timestamp="
                    f"{slot['frame'].get('timestamp')}, "
                    f"frame_age={frame_age:.3f}s exceeds "
                    f"{MAX_ACCEPTABLE_FRAME_AGE_SECONDS:.3f}s"
                )
                continue

            T_base_hand = None
            if topic == "wrist_cam":
                if self.latest_T_base_hand is None:
                    continue
                T_base_hand = self.latest_T_base_hand.copy()

            setattr(self, processed_attr, True)
            return {
                "topic": topic,
                "frame": slot["frame"],
                "sequence": slot["sequence"],
                "received_at": slot["received_at"],
                "received_wall_time": slot["received_wall_time"],
                "capture_time": capture_time,
                "request_id": self.active_request_id,
                "request_generation": self._request_generation,
                "prompt": self.prompt,
                "convert_rgb_to_bgr": convert_rgb_to_bgr,
                "config_path": config_path,
                "instances_attr": instances_attr,
                "T_base_hand": T_base_hand,
            }

        return None

    async def _reset_camera_subscribers(self, request_generation, request_id):
        manager = self._pyzlc_node.subscriber_manager
        subscriber_dict = getattr(manager, "subscriber_dict", None)

        if subscriber_dict is None:
            pyzlc.warning(
                "This pyzlc version does not expose subscriber_dict; "
                "cannot reset camera TCP streams."
            )
            self._mark_camera_streams_ready(request_generation)
            return

        subscribers_and_urls = []
        try:
            for topic in CAMERA_TOPICS:
                subscriber = subscriber_dict.get(topic)
                if subscriber is None:
                    pyzlc.warning(
                        f"Cannot reset {topic} stream: subscriber is not registered."
                    )
                    continue

                urls = list(getattr(subscriber, "sub_urls", []))
                subscribers_and_urls.append((topic, subscriber, urls))
                for url in urls:
                    subscriber._socket.disconnect(url)
                subscriber.sub_urls.clear()

            # Yield to the ZeroMQ event loop after disconnecting so queued
            # messages from the old TCP connections are discarded.
            await asyncio.sleep(0.05)

            for topic, subscriber, urls in subscribers_and_urls:
                for url in urls:
                    subscriber.connect(url)
                pyzlc.info(
                    f"Reset {topic} subscriber for request_id={request_id}; "
                    f"reconnected to {len(urls)} publisher(s)."
                )
        except Exception as exc:
            pyzlc.error(
                f"Failed to reset camera subscribers for request_id={request_id}: {exc}"
            )
            pyzlc.error(traceback.format_exc())
        finally:
            self._mark_camera_streams_ready(request_generation)

    def _mark_camera_streams_ready(self, request_generation):
        with self._frame_condition:
            if request_generation != self._request_generation:
                return

            self._latest_frame_slots.clear()
            self._request_frame_cutoffs = dict(self._frame_sequences)
            self._request_started_monotonic = time.monotonic()
            self._request_started_wall_time = time.time()
            self._camera_stream_ready_generation = request_generation
            self._frame_condition.notify_all()
            pyzlc.info(
                f"Camera streams ready for request_id={self.active_request_id}; "
                f"frame cutoffs={self._request_frame_cutoffs}"
            )

    def _camera_capture_time(self, frame):
        """Return the camera's time.time() capture timestamp."""
        timestamp = frame.get("timestamp")
        if isinstance(timestamp, bool) or timestamp is None:
            return None

        try:
            capture_time = float(timestamp)
        except (TypeError, ValueError):
            return None

        # time.time() is currently around 1.8e9. Reject device-relative
        # clocks, counters, milliseconds, and nanoseconds so publisher/server
        # timestamp contract errors are visible instead of silently guessed.
        if not 1_000_000_000.0 <= capture_time <= 10_000_000_000.0:
            return None
        return capture_time

    def _process_frame_job(self, job):
        topic = job["topic"]
        frame = job["frame"]
        request_generation = job["request_generation"]
        try:
            frame = self._normalize_depth_frame(frame)
            frame_age_at_receive = job["received_wall_time"] - job["capture_time"]
            frame_age_now = time.time() - job["capture_time"]
            pyzlc.info(
                f"Processing fresh {topic} frame for request_id={job['request_id']}: "
                f"sequence={job['sequence']}, timestamp={frame.get('timestamp')}, "
                f"age_at_receive={frame_age_at_receive:.3f}s, "
                f"age_at_processing={frame_age_now:.3f}s"
            )
            masks, phrases = self.process_frame(
                frame,
                visualize_masks=False,
                convert_rgb_to_bgr=job["convert_rgb_to_bgr"],
                prompt=job["prompt"],
            )

            instances = []
            if masks is not None:
                for mask, phrase in zip(masks, phrases):
                    if phrase:
                        instances.append(
                            TableInstance(
                                phrase,
                                mask,
                                rgb=frame["rgb_data"],
                                depth=frame["depth_data"],
                                width=frame["width"],
                                height=frame["height"],
                                channels=frame["channels"],
                            )
                        )
                self._project_instances_in_base(
                    instances,
                    config_path=job["config_path"],
                    T_base_hand=job["T_base_hand"],
                    camera_name=topic,
                    filter=False,
                )

            with self._frame_condition:
                if request_generation != self._request_generation:
                    pyzlc.info(
                        f"Discarding {topic} result from superseded "
                        f"request_id={job['request_id']}."
                    )
                    return
                setattr(self, job["instances_attr"], instances)
                self._processed_frame_metadata[topic] = {
                    "timestamp": job["capture_time"],
                    "age_at_receive_seconds": frame_age_at_receive,
                    "age_at_processing_seconds": frame_age_now,
                    "sequence": job["sequence"],
                }
                self._refresh_static_instances()

            pyzlc.info(f"Created {len(instances)} {topic} instances.")
            self._try_fuse_instances(expected_generation=request_generation)
        except Exception as exc:
            pyzlc.error(f"{topic} frame worker failed: {exc}")
            pyzlc.error(traceback.format_exc())
            with self._frame_condition:
                if request_generation != self._request_generation:
                    return
                setattr(self, job["instances_attr"], [])
                self._refresh_static_instances()
            self._try_fuse_instances(expected_generation=request_generation)

    def process_frame(
        self,
        frame,
        visualize_masks=False,
        convert_rgb_to_bgr=False,
        prompt=None,
    ):
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

        prompt = self.prompt if prompt is None else prompt
        masks, phrases = self.grounded_sam.segment(self.grounded_sam.model,
                                                    segmentation_image,
                                                    prompt,
                                                    0.3,
                                                    0.4,
                                                    "cuda:0",
                                                    with_logits=False)
        pyzlc.info(f"Detected phrases: {phrases}")
        pyzlc.info(f"Number of masks detected: {masks.shape[0]}")
        if masks.shape[0] == 0:
            pyzlc.warning(f"No masks detected for prompt: {prompt}")
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
        with self._frame_condition:
            if request_id != self.active_request_id:
                self._request_generation += 1
                request_generation = self._request_generation
                self.active_request_id = request_id
                self.completed_request_id = ""
                self.spatial_relation = self._empty_spatial_relation()
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
                self._processed_frame_metadata = {}
                self._request_started_monotonic = time.monotonic()
                self._request_started_wall_time = time.time()
                self._request_frame_cutoffs = dict(self._frame_sequences)
                self._camera_stream_ready_generation = 0
                pyzlc.info(
                    f"Resetting camera streams before collecting frames for "
                    f"request_id={request_id}; "
                    f"request_timestamp={self._request_started_wall_time:.6f}"
                )
                self._pyzlc_node.loop_manager.submit_loop_task(
                    self._reset_camera_subscribers(request_generation, request_id)
                )
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
                "frame_metadata": dict(self._processed_frame_metadata),
                "spatial_relation": self.spatial_relation
                if self.completed_request_id == request_id
                else "",
            }
            if request.get("include_static_image", False):
                response["static_image"] = self._static_frame_payload()
            return response
        

            

    def _project_instances_in_base(self, instances, config_path, T_base_hand, camera_name, filter):
        for instance in instances:
            pyzlc.info(f"Projecting {camera_name} segmented point cloud for instance: {instance.name}")
            instance.segmented_point_cloud = instance.segmented_point_cloud_in_base(
                config_path=config_path,
                T_base_hand=T_base_hand,
                visualize=False,
                depth_trunc=10.0,
                filter_noise=filter
            )
            instance.segmented_point_cloud = instance.segmented_point_cloud

    def _try_fuse_instances(self, expected_generation=None):
        if (
            expected_generation is not None
            and expected_generation != self._request_generation
        ):
            return
        if (
            not self.zed_static_processed_for_request
            or not self.depthai_static_processed_for_request
            # or not self.wrist_processed_for_request
        ):
            return

        self.fused_instances = []
        instances_by_key = self._group_projected_instances_by_key(
            self.zed_static_instances + self.depthai_static_instances # +self.wrist_instances
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
            if (
                expected_generation is not None
                and expected_generation != self._request_generation
            ):
                return
            self.spatial_relation = self._empty_spatial_relation()
            self.completed_request_id = self.active_request_id
        else:
            heuristics = TableSceneHeuristics()
            spatial_relations = []
            seen_relations = set()
            placed_instance_keys = set()

            for instance in self.fused_instances:
                for chosen_instance in self.fused_instances:
                    if chosen_instance.name == instance.name:
                        continue
                    if not heuristics.is_in(instance, chosen_instance):
                        continue

                    subject_name = heuristics.spatial_relation_instance_name(instance)
                    target_name = heuristics.spatial_relation_instance_name(chosen_instance)
                    relation = f"{subject_name} is in {target_name}"
                    if relation and relation not in seen_relations:
                        spatial_relations.append(relation)
                        seen_relations.add(relation)
                    placed_instance_keys.add(self._instance_key(instance.name))

            for instance in self.fused_instances:
                if self._instance_key(instance.name) in placed_instance_keys:
                    continue
                for chosen_instance in self.fused_instances:
                    if chosen_instance.name == instance.name:
                        continue

                    if not heuristics.is_on(instance, chosen_instance):
                        continue

                    subject_name = heuristics.spatial_relation_instance_name(instance)
                    target_name = heuristics.spatial_relation_instance_name(chosen_instance)
                    relation = f"{subject_name} on {target_name}"
                    if relation and relation not in seen_relations:
                        spatial_relations.append(relation)
                        seen_relations.add(relation)
                    placed_instance_keys.add(self._instance_key(instance.name))
                    break

            for instance in self.fused_instances:
                if self._instance_key(instance.name) in placed_instance_keys:
                    continue
                if not heuristics.is_on_table(instance):
                    continue

                subject_name = heuristics.spatial_relation_instance_name(instance)
                relation = f"{subject_name} on table"
                if relation and relation not in seen_relations:
                    spatial_relations.append(relation)
                    seen_relations.add(relation)

            if (
                expected_generation is not None
                and expected_generation != self._request_generation
            ):
                return
            self.spatial_relation = self._build_spatial_relation_graph(spatial_relations)
            self.completed_request_id = self.active_request_id
            pyzlc.info(f"Spatial relations:\n{self.spatial_relation}")

            self._visualize_all_fused_point_clouds()

        with self._frame_condition:
            if (
                expected_generation is not None
                and expected_generation != self._request_generation
            ):
                return
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

    def _empty_spatial_relation(self):
        return {
            "objects": [],
            "relations": [],
        }

    def _build_spatial_relation_graph(self, spatial_relations):
        graph = self._empty_spatial_relation()
        seen_object_keys = set()

        for instance in self.fused_instances:
            self._add_spatial_object(graph, seen_object_keys, instance.name)

        for relation in spatial_relations:
            graph["relations"].append({"relation": relation})
            if relation.endswith(" on table"):
                self._add_spatial_object(graph, seen_object_keys, "table")

        return graph

    def _add_spatial_object(self, graph, seen_object_keys, object_id):
        object_key = self._instance_key(object_id)
        if object_key in seen_object_keys:
            return
        graph["objects"].append({"id": object_id})
        seen_object_keys.add(object_key)

    def _refresh_static_instances(self):
        self.static_instances = self.zed_static_instances + self.depthai_static_instances

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
