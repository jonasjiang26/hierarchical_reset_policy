import argparse
import csv
import shutil
import threading
import time
from pathlib import Path
from pprint import pformat

import numpy as np
import pyzlc
from PIL import Image

from grounded_sam import GroundedSAM
from heuristics import TableSceneHeuristics
from instance import TableInstance


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
ZED_STATIC_CAM_TOPIC = "zed_depth"
ZED_STATIC_CAM_CONFIG = CONFIG_DIR / "eye_to_hand_zed.yaml"
DEPTHAI_STATIC_CAM_TOPIC = "static_cam"
DEPTHAI_STATIC_CAM_CONFIG = CONFIG_DIR / "eye_to_hand.yaml"
CAMERA_TOPICS = (
    ZED_STATIC_CAM_TOPIC,
    DEPTHAI_STATIC_CAM_TOPIC,
)
CAMERA_CONFIGS = {
    ZED_STATIC_CAM_TOPIC: ZED_STATIC_CAM_CONFIG,
    DEPTHAI_STATIC_CAM_TOPIC: DEPTHAI_STATIC_CAM_CONFIG,
}
ROLLOUT_STATE_TOPIC = "roll-out state"
RESET_STATE_TOPIC = "reset state"
SPATIAL_RELATION_SEQUENCE_TOPIC = "spatial relation sequence"
ROLLOUT_START_MESSAGE = "roll-out starts"
ROLLOUT_END_MESSAGE = "roll-out ends"

DEFAULT_PROMPT = "lemon. drawer. plate."
DEFAULT_SEGMENTATION_PATCHES = {
    ZED_STATIC_CAM_TOPIC: [400, 240, 868, 623],
    DEPTHAI_STATIC_CAM_TOPIC: [321, 168, 1000, 600],
}
CAMERA_COLOR_CONVERSIONS = {
    ZED_STATIC_CAM_TOPIC: False,
    DEPTHAI_STATIC_CAM_TOPIC: True,
}
DETECTION_HZ = 3.0
DETECTION_PERIOD_SECONDS = 1.0 / DETECTION_HZ
MAX_ACCEPTABLE_FRAME_AGE_SECONDS = 2.0
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "grounding_dino_rgbd"
DEFAULT_KEY_OBJECT = "lemon"
DEFAULT_KEYFRAME_SAMPLE_INTERVAL_SECONDS = 1.5
DETECTION_LOG_COLUMNS = (
    "state",
    "state_message",
    "camera",
    "sequence",
    "timestamp",
    "detected_objects",
    "rgb_path",
    "depth_path",
    "rgbd_path",
)
KEYFRAME_LOG_COLUMNS = (
    "state",
    "session_id",
    "camera",
    "sequence",
    "timestamp",
    "reason",
    "detected_objects",
    "rgb_path",
    "depth_path",
    "rgbd_path",
    "source_rgb_path",
    "source_depth_path",
    "source_rgbd_path",
)
KEYFRAME_RELATION_LOG_COLUMNS = (
    "keyframe_index",
    "cameras",
    "timestamps",
    "objects",
    "relations",
)


class GroundedDinoVocabDetectionNode:
    def __init__(
        self,
        prompt,
        node_ip,
        group_name,
        group_port,
        box_threshold,
        text_threshold,
        device,
        output_dir,
        key_object,
        keyframe_sample_interval,
        relation_sequence_topic,
    ):
        pyzlc.init("grounded_dino_vocab_detection", node_ip, group_name, group_port=group_port)
        pyzlc.info("grounded_dino_vocab_detection node initialized.")

        self.prompt = prompt
        self.group_name = group_name
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.key_object = key_object
        self.keyframe_sample_interval = keyframe_sample_interval
        self.relation_sequence_publisher = pyzlc.Publisher(
            relation_sequence_topic,
            group_name,
        )
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.keyframes_dir = self.output_dir / "keyframes"
        self.detection_log_path = self.output_dir / "detections.csv"
        self._io_lock = threading.Lock()

        self._state_lock = threading.Lock()
        self._state_condition = threading.Condition(self._state_lock)
        self._active_state = None
        self._active_state_message = ""
        self._active_sessions = {}
        self._session_counter = 0
        self._inflight_jobs = 0
        self._latest_frames = {}
        self._frame_sequences = {topic: 0 for topic in CAMERA_TOPICS}
        self._last_processed_sequences = {topic: 0 for topic in CAMERA_TOPICS}
        self._running = True
        self._prepare_output_dir()

        pyzlc.info(f"Forcing {ZED_STATIC_CAM_TOPIC} subscriber to use TCP transport.")
        pyzlc.get_node(group_name).subscriber_manager.local_ip = ""

        for topic in CAMERA_TOPICS:
            pyzlc.register_subscriber_handler(
                topic,
                self._camera_callback(topic),
                group_name,
                conflate=True,
            )
            pyzlc.info(f"Subscribed to camera topic {topic!r}.")

        pyzlc.register_subscriber_handler(
            ROLLOUT_STATE_TOPIC,
            self.rollout_state_callback,
            group_name,
            conflate=True,
        )
        pyzlc.register_subscriber_handler(
            RESET_STATE_TOPIC,
            self.reset_state_callback,
            group_name,
            conflate=True,
        )
        pyzlc.info(
            f"Subscribed to state topics {ROLLOUT_STATE_TOPIC!r} and {RESET_STATE_TOPIC!r}."
        )

        pyzlc.info("Loading GroundingDINO model.")
        self.grounded_sam = GroundedSAM(device=device)
        self._worker = threading.Thread(
            target=self._detection_loop,
            name="grounded-dino-vocab-detection-worker",
            daemon=True,
        )
        self._worker.start()

    def _camera_callback(self, topic):
        def callback(frame):
            with self._state_lock:
                sequence = self._frame_sequences[topic] + 1
                self._frame_sequences[topic] = sequence
                self._latest_frames[topic] = {
                    "frame": frame,
                    "sequence": sequence,
                    "received_at": time.monotonic(),
                    "capture_time": self._camera_capture_time(frame),
                }
            return None

        return callback

    def rollout_state_callback(self, message):
        text = self._message_text(message)
        pyzlc.info(f"Received rollout state: {text!r}")
        if text == ROLLOUT_START_MESSAGE:
            self._start_detection("roll-out", text)
        elif text == ROLLOUT_END_MESSAGE:
            self._stop_detection("roll-out", text)
        return None

    def reset_state_callback(self, message):
        text = self._message_text(message)
        pyzlc.info(f"Received reset state: {text!r}")
        if text.endswith("starts"):
            self._start_detection("reset", text)
        elif text.endswith("ends"):
            self._stop_detection("reset", text)
        return None

    def _start_detection(self, state_name, message):
        with self._state_lock:
            was_inactive = not self._active_sessions
            if state_name not in self._active_sessions:
                self._session_counter += 1
                session_id = (
                    f"{state_name.replace('-', '_')}_"
                    f"{self._session_counter:04d}_"
                    f"{time.time():.6f}".replace(".", "_")
                )
                self._active_sessions[state_name] = {
                    "state": state_name,
                    "session_id": session_id,
                    "start_message": message,
                    "start_time": time.time(),
                    "records": [],
                }
            else:
                self._active_sessions[state_name]["start_message"] = message

            self._active_state = "+".join(sorted(self._active_sessions))
            self._active_state_message = message
            if was_inactive:
                self._last_processed_sequences = dict(self._frame_sequences)
        print(
            f"[GroundingDINO] {state_name} started; detecting {self.prompt!r} at "
            f"{DETECTION_HZ:.1f} Hz.",
            flush=True,
        )
        pyzlc.info(f"{state_name} detection started from state message: {message!r}")

    def _stop_detection(self, state_name, message):
        session = None
        still_active = False
        with self._state_condition:
            while self._inflight_jobs > 0:
                self._state_condition.wait()
            session = self._active_sessions.pop(state_name, None)
            still_active = bool(self._active_sessions)
            if still_active:
                self._active_state = "+".join(sorted(self._active_sessions))
            else:
                self._active_state = None
                self._active_state_message = ""
        print(f"[GroundingDINO] {state_name} ended; detection stopped.", flush=True)
        pyzlc.info(f"{state_name} detection stopped from state message: {message!r}")
        if session is not None:
            try:
                self._write_keyframes_for_session(session)
            except Exception as exc:
                pyzlc.warning(
                    f"Keyframe scene analysis failed for {state_name} "
                    f"session {session['session_id']}: {exc}"
                )
                self._publish_spatial_relation_sequence(session, [])
        if not still_active:
            self._clear_transient_event_storage()

    def _detection_loop(self):
        while self._running:
            jobs = self._claim_detection_jobs()
            for job in jobs:
                with self._state_condition:
                    self._inflight_jobs += 1
                try:
                    saved_paths = self._save_rgbd_frame(job)
                    names = self._detect_frame(job)
                    self._record_detection(job, names, saved_paths)
                    self._print_detection_result(job, names)
                except Exception as exc:
                    pyzlc.warning(f"Detection failed for {job['topic']}: {exc}")
                finally:
                    with self._state_condition:
                        self._inflight_jobs -= 1
                        self._state_condition.notify_all()
            time.sleep(DETECTION_PERIOD_SECONDS)

    def _claim_detection_jobs(self):
        jobs = []
        now_wall = time.time()
        with self._state_lock:
            if not self._active_sessions:
                return jobs
            active_state_names = sorted(self._active_sessions)

            for topic in CAMERA_TOPICS:
                slot = self._latest_frames.get(topic)
                if slot is None:
                    continue
                if slot["sequence"] <= self._last_processed_sequences.get(topic, 0):
                    continue
                capture_time = slot["capture_time"]
                if capture_time is not None:
                    frame_age = now_wall - capture_time
                    if frame_age > MAX_ACCEPTABLE_FRAME_AGE_SECONDS:
                        continue

                self._last_processed_sequences[topic] = slot["sequence"]
                jobs.append(
                    {
                        "active_state": self._active_state,
                        "active_state_names": active_state_names,
                        "state_message": self._active_state_message,
                        "topic": topic,
                        "frame": slot["frame"],
                        "sequence": slot["sequence"],
                        "capture_time": capture_time,
                        "segmentation_patch": DEFAULT_SEGMENTATION_PATCHES.get(topic),
                        "convert_rgb_to_bgr": CAMERA_COLOR_CONVERSIONS.get(topic, False),
                    }
                )
        return jobs

    def _detect_frame(self, job):
        image = self._frame_rgb_image(
            job["frame"],
            convert_rgb_to_bgr=job["convert_rgb_to_bgr"],
        )
        image, _ = self._segmentation_image_patch(image, job["segmentation_patch"])
        _, image_tensor = self.grounded_sam.load_image(image)
        _, phrases = self.grounded_sam.get_grounding_output(
            self.grounded_sam.model,
            image_tensor,
            self.prompt,
            self.box_threshold,
            self.text_threshold,
            with_logits=False,
            device=self.device,
        )
        return [phrase.strip() for phrase in phrases if phrase and phrase.strip()]

    def _print_detection_result(self, job, names):
        unique_names = []
        seen = set()
        for name in names:
            key = name.lower().rstrip(".")
            if key in seen:
                continue
            unique_names.append(name)
            seen.add(key)

        label = ", ".join(unique_names) if unique_names else "none"
        timestamp = job["capture_time"]
        timestamp_text = f"{timestamp:.6f}" if timestamp is not None else "unknown"
        message = (
            f"[GroundingDINO][{job['active_state']}][{job['topic']}] "
            f"seq={job['sequence']} timestamp={timestamp_text} detected: {label}"
        )
        print(message, flush=True)
        pyzlc.info(message)

    def _prepare_output_dir(self):
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if self.detection_log_path.exists():
            return
        with self.detection_log_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=DETECTION_LOG_COLUMNS)
            writer.writeheader()
        pyzlc.info(f"Saving RGBD frames under {self.images_dir}.")
        pyzlc.info(f"Recording detections to {self.detection_log_path}.")

    def _save_rgbd_frame(self, job):
        frame = job["frame"]
        camera_dir = self.images_dir / self._safe_path_name(job["topic"])
        camera_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self._timestamp_for_log(job)
        timestamp_text = timestamp.replace(".", "_")
        rgb_path = camera_dir / f"{timestamp_text}_rgb.png"
        depth_path = camera_dir / f"{timestamp_text}_depth.png"
        rgbd_path = camera_dir / f"{timestamp_text}_rgbd.npz"

        rgb_image = self._frame_rgb_image(
            frame,
            convert_rgb_to_bgr=job["convert_rgb_to_bgr"],
        )
        try:
            depth_image = self._frame_depth_image(frame)
        except Exception as exc:
            pyzlc.warning(f"Could not parse depth image for {job['topic']}: {exc}")
            depth_image = None

        self._write_rgb_image(rgb_path, rgb_image)
        if depth_image is not None:
            self._write_depth_image(depth_path, self._depth_image_for_png(depth_image))
        else:
            depth_path = None

        np.savez_compressed(
            rgbd_path,
            rgb=rgb_image,
            depth=depth_image if depth_image is not None else np.array([], dtype=np.uint16),
            camera=job["topic"],
            timestamp=timestamp,
            sequence=job["sequence"],
        )

        return {
            "timestamp": timestamp,
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "rgbd_path": rgbd_path,
        }

    def _write_rgb_image(self, path, image):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"RGB image must have shape HxWx3. Got {image.shape}.")
        image = np.array(image[:, :, :3], dtype=np.uint8, copy=True)
        Image.fromarray(image, mode="RGB").save(path)

    def _write_depth_image(self, path, image):
        image = np.asarray(image)
        if image.ndim != 2:
            raise ValueError(f"Depth image must have shape HxW. Got {image.shape}.")
        image = np.array(image, dtype=np.uint16, copy=True)
        Image.fromarray(image).save(path)

    def _record_detection(self, job, names, saved_paths):
        unique_names = self._unique_detection_names(names)
        rows = []
        with self._state_lock:
            for state_name in job["active_state_names"]:
                session = self._active_sessions.get(state_name)
                if session is None:
                    continue
                record = {
                    "state": state_name,
                    "state_message": session["start_message"],
                    "session_id": session["session_id"],
                    "camera": job["topic"],
                    "sequence": job["sequence"],
                    "timestamp": saved_paths["timestamp"],
                    "time_value": self._timestamp_value(saved_paths["timestamp"]),
                    "detected_objects": list(unique_names),
                    "rgb_path": str(saved_paths["rgb_path"]),
                    "depth_path": str(saved_paths["depth_path"] or ""),
                    "rgbd_path": str(saved_paths["rgbd_path"]),
                }
                session["records"].append(record)
                rows.append(
                    {
                        "state": record["state"],
                        "state_message": record["state_message"],
                        "camera": record["camera"],
                        "sequence": record["sequence"],
                        "timestamp": record["timestamp"],
                        "detected_objects": ";".join(record["detected_objects"]),
                        "rgb_path": record["rgb_path"],
                        "depth_path": record["depth_path"],
                        "rgbd_path": record["rgbd_path"],
                    }
                )
        if not rows:
            return
        with self._io_lock:
            with self.detection_log_path.open(
                "a",
                encoding="utf-8",
                newline="",
            ) as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=DETECTION_LOG_COLUMNS)
                writer.writerows(rows)

    def _write_keyframes_for_session(self, session):
        records = list(session["records"])
        if not records:
            pyzlc.warning(
                f"No detection records available for {session['state']} "
                f"session {session['session_id']}."
            )
            self._publish_spatial_relation_sequence(session, [])
            return

        selected_records = self._select_keyframes(records)
        session_dir = self.keyframes_dir / session["session_id"]
        session_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = session_dir / "keyframes.csv"

        rows = []
        for record in selected_records:
            destination_paths = self._copy_keyframe_files(record, session_dir)
            rows.append(
                {
                    "state": record["state"],
                    "session_id": record["session_id"],
                    "camera": record["camera"],
                    "sequence": record["sequence"],
                    "timestamp": record["timestamp"],
                    "reason": ";".join(record["keyframe_reasons"]),
                    "detected_objects": ";".join(record["detected_objects"]),
                    "rgb_path": destination_paths["rgb_path"],
                    "depth_path": destination_paths["depth_path"],
                    "rgbd_path": destination_paths["rgbd_path"],
                    "source_rgb_path": record["rgb_path"],
                    "source_depth_path": record["depth_path"],
                    "source_rgbd_path": record["rgbd_path"],
                }
            )

        with manifest_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=KEYFRAME_LOG_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        message = (
            f"[GroundingDINO][{session['state']}] wrote {len(rows)} key RGBD frames "
            f"for key object {self.key_object!r} to {session_dir}"
        )
        print(message, flush=True)
        pyzlc.info(message)
        self._print_keyframe_objects(session, rows)
        self._analyze_keyframe_spatial_relations(session, rows, session_dir)

    def _print_keyframe_objects(self, session, rows):
        if not rows:
            message = f"[GroundingDINO][{session['state']}] keyframes: none"
            print(message, flush=True)
            pyzlc.info(message)
            return

        header = (
            f"[GroundingDINO][{session['state']}] keyframe objects "
            f"for session {session['session_id']}:"
        )
        print(header, flush=True)
        pyzlc.info(header)
        for row in rows:
            objects = row["detected_objects"] or "none"
            message = (
                f"  [{row['camera']}] timestamp={row['timestamp']} "
                f"reason={row['reason']} objects={objects}"
            )
            print(message, flush=True)
            pyzlc.info(message)

    def _analyze_keyframe_spatial_relations(self, session, rows, session_dir):
        keyframe_groups = self._group_keyframe_rows(rows)
        relation_rows = []
        relation_sequence = []
        for keyframe_index, group_rows in enumerate(keyframe_groups, start=1):
            instances = []
            for row in group_rows:
                try:
                    instances.extend(
                        self._segment_and_project_keyframe_row(
                            row,
                            session_dir=session_dir,
                            keyframe_index=keyframe_index,
                        )
                    )
                except Exception as exc:
                    pyzlc.warning(
                        f"Could not segment/project keyframe {keyframe_index} "
                        f"{row['camera']} timestamp={row['timestamp']}: {exc}"
                    )

            fused_instances = self._fuse_projected_instances(instances)
            spatial_relation = self._spatial_relation_graph(fused_instances)
            self._print_keyframe_spatial_relation(
                session,
                keyframe_index,
                group_rows,
                spatial_relation,
            )
            if spatial_relation["relations"]:
                relation_sequence.append(
                    {
                        "relations": spatial_relation["relations"],
                    }
                )
            relation_rows.append(
                {
                    "keyframe_index": keyframe_index,
                    "cameras": ";".join(row["camera"] for row in group_rows),
                    "timestamps": ";".join(row["timestamp"] for row in group_rows),
                    "objects": ";".join(
                        obj["id"] for obj in spatial_relation["objects"]
                    ),
                    "relations": ";".join(
                        relation["relation"]
                        for relation in spatial_relation["relations"]
                    ),
                }
            )

        relation_path = session_dir / "spatial_relations.csv"
        with relation_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=KEYFRAME_RELATION_LOG_COLUMNS)
            writer.writeheader()
            writer.writerows(relation_rows)
        self._write_spatial_relation_sequence(
            session,
            session_dir,
            relation_sequence,
        )

    def _write_spatial_relation_sequence(self, session, session_dir, relation_sequence):
        sequence_path = session_dir / "spatial_relation_sequence.txt"
        formatted_sequence = "\n".join(
            pformat(item, width=80, sort_dicts=False)
            for item in relation_sequence
        )
        sequence_path.write_text(formatted_sequence + "\n", encoding="utf-8")

        header = (
            f"[SceneGraph][{session['state']}] spatial relation sequence "
            f"for session {session['session_id']}:"
        )
        print(header, flush=True)
        pyzlc.info(header)
        for item in relation_sequence:
            text = pformat(item, width=80, sort_dicts=False)
            print(text, flush=True)
            pyzlc.info(text)
        self._publish_spatial_relation_sequence(session, relation_sequence)

    def _publish_spatial_relation_sequence(self, session, relation_sequence):
        payload = {
            "state": session["state"],
            "session_id": session["session_id"],
            "start_message": session["start_message"],
            "relation_sequence": relation_sequence,
        }
        self.relation_sequence_publisher.publish(payload)
        pyzlc.info(
            f"Published spatial relation sequence with {len(relation_sequence)} "
            "non-empty sample(s)."
        )

    def _group_keyframe_rows(self, rows):
        rows = sorted(rows, key=lambda row: self._timestamp_value(row["timestamp"]))
        unused = list(rows)
        groups = []
        max_pair_delta = max(0.75, self.keyframe_sample_interval * 0.5)

        while unused:
            anchor = unused.pop(0)
            anchor_time = self._timestamp_value(anchor["timestamp"])
            group = [anchor]
            for camera in CAMERA_TOPICS:
                if camera == anchor["camera"]:
                    continue
                candidates = [
                    row for row in unused if row["camera"] == camera
                ]
                if not candidates:
                    continue
                nearest = min(
                    candidates,
                    key=lambda row: abs(
                        self._timestamp_value(row["timestamp"]) - anchor_time
                    ),
                )
                if (
                    abs(self._timestamp_value(nearest["timestamp"]) - anchor_time)
                    <= max_pair_delta
                ):
                    group.append(nearest)
                    unused.remove(nearest)
            groups.append(
                sorted(group, key=lambda row: (row["camera"], row["timestamp"]))
            )
        return groups

    def _segment_and_project_keyframe_row(self, row, session_dir, keyframe_index):
        rgb, depth = self._load_rgbd_npz(row["rgbd_path"])
        if depth is None:
            pyzlc.warning(
                f"Skipping projection for {row['camera']} timestamp={row['timestamp']}: "
                "no depth image."
            )
            return []

        segmentation_image, normalized_patch = self._segmentation_image_patch(
            rgb,
            DEFAULT_SEGMENTATION_PATCHES.get(row["camera"]),
        )
        masks, phrases = self.grounded_sam.segment(
            self.grounded_sam.model,
            segmentation_image,
            self.prompt,
            self.box_threshold,
            self.text_threshold,
            self.device,
            with_logits=False,
        )
        masks, phrases = self._valid_sam_masks_and_phrases(
            masks,
            phrases,
            normalized_patch,
        )
        if masks is None or len(phrases) == 0:
            return []

        depth_uint16 = self._depth_image_for_png(depth)
        instances = []
        mask_dir = (
            session_dir
            / "sam_masks"
            / f"keyframe_{keyframe_index:04d}"
            / self._safe_path_name(row["camera"])
        )
        mask_dir.mkdir(parents=True, exist_ok=True)

        height, width = rgb.shape[:2]
        for instance_index, (mask, phrase) in enumerate(zip(masks, phrases), start=1):
            instance = TableInstance(
                phrase,
                mask,
                rgb=rgb.tobytes(),
                depth=depth_uint16.tobytes(),
                width=width,
                height=height,
                channels=3,
            )
            self._save_instance_mask_outputs(
                instance,
                mask_dir=mask_dir,
                instance_index=instance_index,
            )
            instance.segmented_point_cloud = instance.segmented_point_cloud_in_base(
                config_path=CAMERA_CONFIGS[row["camera"]],
                T_base_hand=None,
                visualize=False,
                depth_trunc=10.0,
                filter_noise=False,
                debug=False,
                show_range=False,
            )
            instances.append(instance)

        pyzlc.info(
            f"Segmented/projected {len(instances)} instances for "
            f"keyframe {keyframe_index} {row['camera']} timestamp={row['timestamp']}."
        )
        return instances

    def _load_rgbd_npz(self, rgbd_path):
        with np.load(rgbd_path, allow_pickle=False) as data:
            rgb = np.asarray(data["rgb"], dtype=np.uint8)
            depth = np.asarray(data["depth"])
        if depth.size == 0:
            depth = None
        return np.ascontiguousarray(rgb[:, :, :3]), depth

    def _valid_sam_masks_and_phrases(self, masks, phrases, patch):
        if masks is None:
            return None, []
        masks = np.asarray(masks)
        if masks.shape[0] == 0:
            return masks, []

        valid_indices = [index for index, phrase in enumerate(phrases) if phrase.strip()]
        if not valid_indices:
            return masks[:0], []
        phrases = [phrases[index].strip() for index in valid_indices]
        masks = masks[valid_indices]
        masks = self._clip_masks_to_patch(masks, patch)
        return masks, phrases

    def _clip_masks_to_patch(self, masks, patch):
        if patch is None or masks is None:
            return masks

        clipped_masks = np.asarray(masks).copy()
        if clipped_masks.ndim == 4 and clipped_masks.shape[1] == 1:
            mask_height, mask_width = clipped_masks.shape[-2:]
            keep = np.zeros((mask_height, mask_width), dtype=bool)
            x_min, y_min, x_max, y_max = patch
            keep[y_min:y_max, x_min:x_max] = True
            clipped_masks &= keep[None, None, :, :]
            return clipped_masks

        if clipped_masks.ndim == 3:
            mask_height, mask_width = clipped_masks.shape[-2:]
            keep = np.zeros((mask_height, mask_width), dtype=bool)
            x_min, y_min, x_max, y_max = patch
            keep[y_min:y_max, x_min:x_max] = True
            clipped_masks &= keep[None, :, :]
            return clipped_masks

        pyzlc.warning(
            f"Could not clip masks to segmentation patch; unexpected mask shape: "
            f"{clipped_masks.shape}"
        )
        return masks

    def _save_instance_mask_outputs(self, instance, mask_dir, instance_index):
        safe_name = self._safe_path_name(instance.name.strip().rstrip(".") or "object")
        prefix = f"{instance_index:02d}_{safe_name}"
        mask = instance.mask_array()
        masked_depth = instance.segment_depth()
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(
            mask_dir / f"{prefix}_mask.png"
        )
        self._write_depth_image(mask_dir / f"{prefix}_masked_depth.png", masked_depth)

    def _fuse_projected_instances(self, instances):
        projected_instances = []
        for instance in instances:
            pcd = getattr(instance, "segmented_point_cloud", None)
            if pcd is not None and len(pcd.points) > 0:
                projected_instances.append(instance)
        instances_by_key = {}
        for instance in projected_instances:
            instances_by_key.setdefault(self._object_key(instance.name), []).append(instance)

        fused_instances = []
        for _, grouped_instances in sorted(instances_by_key.items()):
            fused_instance = grouped_instances[0]
            fused_pcd = None
            for instance in grouped_instances:
                fused_pcd = fused_instance.fuse_projected_point_clouds(
                    fused_pcd,
                    instance.segmented_point_cloud,
                    visualize=False,
                    show_range=False,
                    debug=False,
                )
            fused_instance.segmented_point_cloud = fused_pcd
            if len(fused_pcd.points) > 0:
                fused_instances.append(fused_instance)
        return fused_instances

    def _spatial_relation_graph(self, fused_instances):
        graph = self._empty_spatial_relation()
        seen_object_keys = set()
        for instance in fused_instances:
            self._add_spatial_object(graph, seen_object_keys, instance.name)

        if not fused_instances:
            return graph

        heuristics = TableSceneHeuristics()
        spatial_relations = []
        seen_relations = set()
        placed_instance_keys = set()

        for instance in fused_instances:
            for chosen_instance in fused_instances:
                if chosen_instance.name == instance.name:
                    continue
                if not heuristics.is_in(instance, chosen_instance):
                    continue

                subject_name = heuristics.spatial_relation_instance_name(instance)
                target_name = heuristics.spatial_relation_instance_name(chosen_instance)
                relation = f"{subject_name} is in {target_name}"
                if relation not in seen_relations:
                    spatial_relations.append(relation)
                    seen_relations.add(relation)
                placed_instance_keys.add(self._object_key(instance.name))

        for instance in fused_instances:
            if self._object_key(instance.name) in placed_instance_keys:
                continue
            for chosen_instance in fused_instances:
                if chosen_instance.name == instance.name:
                    continue
                if not heuristics.is_on(instance, chosen_instance):
                    continue

                subject_name = heuristics.spatial_relation_instance_name(instance)
                target_name = heuristics.spatial_relation_instance_name(chosen_instance)
                relation = f"{subject_name} on {target_name}"
                if relation not in seen_relations:
                    spatial_relations.append(relation)
                    seen_relations.add(relation)
                placed_instance_keys.add(self._object_key(instance.name))
                break

        for instance in fused_instances:
            if self._object_key(instance.name) in placed_instance_keys:
                continue
            if not heuristics.is_on_table(instance):
                continue

            subject_name = heuristics.spatial_relation_instance_name(instance)
            relation = f"{subject_name} on table"
            if relation not in seen_relations:
                spatial_relations.append(relation)
                seen_relations.add(relation)
                self._add_spatial_object(graph, seen_object_keys, "table")

        for relation in spatial_relations:
            graph["relations"].append({"relation": relation})
        return graph

    def _print_keyframe_spatial_relation(
        self,
        session,
        keyframe_index,
        group_rows,
        spatial_relation,
    ):
        cameras = ", ".join(row["camera"] for row in group_rows)
        timestamps = ", ".join(row["timestamp"] for row in group_rows)
        objects = [obj["id"] for obj in spatial_relation["objects"]]
        relations = [relation["relation"] for relation in spatial_relation["relations"]]
        header = (
            f"[SceneGraph][{session['state']}] keyframe={keyframe_index:04d} "
            f"cameras={cameras} timestamps={timestamps}"
        )
        print(header, flush=True)
        pyzlc.info(header)
        object_line = "  objects: " + (", ".join(objects) if objects else "none")
        print(object_line, flush=True)
        pyzlc.info(object_line)
        if relations:
            for relation in relations:
                line = f"  relation: {relation}"
                print(line, flush=True)
                pyzlc.info(line)
        else:
            line = "  relations: none"
            print(line, flush=True)
            pyzlc.info(line)

    def _empty_spatial_relation(self):
        return {
            "objects": [],
            "relations": [],
        }

    def _add_spatial_object(self, graph, seen_object_keys, object_id):
        object_key = self._object_key(object_id)
        if object_key in seen_object_keys:
            return
        graph["objects"].append({"id": object_id})
        seen_object_keys.add(object_key)

    def _clear_transient_event_storage(self):
        with self._io_lock:
            if self.images_dir.exists():
                shutil.rmtree(self.images_dir)
            self.images_dir.mkdir(parents=True, exist_ok=True)
            with self.detection_log_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=DETECTION_LOG_COLUMNS)
                writer.writeheader()
        pyzlc.info(
            "Cleared transient RGBD images and detection records for the next event."
        )

    def _select_keyframes(self, records):
        selected = {}
        for camera in CAMERA_TOPICS:
            camera_records = sorted(
                (record for record in records if record["camera"] == camera),
                key=lambda record: record["time_value"],
            )
            self._add_key_object_transition_frames(selected, camera_records)
            self._add_sampled_frames(selected, camera_records)

        return sorted(
            selected.values(),
            key=lambda record: (record["time_value"], record["camera"], record["sequence"]),
        )

    def _add_key_object_transition_frames(self, selected, records):
        previous_present = False
        for record in records:
            present = self._record_has_key_object(record)
            if present and not previous_present:
                self._mark_keyframe(selected, record, f"{self.key_object}_appears")
            elif previous_present and not present:
                self._mark_keyframe(selected, record, f"{self.key_object}_disappears")
            previous_present = present

    def _add_sampled_frames(self, selected, records):
        next_sample_time = None
        sample_reason = f"sample_{self.keyframe_sample_interval:.3f}s"
        for record in records:
            if next_sample_time is None or record["time_value"] >= next_sample_time:
                self._mark_keyframe(selected, record, sample_reason)
                next_sample_time = (
                    record["time_value"] + self.keyframe_sample_interval
                )

    def _mark_keyframe(self, selected, record, reason):
        key = (record["camera"], record["timestamp"], record["sequence"])
        if key not in selected:
            selected[key] = dict(record)
            selected[key]["keyframe_reasons"] = []
        if reason not in selected[key]["keyframe_reasons"]:
            selected[key]["keyframe_reasons"].append(reason)

    def _copy_keyframe_files(self, record, session_dir):
        camera_dir = session_dir / self._safe_path_name(record["camera"])
        camera_dir.mkdir(parents=True, exist_ok=True)

        copied_paths = {}
        for field in ("rgb_path", "depth_path", "rgbd_path"):
            source = record.get(field, "")
            if not source:
                copied_paths[field] = ""
                continue

            source_path = Path(source)
            destination = camera_dir / source_path.name
            if source_path.exists():
                shutil.copy2(source_path, destination)
                copied_paths[field] = str(destination)
            else:
                pyzlc.warning(f"Keyframe source file does not exist: {source_path}")
                copied_paths[field] = ""
        return copied_paths

    def _record_has_key_object(self, record):
        key_object = self._object_key(self.key_object)
        return any(
            self._object_key(name) == key_object
            for name in record["detected_objects"]
        )

    def _frame_rgb_image(self, frame, convert_rgb_to_bgr=False):
        width = int(frame["width"])
        height = int(frame["height"])
        channels = int(frame["channels"])
        rgb_data = frame["rgb_data"]
        image = np.frombuffer(rgb_data, dtype=np.uint8).reshape((height, width, channels))
        image = image[:, :, :3]
        if convert_rgb_to_bgr:
            image = image[:, :, ::-1]
        return np.ascontiguousarray(image)

    def _frame_depth_image(self, frame):
        depth_data = frame.get("depth_data")
        if depth_data is None:
            return None

        width = int(frame["width"])
        height = int(frame["height"])
        raw = bytes(depth_data)
        pixel_count = height * width

        if len(raw) == pixel_count * np.dtype(np.uint16).itemsize:
            depth = np.frombuffer(raw, dtype=np.uint16).reshape((height, width))
        elif len(raw) == pixel_count * np.dtype(np.float32).itemsize:
            depth = np.frombuffer(raw, dtype=np.float32).reshape((height, width))
        else:
            raise ValueError(
                f"depth_data has {len(raw)} bytes, expected uint16 or float32 "
                f"depth for shape ({height}, {width})."
            )
        return np.ascontiguousarray(depth)

    def _depth_image_for_png(self, depth_image):
        if depth_image.dtype == np.uint16:
            return depth_image
        if np.issubdtype(depth_image.dtype, np.floating):
            finite_depth = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(finite_depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(
                np.uint16
            )
        return np.clip(depth_image, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    def _segmentation_image_patch(self, image, patch):
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] > 3:
            image = image[:, :, :3]
        image = np.ascontiguousarray(image)

        normalized_patch = self._normalize_segmentation_patch(
            patch,
            width=image.shape[1],
            height=image.shape[0],
        )
        if normalized_patch is None:
            return image, None

        x_min, y_min, x_max, y_max = normalized_patch
        patched_image = np.zeros_like(image)
        patched_image[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]
        return patched_image, normalized_patch

    def _normalize_segmentation_patch(self, patch, width, height):
        if patch is None:
            return None
        if isinstance(patch, str):
            patch = [part.strip() for part in patch.split(",")]
        if len(patch) != 4:
            raise ValueError("Segmentation patch must be [x_min, y_min, x_max, y_max].")

        x_min, y_min, x_max, y_max = (int(round(float(value))) for value in patch)
        x_min = max(0, min(width, x_min))
        x_max = max(0, min(width, x_max))
        y_min = max(0, min(height, y_min))
        y_max = max(0, min(height, y_max))
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"Segmentation patch is empty after clamping to image shape "
                f"{width}x{height}: {(x_min, y_min, x_max, y_max)}"
            )
        return x_min, y_min, x_max, y_max

    def _camera_capture_time(self, frame):
        timestamp = frame.get("timestamp")
        if isinstance(timestamp, bool) or timestamp is None:
            return None
        try:
            capture_time = float(timestamp)
        except (TypeError, ValueError):
            return None
        if not 1_000_000_000.0 <= capture_time <= 10_000_000_000.0:
            return None
        return capture_time

    def _message_text(self, message):
        if isinstance(message, bytes):
            return message.decode("utf-8", errors="replace").strip()
        return str(message).strip()

    def _timestamp_for_log(self, job):
        timestamp = job["capture_time"]
        if timestamp is None:
            timestamp = job.setdefault("fallback_timestamp", time.time())
            return f"received_{timestamp:.6f}"
        return f"{timestamp:.6f}"

    def _safe_path_name(self, name):
        return "".join(char if char.isalnum() or char in "-_." else "_" for char in name)

    def _timestamp_value(self, timestamp):
        timestamp = str(timestamp)
        if timestamp.startswith("received_"):
            timestamp = timestamp.removeprefix("received_")
        try:
            return float(timestamp)
        except ValueError:
            return time.time()

    def _unique_detection_names(self, names):
        unique_names = []
        seen = set()
        for name in names:
            key = self._object_key(name)
            if key in seen:
                continue
            unique_names.append(name)
            seen.add(key)
        return unique_names

    def _object_key(self, name):
        return str(name).strip().lower().rstrip(".")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Subscriber node that runs GroundingDINO/SAM vocab detection on "
            "static and ZED camera frames while roll-out/reset state topics are active."
        )
    )
    parser.add_argument("--node-ip", default="141.3.53.25")
    parser.add_argument("--group-name", default="robot_lab_robotiq_202")
    parser.add_argument("--group-port", type=int, default=7725)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--key-object",
        default=DEFAULT_KEY_OBJECT,
        help="Object name used to mark appear/disappear keyframes.",
    )
    parser.add_argument(
        "--keyframe-sample-interval",
        type=float,
        default=DEFAULT_KEYFRAME_SAMPLE_INTERVAL_SECONDS,
        help="Seconds between regularly sampled RGBD keyframes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for timestamp-named RGBD frames and detections.csv.",
    )
    parser.add_argument(
        "--relation-sequence-topic",
        default=SPATIAL_RELATION_SEQUENCE_TOPIC,
        help="Topic used to publish the event-end spatial relation sequence.",
    )
    args = parser.parse_args()
    if args.keyframe_sample_interval <= 0:
        parser.error("--keyframe-sample-interval must be positive")
    if not args.key_object.strip():
        parser.error("--key-object cannot be empty")
    args.key_object = args.key_object.strip()
    return args


if __name__ == "__main__":
    args = parse_args()
    GroundedDinoVocabDetectionNode(
        prompt=args.prompt,
        node_ip=args.node_ip,
        group_name=args.group_name,
        group_port=args.group_port,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        output_dir=args.output_dir,
        key_object=args.key_object,
        keyframe_sample_interval=args.keyframe_sample_interval,
        relation_sequence_topic=args.relation_sequence_topic,
    )
    pyzlc.spin()
