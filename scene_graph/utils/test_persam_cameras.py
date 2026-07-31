#!/usr/bin/env python3
"""Test PerSAM target-object segmentation on static and ZED camera images.

Examples:
  python scene_graph/utils/test_persam_cameras.py \
    --persam-root /home/jjiang/jing/Personalize-SAM \
    --ref-image /path/to/target_ref.jpg \
    --ref-mask /path/to/target_ref_mask.png \
    --static-image /path/to/static_cam.png \
    --zed-image /path/to/zed_cam.png

  python scene_graph/utils/test_persam_cameras.py \
    --live \
    --persam-root /home/jjiang/jing/Personalize-SAM \
    --ref-image /path/to/target_ref.jpg \
    --ref-mask /path/to/target_ref_mask.png
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from torch.nn import functional as F


DEFAULT_GROUP_NAME = "robot_lab_robotiq_202"
DEFAULT_GROUP_PORT = 7725
DEFAULT_NODE_IP = "141.3.53.25"
DEFAULT_STATIC_TOPIC = "static_cam"
DEFAULT_ZED_TOPIC = "zed_depth"
DEFAULT_SAM_CHECKPOINT = Path(
    "/home/jjiang/jing/Grounded-Segment-Anything/sam_vit_b_01ec64.pth"
)


@dataclass
class CameraInput:
    name: str
    image: np.ndarray
    timestamp: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segment the same target object in static_cam and ZED camera images "
            "using a one-shot PerSAM reference image and mask."
        )
    )
    parser.add_argument(
        "--persam-root",
        type=Path,
        default=None,
        help=(
            "Path to the Personalize-SAM checkout. Required if "
            "per_segment_anything is not already importable."
        ),
    )
    parser.add_argument("--ref-image", type=Path, required=True)
    parser.add_argument("--ref-mask", type=Path, required=True)
    parser.add_argument(
        "--static-image",
        type=Path,
        default=None,
        help="Saved static_cam image. Omit with --live to capture from pyzlc.",
    )
    parser.add_argument(
        "--zed-image",
        type=Path,
        default=None,
        help="Saved ZED camera image. Omit with --live to capture from pyzlc.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Subscribe to camera topics instead of reading saved images.",
    )
    parser.add_argument("--node-ip", default=DEFAULT_NODE_IP)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--group-port", type=int, default=DEFAULT_GROUP_PORT)
    parser.add_argument("--static-topic", default=DEFAULT_STATIC_TOPIC)
    parser.add_argument("--zed-topic", default=DEFAULT_ZED_TOPIC)
    parser.add_argument("--frame-timeout", type=float, default=10.0)
    parser.add_argument("--latest-frame-window", type=float, default=0.2)
    parser.add_argument(
        "--static-color-order",
        choices=("bgr", "rgb"),
        default="bgr",
        help="Color order used by static_cam rgb_data. DepthAI publishes BGR.",
    )
    parser.add_argument(
        "--zed-color-order",
        choices=("bgr", "rgb"),
        default="rgb",
        help="Color order used by the ZED camera rgb_data.",
    )
    parser.add_argument("--sam-type", default="vit_b")
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=DEFAULT_SAM_CHECKPOINT,
        help="SAM checkpoint used by PerSAM.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for PerSAM inference.",
    )
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/persam_test"))
    args = parser.parse_args()

    if args.frame_timeout <= 0:
        parser.error("--frame-timeout must be positive")
    if args.latest_frame_window < 0:
        parser.error("--latest-frame-window cannot be negative")
    if args.topk <= 0:
        parser.error("--topk must be positive")
    if not args.live and args.static_image is None and args.zed_image is None:
        parser.error("provide --static-image and/or --zed-image, or use --live")
    return args


def import_persam(persam_root: Path | None):
    if persam_root is not None:
        sys.path.insert(0, str(persam_root.expanduser().resolve()))

    try:
        from per_segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise SystemExit(
            "Could not import PerSAM's per_segment_anything package. Clone "
            "https://github.com/ZrrSkywalker/Personalize-SAM and pass "
            "--persam-root /path/to/Personalize-SAM."
        ) from exc

    return SamPredictor, sam_model_registry


def read_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_binary_mask(path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if mask.shape != image_shape:
        mask = cv2.resize(
            mask,
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask > 0


def frame_to_rgb(frame: Mapping[str, Any], color_order: str) -> np.ndarray:
    width = int(frame["width"])
    height = int(frame["height"])
    channels = int(frame.get("channels", 3))
    raw = bytes(frame["rgb_data"])
    expected = width * height * channels
    if len(raw) != expected:
        raise ValueError(
            f"Camera frame has {len(raw)} bytes, expected {expected} "
            f"for shape ({height}, {width}, {channels})."
        )

    image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels)
    if channels == 1:
        image = np.repeat(image, 3, axis=2)
    else:
        image = image[:, :, :3]

    if color_order == "bgr":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


class LatestFrameReceiver:
    def __init__(self) -> None:
        self.frame: Mapping[str, Any] | None = None
        self._event = threading.Event()
        self._lock = threading.Lock()

    def callback(self, frame: Mapping[str, Any]) -> None:
        with self._lock:
            current_timestamp = self._capture_time(self.frame)
            next_timestamp = self._capture_time(frame)
            if current_timestamp is not None and next_timestamp is not None:
                if next_timestamp <= current_timestamp:
                    return None
            self.frame = frame
            self._event.set()
        return None

    def wait(self, timeout: float, latest_frame_window: float) -> Mapping[str, Any]:
        if not self._event.wait(timeout):
            raise TimeoutError(f"No frame received within {timeout:.1f}s.")
        time.sleep(latest_frame_window)
        with self._lock:
            if self.frame is None:
                raise RuntimeError("Frame receiver woke up without a frame.")
            return self.frame

    @staticmethod
    def _capture_time(frame: Mapping[str, Any] | None) -> float | None:
        if frame is None:
            return None
        try:
            return float(frame.get("timestamp"))
        except (TypeError, ValueError):
            return None


def load_live_camera_inputs(args: argparse.Namespace) -> list[CameraInput]:
    try:
        import pyzlc
    except ImportError as exc:
        raise SystemExit("--live requires pyzlc to be installed.") from exc

    pyzlc.init(
        "persam_camera_test",
        args.node_ip,
        args.group_name,
        group_port=args.group_port,
    )
    pyzlc.get_node(args.group_name).subscriber_manager.local_ip = ""

    receivers = {
        "static_cam": (args.static_topic, args.static_color_order, LatestFrameReceiver()),
        "zed_cam": (args.zed_topic, args.zed_color_order, LatestFrameReceiver()),
    }
    for _, (topic, _, receiver) in receivers.items():
        pyzlc.register_subscriber_handler(
            topic,
            receiver.callback,
            args.group_name,
            buffer_size=1,
            conflate=True,
        )

    camera_inputs: list[CameraInput] = []
    for name, (topic, color_order, receiver) in receivers.items():
        print(f"Waiting for {name} frame on topic {topic!r}...")
        frame = receiver.wait(args.frame_timeout, args.latest_frame_window)
        camera_inputs.append(
            CameraInput(
                name=name,
                image=frame_to_rgb(frame, color_order),
                timestamp=frame.get("timestamp"),
            )
        )
    return camera_inputs


def load_saved_camera_inputs(args: argparse.Namespace) -> list[CameraInput]:
    camera_inputs: list[CameraInput] = []
    if args.static_image is not None:
        camera_inputs.append(CameraInput("static_cam", read_rgb_image(args.static_image)))
    if args.zed_image is not None:
        camera_inputs.append(CameraInput("zed_cam", read_rgb_image(args.zed_image)))
    return camera_inputs


def point_selection(mask_sim: torch.Tensor, topk: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask_sim.shape
    flat = mask_sim.flatten()
    topk = min(topk, flat.numel())

    positive = flat.topk(topk).indices
    positive_xy = torch.stack((positive % width, positive // width), dim=1)
    positive_labels = np.ones(topk, dtype=np.int64)

    negative = flat.topk(topk, largest=False).indices
    negative_xy = torch.stack((negative % width, negative // width), dim=1)
    negative_labels = np.zeros(topk, dtype=np.int64)

    points = torch.cat((positive_xy, negative_xy), dim=0).cpu().numpy()
    labels = np.concatenate((positive_labels, negative_labels), axis=0)
    return points, labels


class PerSAMSegmenter:
    def __init__(
        self,
        sam_predictor_cls,
        sam_model_registry,
        sam_type: str,
        checkpoint: Path,
        device: str,
        topk: int,
    ) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint does not exist: {checkpoint}")
        self.device = device
        self.topk = topk
        sam = sam_model_registry[sam_type](checkpoint=str(checkpoint)).to(device)
        sam.eval()
        self.predictor = sam_predictor_cls(sam)
        self.target_feat: torch.Tensor | None = None
        self.target_embedding: torch.Tensor | None = None

    @torch.inference_mode()
    def set_reference(self, image: np.ndarray, mask: np.ndarray) -> None:
        mask_image = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint8) * 255

        try:
            encoded_mask = self.predictor.set_image(image, mask_image)
        except TypeError as exc:
            raise RuntimeError(
                "The imported SamPredictor does not look like the PerSAM "
                "version. Make sure --persam-root points to Personalize-SAM, "
                "not the regular segment-anything package."
            ) from exc

        ref_feat = self.predictor.features.squeeze().permute(1, 2, 0)
        encoded_mask = F.interpolate(
            encoded_mask,
            size=ref_feat.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()[0]
        target_pixels = ref_feat[encoded_mask > 0]
        if target_pixels.numel() == 0:
            raise ValueError("Reference mask is empty after PerSAM resizing.")

        target_embedding = target_pixels.mean(0).unsqueeze(0)
        self.target_feat = target_embedding / target_embedding.norm(
            dim=-1,
            keepdim=True,
        )
        self.target_embedding = target_embedding.unsqueeze(0)

    @torch.inference_mode()
    def segment(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.target_feat is None or self.target_embedding is None:
            raise RuntimeError("set_reference must be called before segment.")

        self.predictor.set_image(image)
        test_feat = self.predictor.features.squeeze()
        channels, height, width = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(channels, height * width)

        sim = self.target_feat @ test_feat
        sim = sim.reshape(1, 1, height, width)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.predictor.model.postprocess_masks(
            sim,
            input_size=self.predictor.input_size,
            original_size=self.predictor.original_size,
        ).squeeze()

        points, labels = point_selection(sim, topk=self.topk)
        attn_sim = (sim - sim.mean()) / torch.std(sim).clamp_min(1e-6)
        attn_sim = F.interpolate(
            attn_sim.unsqueeze(0).unsqueeze(0),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )
        attn_sim = attn_sim.sigmoid_().unsqueeze(0).flatten(3)

        masks, _, logits, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False,
            attn_sim=attn_sim,
            target_embedding=self.target_embedding,
        )

        masks, scores, logits, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=logits[0:1, :, :],
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))

        ys, xs = np.nonzero(masks[best_idx])
        if len(xs) == 0 or len(ys) == 0:
            return masks[best_idx].astype(bool), points, labels

        input_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        masks, scores, _, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=input_box[None, :],
            mask_input=logits[best_idx : best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool), points, labels


def save_outputs(
    output_dir: Path,
    camera_input: CameraInput,
    mask: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = camera_input.name
    mask_path = output_dir / f"{stem}_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    image_path = output_dir / f"{stem}_rgb.png"

    cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
    cv2.imwrite(str(image_path), cv2.cvtColor(camera_input.image, cv2.COLOR_RGB2BGR))

    overlay = camera_input.image.copy()
    color = np.array([255, 64, 64], dtype=np.uint8)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    for (x, y), label in zip(points.astype(int), labels.astype(int)):
        point_color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(overlay, (int(x), int(y)), 5, point_color, thickness=-1)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(
        f"{stem}: mask_pixels={int(mask.sum())}, "
        f"mask={mask_path}, overlay={overlay_path}"
    )


def main() -> None:
    args = parse_args()
    SamPredictor, sam_model_registry = import_persam(args.persam_root)

    ref_image = read_rgb_image(args.ref_image)
    ref_mask = read_binary_mask(args.ref_mask, ref_image.shape[:2])
    camera_inputs = (
        load_live_camera_inputs(args) if args.live else load_saved_camera_inputs(args)
    )

    segmenter = PerSAMSegmenter(
        SamPredictor,
        sam_model_registry,
        sam_type=args.sam_type,
        checkpoint=args.sam_checkpoint,
        device=args.device,
        topk=args.topk,
    )
    segmenter.set_reference(ref_image, ref_mask)

    for camera_input in camera_inputs:
        mask, points, labels = segmenter.segment(camera_input.image)
        save_outputs(args.output_dir, camera_input, mask, points, labels)


if __name__ == "__main__":
    main()
