import numpy as np
import cv2
from pathlib import Path
import open3d as o3d
import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

class TableInstance:
    def __init__(self, name, mask, rgb, depth, width, height, channels):
        self.depth_data = depth
        self.rgb_data = rgb
        self.mask = mask
        self.name = name
        self.width = width
        self.height = height
        self.channels = channels

    def mask_array(self):
        mask = self.mask
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()

        mask = np.asarray(mask)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]

        return mask.astype(bool)

    def rgb_array(self):
        return np.frombuffer(self.rgb_data, dtype=np.uint8).reshape(
            (self.height, self.width, self.channels)
        )

    def depth_array(self):
        return np.frombuffer(self.depth_data, dtype=np.uint16).reshape(
            (self.height, self.width)
        )

    def segment_rgb(self):
        mask = self.mask_array()
        rgb = self.rgb_array()
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"mask shape {mask.shape} does not match RGB shape {rgb.shape[:2]}")

        masked_rgb = np.zeros_like(rgb)
        masked_rgb[mask] = rgb[mask]
        return masked_rgb

    def segment_depth(self):
        mask = self.mask_array()
        depth = self.depth_array()
        if mask.shape != depth.shape:
            raise ValueError(f"mask shape {mask.shape} does not match depth shape {depth.shape}")

        masked_depth = np.zeros_like(depth)
        masked_depth[mask] = depth[mask]
        return masked_depth

    def segmented_point_cloud_in_base(
        self,
        config_path=None,
        T_base_hand=None,
        depth_scale=1000.0,
        depth_trunc=3.0,
        visualize=False,
        show_range=True,
        debug=True,
        frame_size=0.1,
    ):
        if config_path is None:
            config_path = CONFIG_DIR / "eye_to_hand.yaml"

        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        intrinsics = config["intrinsics"]
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.width,
            self.height,
            intrinsics["fx"],
            intrinsics["fy"],
            intrinsics["cx"],
            intrinsics["cy"],
        )

        masked_rgb = self.segment_rgb()
        masked_depth = self.segment_depth()

        if debug:
            self._print_depth_debug(masked_depth, depth_scale, depth_trunc)

        rgb_o3d = o3d.geometry.Image(masked_rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(masked_depth.astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            camera_intrinsic,
        )

        T_base_camera = self._base_camera_transform(config, T_base_hand)
        pcd.transform(T_base_camera)

        points = np.asarray(pcd.points)
        if show_range:
            if len(points) == 0:
                print(f"[{self.name}] point cloud in base is empty.")
            else:
                min_bound = points.min(axis=0)
                max_bound = points.max(axis=0)
                print(
                    f"[{self.name}] point cloud in base: "
                    f"count={len(points)}, "
                    f"x=[{min_bound[0]:.4f}, {max_bound[0]:.4f}], "
                    f"y=[{min_bound[1]:.4f}, {max_bound[1]:.4f}], "
                    f"z=[{min_bound[2]:.4f}, {max_bound[2]:.4f}]"
                )

        if visualize:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=frame_size,
                origin=[0.0, 0.0, 0.0],
            )
            geometries = [frame] if len(points) == 0 else [pcd, frame]
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"{self.name} segmented point cloud in base",
            )

        return pcd

    def _print_depth_debug(self, masked_depth, depth_scale, depth_trunc):
        mask = self.mask_array()
        depth = self.depth_array()
        raw_nonzero = depth[depth > 0]
        masked_nonzero = masked_depth[masked_depth > 0]
        print(
            f"[{self.name}] mask_pixels={int(mask.sum())}, "
            f"raw_depth_nonzero={len(raw_nonzero)}, "
            f"masked_depth_nonzero={len(masked_nonzero)}, "
            f"depth_dtype={depth.dtype}, depth_scale={depth_scale}, depth_trunc={depth_trunc}"
        )

        if len(raw_nonzero) > 0:
            print(
                f"[{self.name}] raw depth range: "
                f"[{raw_nonzero.min()}, {raw_nonzero.max()}] raw, "
                f"[{raw_nonzero.min() / depth_scale:.4f}, {raw_nonzero.max() / depth_scale:.4f}] m"
            )

        if len(masked_nonzero) > 0:
            print(
                f"[{self.name}] masked depth range: "
                f"[{masked_nonzero.min()}, {masked_nonzero.max()}] raw, "
                f"[{masked_nonzero.min() / depth_scale:.4f}, {masked_nonzero.max() / depth_scale:.4f}] m"
            )
        else:
            print(f"[{self.name}] masked depth is all zero, so Open3D will create an empty point cloud.")

    def _base_camera_transform(self, config, T_base_hand=None):
        calibration_type = config.get("calibration_type")
        transforms = config.get("transforms", {})

        if calibration_type == "eye_to_hand":
            return np.asarray(transforms["T_base_camera"]["matrix"], dtype=np.float64)

        if calibration_type == "eye_in_hand":
            if T_base_hand is None:
                raise ValueError(
                    "T_base_hand is required for eye_in_hand wrist camera projection."
                )
            T_base_hand = np.asarray(T_base_hand, dtype=np.float64)
            T_hand_camera = np.asarray(
                transforms["T_hand_camera"]["matrix"],
                dtype=np.float64,
            )
            return T_base_hand @ T_hand_camera

        raise ValueError(f"Unsupported calibration_type: {calibration_type}")
