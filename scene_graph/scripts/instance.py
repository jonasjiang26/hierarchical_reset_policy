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
        self.segemtned_point_cloud = None

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
        filter_noise=True,
        outlier_nb_neighbors=30,
        outlier_std_ratio=1.5,
        cluster_eps=0.02,
        cluster_min_points=20,
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

        if filter_noise:
            pcd = self.filter_point_cloud_noise(
                pcd,
                outlier_nb_neighbors=outlier_nb_neighbors,
                outlier_std_ratio=outlier_std_ratio,
                cluster_eps=cluster_eps,
                cluster_min_points=cluster_min_points,
                debug=debug,
            )
        self.segemtned_point_cloud = pcd
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

    def filter_point_cloud_noise(
        self,
        pcd,
        outlier_nb_neighbors=20,
        outlier_std_ratio=6.0,
        cluster_eps=0.08,
        cluster_min_points=10,
        debug=True,
    ):
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return pcd

        finite_mask = np.isfinite(points).all(axis=1)
        if not finite_mask.all():
            pcd = pcd.select_by_index(np.flatnonzero(finite_mask))
            points = np.asarray(pcd.points)

        if len(points) < max(3, outlier_nb_neighbors):
            return pcd

        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=outlier_nb_neighbors,
            std_ratio=outlier_std_ratio,
        )

        points = np.asarray(pcd.points)
        if len(points) < cluster_min_points:
            if debug:
                removed = len(finite_mask) - len(points)
                print(f"[{self.name}] noise filter kept {len(points)} points, removed {removed}.")
            return pcd

        labels = np.asarray(
            pcd.cluster_dbscan(
                eps=cluster_eps,
                min_points=cluster_min_points,
                print_progress=False,
            )
        )
        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            if debug:
                print(f"[{self.name}] noise filter found no connected cluster; kept statistical inliers.")
            return pcd

        cluster_ids, cluster_counts = np.unique(valid_labels, return_counts=True)
        largest_cluster = cluster_ids[np.argmax(cluster_counts)]
        cluster_indices = np.flatnonzero(labels == largest_cluster)
        filtered_pcd = pcd.select_by_index(cluster_indices)

        if debug:
            removed = len(finite_mask) - len(cluster_indices)
            print(
                f"[{self.name}] noise filter kept {len(cluster_indices)} points "
                f"from largest cluster, removed {removed}."
            )

        return filtered_pcd

    def fuse_projected_point_clouds(
        self,
        first_pcd,
        second_pcd,
        voxel_size=0.003,
        filter_noise=False,
        outlier_nb_neighbors=30,
        outlier_std_ratio=1.5,
        cluster_eps=0.015,
        cluster_min_points=30,
        visualize=True,
        show_range=True,
        debug=True,
        frame_size=0.1,
    ):
        point_clouds = [pcd for pcd in (first_pcd, second_pcd) if pcd is not None]
        if len(point_clouds) == 0:
            raise ValueError("At least one point cloud is required for fusion.")

        fused_pcd = o3d.geometry.PointCloud()
        for pcd in point_clouds:
            fused_pcd += pcd

        if voxel_size is not None and voxel_size > 0.0 and len(fused_pcd.points) > 0:
            fused_pcd = fused_pcd.voxel_down_sample(voxel_size=voxel_size)

        if filter_noise:
            fused_pcd = self.filter_point_cloud_noise(
                fused_pcd,
                outlier_nb_neighbors=outlier_nb_neighbors,
                outlier_std_ratio=outlier_std_ratio,
                cluster_eps=cluster_eps,
                cluster_min_points=cluster_min_points,
                debug=debug,
            )

        points = np.asarray(fused_pcd.points)
        if show_range:
            if len(points) == 0:
                print(f"[{self.name}] fused point cloud is empty.")
            else:
                min_bound = points.min(axis=0)
                max_bound = points.max(axis=0)
                print(
                    f"[{self.name}] fused point cloud: "
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
            geometries = [frame] if len(points) == 0 else [fused_pcd, frame]
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"{self.name} fused projected point cloud",
            )

        return fused_pcd

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

    @staticmethod
    def arm_state_to_base_hand_transform(arm_state):
        if "O_T_EE" in arm_state:
            values = np.asarray(arm_state["O_T_EE"], dtype=np.float64)
            if values.size != 16:
                raise ValueError(f"O_T_EE must contain 16 values, got {values.size}")
            return values.reshape((4, 4), order="F")

        ee_pos = arm_state.get("EE_pos", None)
        ee_quat = arm_state.get("EE_quat", None)
        if ee_pos is None or ee_quat is None:
            raise ValueError("arm_state must contain O_T_EE or both EE_pos and EE_quat")

        T_base_hand = np.eye(4, dtype=np.float64)
        T_base_hand[:3, :3] = TableInstance.quat_xyzw_to_matrix(ee_quat)
        T_base_hand[:3, 3] = np.asarray(ee_pos, dtype=np.float64)
        return T_base_hand

    @staticmethod
    def quat_xyzw_to_matrix(quat):
        x, y, z, w = np.asarray(quat, dtype=np.float64)
        norm = np.linalg.norm([x, y, z, w])
        if norm == 0.0:
            raise ValueError("EE_quat has zero norm")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
