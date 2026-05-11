import numpy as np

from instance import TableInstance


class TableSceneHeuristics:
    def __init__(self):
        self._world_bbox_cache = {}

    def _get_world_bbox(self, instance: TableInstance):
        cache_key = id(instance)
        if cache_key in self._world_bbox_cache:
            return self._world_bbox_cache[cache_key]
        point_cloud = getattr(instance, "segmented_point_cloud", None)
        bbox = point_cloud.get_axis_aligned_bounding_box()
        xyz_min = bbox.get_min_bound()
        xyz_max = bbox.get_max_bound()
        self._world_bbox_cache[cache_key] = (xyz_min, xyz_max)
        return xyz_min, xyz_max

    def is_on_table(self, instance: TableInstance, tabletop_high=-0.01, tabletop_low=-0.05):
        xyz_min, _ = self._get_world_bbox(instance)
        return tabletop_low < xyz_min[2] < tabletop_high

    def is_on(self, obj1: TableInstance, obj2: TableInstance, height_threshold=0.02):
        bbox_1_min, bbox_1_max = self._get_world_bbox(obj1)
        bbox_2_min, bbox_2_max = self._get_world_bbox(obj2)

        obj1_bottom = bbox_1_min[2]
        obj2_top = bbox_2_max[2]
        if abs(obj1_bottom - obj2_top) > height_threshold:
            return False

        center1 = (bbox_1_min + bbox_1_max) / 2.0
        return (
            bbox_2_min[0] <= center1[0] <= bbox_2_max[0]
            and bbox_2_min[1] <= center1[1] <= bbox_2_max[1]
        )

    def is_in(self, obj1: TableInstance, obj2: TableInstance, containment_ratio=0.8):
        bbox_1_min, bbox_1_max = self._get_world_bbox(obj1)
        bbox_2_min, bbox_2_max = self._get_world_bbox(obj2)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)
        if np.any(overlap_max < overlap_min):
            return False

        overlap_volume = np.prod(overlap_max - overlap_min)
        obj1_volume = np.prod(bbox_1_max - bbox_1_min)
        return obj1_volume > 0 and overlap_volume / obj1_volume >= containment_ratio

    def get_spatial_relation(self, obj1: TableInstance, obj2: TableInstance):
        name1 = getattr(obj1, "name", "obj1")
        name2 = getattr(obj2, "name", "obj2")
        if self.is_on(obj1, obj2):
            return f"{name1} on {name2}"
        elif self.is_on_table(obj1):
            return f"{name1} on the table"
        else:
            return None
        
