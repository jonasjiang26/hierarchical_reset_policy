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

    def is_on_table(self, instance: TableInstance, tabletop_high=0.01):
        xyz_min, _ = self._get_world_bbox(instance)
        print(f"Object {instance.name} has min z: {xyz_min[2]:.3f}")
        return xyz_min[2] < tabletop_high

    def is_on(self, obj1: TableInstance, obj2: TableInstance, height_threshold=0.04):
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
    
    def is_in_initial_position(self, instance: TableInstance, initial_position: np.ndarray, position_threshold=0.05):
        bbox_min, bbox_max = self._get_world_bbox(instance)
        center = (bbox_min + bbox_max) / 2.0
        distance = np.linalg.norm(center - initial_position)
        print(f"Object {instance.name} is {distance:.3f}m from initial position")
        return distance < position_threshold

    def get_spatial_relation(self, obj1: TableInstance, obj2: TableInstance):
        name1 = self.spatial_relation_instance_name(obj1)
        name2 = self.spatial_relation_instance_name(obj2)
        if self.is_on(obj1, obj2):
            return f"{name1} on {name2}"
        elif self.is_on_table(obj1):
            return f"{name1} on table"
        else:
            return None
        
    def spatial_relation_subject_name(self, instance: TableInstance):
        return self.spatial_relation_instance_name(instance)

    def spatial_relation_instance_name(self, instance: TableInstance):
        name = getattr(instance, "name", "")
        if "drawer" not in self._instance_key(name):
            return name

        drawer_state = "opened" if self.is_drawer_open(instance) else "closed"
        return self._with_drawer_state_prefix(name, drawer_state)

    def _with_drawer_state_prefix(self, name, drawer_state):
        stripped_name = name.strip().rstrip(".")
        if not stripped_name:
            return name

        words = stripped_name.split(maxsplit=1)
        first_word = words[0].lower().rstrip(".")
        if first_word not in {"opened", "closed"}:
            return f"{drawer_state} {stripped_name}"

        if len(words) == 1:
            return drawer_state
        return f"{drawer_state} {words[1].rstrip('.')}"

    def _instance_key(self, name):
        return name.strip().lower().rstrip(".")

    def is_drawer_open(self, drawer: TableInstance, open_threshold=-0.1336):
        bbox_min, _ = self._get_world_bbox(drawer)
        return bbox_min[1] < open_threshold
        
