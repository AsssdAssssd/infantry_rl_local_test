"""均匀倾斜平面：整块地面按固定倾角倾斜，用于旋转/平移验证。"""

from __future__ import annotations

import math

import numpy as np

from .base import TerrainGeometry

PRESET = "tilted_plane"
PARAMETER_KEY = "tilted_plane"
PARAMETERS = dict(slope_angle_deg=5.0, direction_deg=0.0, base_thickness=0.10)

# difficulty -> tilt magnitude (degrees)
LEVELS = {
    0: dict(slope_angle_deg=3.0),
    1: dict(slope_angle_deg=6.0),
    2: dict(slope_angle_deg=10.0),
    3: dict(slope_angle_deg=14.0),
    4: dict(slope_angle_deg=17.0),
    5: dict(slope_angle_deg=20.0),
}


class TiltedPlane(TerrainGeometry):
    """Constant-tilt slab: z = tan(angle) * (x*cos(dir) + y*sin(dir))."""

    # 出生点只抬 z，rpy 保持水平；base 仍由策略学着用一长一短腿贴坡。
    SPAWN_CLEARANCE_M = 0.03
    SPAWN_FOOTPRINT_HALF_X = 0.30
    SPAWN_FOOTPRINT_HALF_Y = 0.25
    MIN_BASE_THICKNESS_M = 0.50

    def __init__(self, parameters, tile_size, horizontal_scale=0.1, *, boundary_margin=None):
        super().__init__(parameters, tile_size, horizontal_scale, boundary_margin=boundary_margin)
        self.slope_angle_deg = float(parameters["slope_angle_deg"])
        self.direction_deg = float(parameters.get("direction_deg", 0.0))
        # 至少 0.5m 厚，避免高坡/落地冲击把轮子打穿薄板。
        self.base_thickness = max(float(parameters["base_thickness"]), self.MIN_BASE_THICKNESS_M)
        self.spawn_clearance_m = float(parameters.get("spawn_clearance_m", self.SPAWN_CLEARANCE_M))
        self.spawn_half_x = float(parameters.get("spawn_footprint_half_x", self.SPAWN_FOOTPRINT_HALF_X))
        self.spawn_half_y = float(parameters.get("spawn_footprint_half_y", self.SPAWN_FOOTPRINT_HALF_Y))
        if self.spawn_clearance_m < 0.0 or self.spawn_half_x < 0.0 or self.spawn_half_y < 0.0:
            raise ValueError("tilted_plane spawn clearance/footprint must be nonnegative")
        if not 0.0 < self.slope_angle_deg < 45.0:
            raise ValueError("tilted_plane slope_angle_deg must be in (0, 45)")
        if self.base_thickness <= 0.0:
            raise ValueError("tilted_plane base_thickness must be positive")
        angle = math.radians(self.slope_angle_deg)
        direction = math.radians(self.direction_deg)
        self.gradient_x = math.tan(angle) * math.cos(direction)
        self.gradient_y = math.tan(angle) * math.sin(direction)
        # aliases used by terrain_preview camera placement
        self.height = 0.5 * (
            abs(self.gradient_x) * self.tile_size[0] + abs(self.gradient_y) * self.tile_size[1]
        )
        self.start_x = 0.0
        self.traversal_end = 1.0

    def height_at(self, local_xy):
        return local_xy[:, 0] * self.gradient_x + local_xy[:, 1] * self.gradient_y

    def adjust_spawn(self, positions, rpy, *, mask=None, offset=None):
        """只抬高出生 z；rpy 不动，base 保持水平。

        坡面在机器人支撑 footprint 内的高度差为
        ``|gx|*half_x + |gy|*half_y``。把出生 z 抬高这个量再加 clearance，
        可避免上坡侧的轮/腿一开始就插进地面。
        """
        lift = (
            abs(self.gradient_x) * self.spawn_half_x
            + abs(self.gradient_y) * self.spawn_half_y
            + self.spawn_clearance_m
        )
        if mask is None:
            positions[:, 2] += lift
        else:
            positions[mask, 2] += lift

    def add_to_scene(self, scene):
        import genesis as gs
        import trimesh

        lx, ly = self.tile_size
        corners = [(sx * lx / 2, sy * ly / 2) for sx in (-1, 1) for sy in (-1, 1)]
        top = [(x, y, x * self.gradient_x + y * self.gradient_y) for x, y in corners]
        bottom = [(x, y, z - self.base_thickness) for x, y, z in top]
        mesh = trimesh.convex.convex_hull(np.asarray(top + bottom, dtype=float))
        morph = gs.morphs.MeshSet(
            files=(mesh,),
            fixed=True,
            convexify=True,
            decimate=False,
            batch_fixed_verts=False,
        )
        return scene.add_entity(morph)


TERRAIN_CLASS = TiltedPlane
