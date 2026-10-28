"""六边形堡垒：固定平台宽度与高度，按坡角改变外圈尺寸。"""

from __future__ import annotations

import math

import numpy as np

from .base import TerrainGeometry

PRESET = "fort"
PARAMETER_KEY = "fort"
PARAMETERS = dict(platform_width=1.131, height=0.15, slope_angle_deg=20.0, base_thickness=0.10)

LEVELS = {
    0: dict(slope_angle_deg=10.0),
    1: dict(slope_angle_deg=17.1),
    2: dict(slope_angle_deg=20.0),
    3: dict(slope_angle_deg=23.0),
}


class Fort(TerrainGeometry):
    """Pointed hexagonal plateau with six planar ramp faces."""

    def __init__(self, parameters, tile_size, horizontal_scale=0.1, *, boundary_margin=2.0):#给至少1m的平台余量
        super().__init__(parameters, tile_size, horizontal_scale, boundary_margin=boundary_margin)
        for name in ("platform_width", "height", "slope_angle_deg", "base_thickness"):
            value = float(parameters[name])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"fort.{name} must be positive and finite")
            setattr(self, name, value)

        self.outer_width = self.platform_width + 2.0 * self.height / math.tan(math.radians(self.slope_angle_deg))
        self.inner_ratio = self.platform_width / self.outer_width
        margin = boundary_margin or 0.0
        if (self.outer_width / 2.0 >= self.tile_size[0] / 2.0 - margin
                or self.outer_width / math.sqrt(3.0) >= self.tile_size[1] / 2.0 - margin):
            raise ValueError("fort must fit inside tile_size and boundary_margin")

        self.start_x = -self.outer_width / 2.0
        self.traversal_end = self.outer_width / 2.0

    def height_at(self, local_xy):
        x = local_xy[:, 0].abs()
        y = local_xy[:, 1].abs()
        radius = (2.0 * x / self.outer_width).maximum(
            (x + math.sqrt(3.0) * y) / self.outer_width
        )
        return ((1.0 - radius) / (1.0 - self.inner_ratio)).clamp(0.0, 1.0) * self.height

    def add_to_scene(self, scene):
        import genesis as gs
        import trimesh

        lx, ly = self.tile_size
        ground = scene.add_entity(gs.morphs.Box(
            lower=(-lx / 2.0, -ly / 2.0, -self.base_thickness),
            upper=(lx / 2.0, ly / 2.0, 0.0),
            fixed=True,
            batch_fixed_verts=False,
        ))
        angles = np.arange(6) * math.pi / 3.0 + math.pi / 2.0
        outer_radius = self.outer_width / math.sqrt(3.0)
        inner_radius = self.platform_width / math.sqrt(3.0)
        vertices = np.asarray(
            [(radius * math.cos(angle), radius * math.sin(angle), z)
             for radius, z in ((outer_radius, 0.0), (inner_radius, self.height))
             for angle in angles],
            dtype=float,
        )
        mesh = trimesh.convex.convex_hull(vertices)
        fort = scene.add_entity(gs.morphs.MeshSet(
            files=(mesh,), fixed=True, convexify=True, decimate=False,
            batch_fixed_verts=False,
        ), surface=gs.surfaces.Default(smooth=False))
        return ground, fort


TERRAIN_CLASS = Fort
