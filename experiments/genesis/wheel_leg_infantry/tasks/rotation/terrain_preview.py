"""Headless terrain preview: render a terrain preset to PNG (no policy/robot).

Kept inside ``tasks/rotation`` for now; it only uses the shared terrain API.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import genesis as gs
import imageio.v2 as imageio

from ...terrains import TERRAIN_PRESETS, default_terrain_cfg, resolve_terrain_cfg
from ...terrains.manager import TerrainManager
from ...terrains.registry import level_catalog


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", choices=TERRAIN_PRESETS, default="slope_up")
    parser.add_argument("--difficulty", type=int, default=None)
    parser.add_argument("--tile-size", type=float, nargs=2, metavar=("LENGTH", "WIDTH"), default=None)
    parser.add_argument("--output-dir", type=str, default="eval_renders/terrain_preview")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _terrain_cfg(args):
    cfg = default_terrain_cfg(args.terrain)
    cfg["level_catalog"] = level_catalog()
    if args.difficulty is not None:
        cfg["difficulty"] = args.difficulty
    if args.tile_size is not None:
        cfg["tile_size"] = list(args.tile_size)
    return resolve_terrain_cfg(cfg)


def _views(geometry):
    start_x = float(getattr(geometry, "start_x", 0.0))
    end_x = float(getattr(geometry, "traversal_end", 1.0))
    height = float(getattr(geometry, "height", 0.2))
    mid_x = 0.5 * (start_x + end_x)
    return {
        "persp": ((6.0, -8.0, 4.0), (0.8, 0.0, 0.1)),
        "close": ((3.5, -3.5, 1.6), (mid_x, 0.0, height * 0.5)),
        "side": ((mid_x, -9.0, 2.2), (mid_x, 0.0, height * 0.5)),
        "top": ((mid_x, 0.0, 10.0), (mid_x, 0.0, 0.0)),
    }


def main():
    args = _parse_args()
    cfg = _terrain_cfg(args)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)

    terrain = TerrainManager(cfg)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=20),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False, tolerance=1e-5, max_collision_pairs=20),
        show_viewer=False,
    )
    ground = terrain.add_to_scene(scene)
    terrain.bind_entity(ground, device=gs.device, dtype=gs.tc_float)

    views = _views(terrain.geometry)
    cam = scene.add_camera(
        res=(args.width, args.height),
        pos=views["persp"][0],
        lookat=views["persp"][1],
        up=(0.0, 0.0, 1.0),
        fov=45,
        debug=True,
    )
    scene.build()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.terrain if args.difficulty is None else f"{args.terrain}_d{args.difficulty}"
    terrain_angle = getattr(terrain.geometry, "slope_angle_deg", None)
    print(f"[terrain_preview] preset={args.terrain} difficulty={args.difficulty} angle={terrain_angle}")
    for name, (pos, lookat) in views.items():
        cam.set_pose(pos=pos, lookat=lookat)
        rgb, *_ = cam.render()
        image = rgb[0] if getattr(rgb, "ndim", 0) == 4 else rgb
        path = out_dir / f"{tag}_{name}.png"
        imageio.imwrite(path, image)
        print(f"[terrain_preview] {path} shape={image.shape}")


if __name__ == "__main__":
    main()
