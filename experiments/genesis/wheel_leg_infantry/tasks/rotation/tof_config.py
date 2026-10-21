"""Rotation-specific ToF layout.

8个tof，原来的4个+底盘前面没被腿遮挡的地方侧向向下2个+后面斜向下2个，左右对称其余都不对
"""

from __future__ import annotations

from copy import deepcopy

from ...core.tof import resolve_tof_cfg

FORWARD_REFERENCE_DISTANCE_M = 1.0 #tof里面的default、
DOWNWARD_REFERENCE_DISTANCE_M = 0.25

ROTATE_TOF_SENSORS = [
    {
        "name": "tof_left",
        "pos_offset": [0.23, 0.15, -0.16],
        "downward_angle_deg": 45.0,
        "reference_distance_m": FORWARD_REFERENCE_DISTANCE_M,
    },
    {
        "name": "tof_right",
        "pos_offset": [0.23, -0.15, -0.16],
        "downward_angle_deg": 45.0,
        "reference_distance_m": FORWARD_REFERENCE_DISTANCE_M,
    },
    {
        "name": "tof_down_left",
        "pos_offset": [0.10, 0.12, -0.16],
        "downward_angle_deg": 90.0,
        "reference_distance_m": DOWNWARD_REFERENCE_DISTANCE_M,
    },
    {
        "name": "tof_down_right",
        "pos_offset": [0.10, -0.12, -0.16],
        "downward_angle_deg": 90.0,
        "reference_distance_m": DOWNWARD_REFERENCE_DISTANCE_M,
    },
    {
        "name": "tof_side_left",
        "pos_offset": [0.15, 0.12, -0.16],
        "downward_angle_deg": 45.0,
        "yaw_angle": 90.0,
        "reference_distance_m": FORWARD_REFERENCE_DISTANCE_M,
    },
    {
        "name": "tof_side_right",
        "pos_offset": [0.15, -0.12, -0.16],
        "downward_angle_deg": 45.0,
        "yaw_angle": -90.0,
        "reference_distance_m": FORWARD_REFERENCE_DISTANCE_M,
    },
    {
        "name": "tof_back_left",
        "pos_offset": [-0.22, 0.15, -0.14],
        "downward_angle_deg": 45.0,
        "yaw_angle": 180.0,
        "reference_distance_m": FORWARD_REFERENCE_DISTANCE_M,
    },
    {
        "name": "tof_back_right",
        "pos_offset": [-0.22, -0.15, -0.14],
        "downward_angle_deg": 45.0,
        "yaw_angle": 180.0,
        "reference_distance_m": FORWARD_REFERENCE_DISTANCE_M,
    },
]


def apply_rotation_tof(obs_cfg: dict) -> dict:
    """Replace the inherited ToF sensors with the rotation layout and validate."""
    tof = deepcopy(obs_cfg.get("tof") or {})
    if "sensors" not in tof:
        raise KeyError("obs_cfg['tof'] is missing the inherited sensor config")
    tof["sensors"] = deepcopy(ROTATE_TOF_SENSORS)
    obs_cfg["tof"] = resolve_tof_cfg(tof)
    return obs_cfg
