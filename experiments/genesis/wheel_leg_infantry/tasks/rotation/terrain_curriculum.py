"""Rotation 专用地形课程辅助。

放在 ``tasks/rotation`` 内，不修改共享的 ``multi_terrain`` 或 ``terrains``。
目标是把“小陀螺 + 上/下坡”的难度推进写成 rotation 自己的课程阶段。
"""

from __future__ import annotations

from copy import deepcopy

# (start_iteration, difficulty)
# 默认挂在 rotation 已有的阶段边界上：
#   7000  rotation_fast        -> 10°
#   10000 locomotion_straight  -> 20°
#   13500 full_range           -> 25°
DEFAULT_SLOPE_SCHEDULE: tuple[tuple[int, int], ...] = (
    (7000, 0),
    (10000, 2),
    (13500, 4),
)

DEFAULT_TILT_SCHEDULE: tuple[tuple[int, int], ...] = (
    (7000, 0),
    (10000, 1),
    (13500, 2),
)

DEFAULT_TERRAIN_SCHEDULES = {
    "slope_up": DEFAULT_SLOPE_SCHEDULE,
    "slope_down": DEFAULT_SLOPE_SCHEDULE,
    "tilted_plane": DEFAULT_TILT_SCHEDULE,
}


def terrain_stage(name: str, start_iteration: int, preset: str, difficulty: int) -> dict:
    """构造一个只覆盖 ``terrain`` 目标的课程阶段。"""
    return {
        "name": str(name),
        "start_iteration": int(start_iteration),
        "targets": {
            "terrain": {
                "preset": str(preset),
                "difficulty": int(difficulty),
            }
        },
    }


def slope_schedule_stages(
    preset: str,
    schedule: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
    *,
    prefix: str | None = None,
) -> list[dict]:
    """按 ``(start_iteration, difficulty)`` 生成坡地课程阶段。"""
    if preset not in DEFAULT_TERRAIN_SCHEDULES:
        raise ValueError(f"preset must be one of {sorted(DEFAULT_TERRAIN_SCHEDULES)}")
    if schedule is None:
        schedule = DEFAULT_TERRAIN_SCHEDULES[preset]
    prefix = prefix or f"terrain_{preset}"
    stages = []
    previous = -1
    for start, difficulty in schedule:
        start = int(start)
        if start < 0 or start <= previous:
            raise ValueError("terrain schedule start_iteration must be non-negative and strictly increasing")
        previous = start
        stages.append(terrain_stage(f"{prefix}_{start}", start, preset, difficulty))
    return stages


def merge_curriculum_stages(curriculum_cfg: dict, extra_stages: list[dict]) -> dict:
    """把地形阶段合并进现有课程。

    - 已有同 ``start_iteration`` 的阶段：只合并 ``terrain`` 目标，不新增阶段；
    - 没有同起点：插入一个只含 ``terrain`` 目标的新阶段（其他目标按累计覆盖继承）。
    """
    merged = deepcopy(curriculum_cfg)
    stages = list(merged.get("stages", []))
    by_start = {int(stage["start_iteration"]): stage for stage in stages}

    for extra in extra_stages:
        start = int(extra["start_iteration"])
        if start in by_start:
            targets = by_start[start].setdefault("targets", {})
            terrain = targets.setdefault("terrain", {})
            terrain.update(deepcopy(extra["targets"]["terrain"]))
        else:
            stage = deepcopy(extra)
            stages.append(stage)
            by_start[start] = stage

    stages.sort(key=lambda stage: int(stage["start_iteration"]))
    names = [str(stage["name"]) for stage in stages]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate curriculum stage names after terrain merge: {names}")
    merged["stages"] = stages
    return merged


def parse_schedule(text: str) -> tuple[tuple[int, int], ...]:
    """解析 ``"7000:0,10000:2,13500:4"``。"""
    schedule = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            start, difficulty = item.split(":", 1)
            schedule.append((int(start), int(difficulty)))
        except ValueError as exc:
            raise ValueError(f"invalid terrain schedule item {item!r}; expected start:difficulty") from exc
    if not schedule:
        raise ValueError("terrain schedule cannot be empty")
    return tuple(schedule)
