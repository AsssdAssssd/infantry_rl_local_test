"""Rotate 的多地形混合：按权重把 (task, terrain, difficulty) 变体分配给并行环境。

一个变体就是一块独立地形 patch；环境在 reset 时绑定到其中一个 patch。
当前 task 维只用于日志和后续扩展，所有变体共享同一个 RotateEnv 的奖励/命令。
"""

from __future__ import annotations

import math
from copy import deepcopy

import torch

from ...terrains.config import default_terrain_cfg, resolve_terrain_cfg
from ...terrains.manager import TerrainManager
from ...terrains.registry import TERRAIN_REGISTRY
from ...terrains.scene import TerrainPatches


def _validate_number(name, value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or (positive and value == 0.0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _normalize_variants(spec):
    """把两种写法统一成 ``[{task, preset, difficulty, weight}, ...]``。

    简写：``{"plane": [0.2], "tilted_plane": [0.3, 0.3, 0.2]}``，列表下标是难度。
    完整写法：``[{"task": "rotation", "terrain": "plane", "difficulty": 0, "weight": 1.0}]``。
    """
    if isinstance(spec, dict):
        entries = []
        for preset, weights in spec.items():
            if not isinstance(preset, str):
                raise TypeError("terrain_weights keys must be terrain preset names")
            if isinstance(weights, (int, float)):
                weights = [weights]
            if not isinstance(weights, (list, tuple)):
                raise TypeError(f"terrain_weights[{preset!r}] must be a number or a list of weights")
            for difficulty, weight in enumerate(weights):
                entries.append({
                    "task": "rotation",
                    "preset": preset,
                    "difficulty": difficulty,
                    "weight": _validate_number(f"terrain_weights[{preset!r}][{difficulty}]", weight),
                })
    elif isinstance(spec, (list, tuple)):
        entries = []
        for index, item in enumerate(spec):
            if not isinstance(item, dict):
                raise TypeError(f"variants[{index}] must be a mapping")
            preset = item.get("terrain", item.get("preset"))
            if not isinstance(preset, str):
                raise TypeError(f"variants[{index}] requires a terrain preset name")
            difficulty = item.get("difficulty", 0)
            if isinstance(difficulty, bool) or not isinstance(difficulty, int) or difficulty < 0:
                raise ValueError(f"variants[{index}].difficulty must be a nonnegative integer")
            task = item.get("task", "rotation")
            if not isinstance(task, str) or not task:
                raise ValueError(f"variants[{index}].task must be a nonempty string")
            entries.append({
                "task": task,
                "preset": preset,
                "difficulty": difficulty,
                "weight": _validate_number(f"variants[{index}].weight", item.get("weight", 1.0)),
            })
    else:
        raise TypeError("terrain_weights must be a mapping or a list of variant mappings")

    if not entries:
        raise ValueError("terrain_weights must contain at least one variant")
    keys = [(entry["task"], entry["preset"], entry["difficulty"]) for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("terrain_weights contains duplicate (task, terrain, difficulty) variants")
    return entries


def resolve_terrain_variants(spec, *, base=None):
    """解析所有变体，并为每个变体生成一份完整的 TerrainManager 配置。"""
    base = None if base is None else deepcopy(base)
    if isinstance(base, dict):
        if base.get("preset") == "mix":
            base = base.get("base_terrain")
        if isinstance(base, dict):
            base.pop("variants", None)
            base.pop("base_terrain", None)
    if base is None:
        base = default_terrain_cfg()
    base_cfg = resolve_terrain_cfg(base)

    variants = []
    for entry in _normalize_variants(spec):
        preset = entry["preset"]
        if preset not in TERRAIN_REGISTRY:
            raise ValueError(f"unsupported terrain preset {preset!r}; choose from {sorted(TERRAIN_REGISTRY)}")
        config = {
            "preset": preset,
            "difficulty": entry["difficulty"],
            "tile_size": list(base_cfg["tile_size"]),
            "horizontal_scale": base_cfg["horizontal_scale"],
            "vertical_scale": base_cfg["vertical_scale"],
            "boundary_margin": base_cfg["boundary_margin"],
            "bounded": True,
            "level_catalog": base_cfg.get("level_catalog"),
        }
        variants.append({
            "task": entry["task"],
            "preset": preset,
            "difficulty": entry["difficulty"],
            "weight": entry["weight"],
            "config": resolve_terrain_cfg(config),
        })
    if sum(variant["weight"] for variant in variants) <= 0.0:
        raise ValueError("at least one terrain variant must have positive weight")
    return variants, base_cfg


def allocate_counts(weights, num_envs):
    """最大余数分配；权重为 0 的变体得到 0 个环境。"""
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    weights = [_validate_number("variant weight", weight) for weight in weights]
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("at least one terrain variant must have positive weight")
    quotas = [num_envs * weight / total for weight in weights]
    counts = [math.floor(quota) for quota in quotas]
    remaining = num_envs - sum(counts)
    order = sorted(range(len(counts)), key=lambda i: (quotas[i] - counts[i], weights[i]), reverse=True)
    for index in order[:remaining]:
        counts[index] += 1
    return counts


class RotationTerrain(TerrainPatches):
    """多块地形 patch；每个环境在 reset 时按权重绑定到一个变体。"""

    preset = "mix"
    is_plane = False
    supports_course = False

    # 当前 rotation 机构的 base_height 可行接触区间（含轮半径、hip offset）。
    # 仅对左右腿一长一短（坡沿 y）产生额外 delta。
    WHEEL_HALF_TRACK_M = 0.222
    SUPPORT_HALF_X_M = 0.24
    BASE_HEIGHT_CONTACT_MIN_M = 0.1848
    BASE_HEIGHT_CONTACT_MAX_M = 0.3988
    BASE_HEIGHT_SAFETY_M = 0.01

    def __init__(self, spec, *, base=None, num_envs=1):
        self.variants, self.base_config = resolve_terrain_variants(spec, base=base)
        self.task_names = tuple(dict.fromkeys(variant["task"] for variant in self.variants))
        self._base_height_lower = []
        self._base_height_upper = []
        for variant in self.variants:
            lower, upper = self._variant_base_height_bounds(variant)
            self._base_height_lower.append(lower)
            self._base_height_upper.append(upper)
        self._variant_tasks = torch.tensor(
            [self.task_names.index(variant["task"]) for variant in self.variants],
            dtype=torch.long,
        )
        self.terrains = [TerrainManager(variant["config"]) for variant in self.variants]
        super().__init__(self.terrains)
        # config 保持为合法的单地形配置，env_cfg 存档/续训时可直接作为 patch 模板。
        self.config = {
            **self.base_config,
            "variants": [
                {
                    "task": variant["task"],
                    "terrain": variant["preset"],
                    "difficulty": variant["difficulty"],
                    "weight": variant["weight"],
                }
                for variant in self.variants
            ],
        }
        self.num_envs = num_envs
        self.weights = [variant["weight"] for variant in self.variants]
        self.counts = allocate_counts(self.weights, num_envs)
        self.target_counts = list(self.counts)
        self.fixed_types = None
        self.target_types = None
        self.task_ids = None

    def _variant_base_height_bounds(self, variant):
        if variant["preset"] != "tilted_plane":
            return 0.0, float("inf")
        params = variant["config"].get("subterrain_parameters", {}).get("tilted_plane")
        if not params:
            return 0.0, float("inf")
        angle = math.radians(abs(float(params.get("slope_angle_deg", 0.0))))
        direction = math.radians(float(params.get("direction_deg", 0.0)))
        # 坡沿 y 时左右腿一长一短；坡沿 x 时用前后支撑 footprint。
        delta = abs(math.tan(angle)) * (
            abs(math.cos(direction)) * self.SUPPORT_HALF_X_M
            + abs(math.sin(direction)) * self.WHEEL_HALF_TRACK_M
        )
        lower = self.BASE_HEIGHT_CONTACT_MIN_M + delta + self.BASE_HEIGHT_SAFETY_M
        upper = self.BASE_HEIGHT_CONTACT_MAX_M - delta - self.BASE_HEIGHT_SAFETY_M
        return lower, upper

    def base_height_bounds(self, tile_indices):
        lower = torch.tensor(self._base_height_lower, device=tile_indices.device, dtype=torch.float32)
        upper = torch.tensor(self._base_height_upper, device=tile_indices.device, dtype=torch.float32)
        return lower[tile_indices], upper[tile_indices]

    def _key(self, variant):
        return (variant["task"], variant["preset"], variant["difficulty"])

    def _assignment(self, counts, device):
        indices = torch.arange(len(self.variants), device=device, dtype=torch.long)
        return torch.repeat_interleave(indices, torch.tensor(counts, device=device, dtype=torch.long))

    def bind_entity(self, entity, *, device, dtype):
        super().bind_entity(entity, device=device, dtype=dtype)
        self._variant_tasks = self._variant_tasks.to(device=device)
        self.fixed_types = self._assignment(self.counts, device)
        if self.fixed_types.numel() != self.num_envs:
            raise ValueError("terrain variant counts do not match num_envs")
        self.target_types = self.fixed_types.clone()
        self.task_ids = self._variant_tasks[self.fixed_types]

    def set_weights(self, spec):
        """课程阶段只改权重，不重建几何；新变体必须在初始 terrain_weights 中声明。"""
        entries = _normalize_variants(spec)
        lookup = {self._key(variant): index for index, variant in enumerate(self.variants)}
        weights = [0.0] * len(self.variants)
        for entry in entries:
            key = (entry["task"], entry["preset"], entry["difficulty"])
            if key not in lookup:
                available = sorted(lookup)
                raise ValueError(f"unknown terrain variant {key}; available: {available}")
            weights[lookup[key]] = entry["weight"]
        if sum(weights) <= 0.0:
            raise ValueError("at least one terrain variant must have positive weight")
        for index, weight in enumerate(weights):
            if weight > 0.0 and self._base_height_lower[index] >= self._base_height_upper[index]:
                variant = self.variants[index]
                raise ValueError(
                    f"terrain variant {variant['preset']}/{variant['difficulty']} has no feasible "
                    f"base_height (lower={self._base_height_lower[index]:.3f}, "
                    f"upper={self._base_height_upper[index]:.3f})"
                )
        self.weights = weights
        self.counts = allocate_counts(weights, self.num_envs)
        self.target_counts = list(self.counts)
        if self.target_types is not None:
            self.target_types.copy_(self._assignment(self.counts, self.target_types.device))

    def sample_spawn_tiles(self, env_ids):
        self.fixed_types[env_ids] = self.target_types[env_ids]
        tiles = self.fixed_types[env_ids]
        self.task_ids[env_ids] = self._variant_tasks[tiles]
        return self.centers[tiles], tiles

    def out_of_bounds(self, xy, centers):
        local = xy - centers
        indices = torch.round(centers[:, 1] / self.stride).long()
        result = torch.zeros_like(xy[:, 0], dtype=torch.bool)
        for index, terrain in enumerate(self.terrains):
            margin = terrain.config.get("boundary_margin")
            if margin is None:
                margin = min(0.75, min(terrain.tile_size) / 4.0)
            extent = xy.new_tensor(terrain.tile_size) / 2.0 - margin
            result |= (indices == index) & (local.abs() >= extent).any(dim=-1)
        return result

    def tile_type(self, tile_index):
        variant = self.variants[int(tile_index)]
        return f"{variant['preset']}/{variant['difficulty']}/{variant['task']}"
