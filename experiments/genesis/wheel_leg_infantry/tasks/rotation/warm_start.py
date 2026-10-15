"""按观测字段将 locomotion actor 迁移到 rotate；不加载 critic/optimizer。"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch

from ...core.tof import resolve_tof_cfg
from ..jump.observation import LOCOMOTION_POLICY_LAYOUT, layout_slices


@dataclass(frozen=True)
class WarmStartReport:
    source_input_dim: int
    target_input_dim: int
    mapped_input_columns: int
    zero_input_columns: tuple[int, ...]

    def to_dict(self):
        return asdict(self)


def validate_source(source: Mapping, target: Mapping) -> None:
    """维度相同不足以证明兼容，检查动作顺序、缩放和网络配置。"""
    if source['command_cfg'].get('num_commands') != 3:
        raise ValueError('warm-start source must be locomotion with [vx, wz, height] commands')
    a, b = source['env_cfg'], target['env_cfg']
    for key in ('num_actions', 'joint_names', 'wheel_names', 'joint_pos_scale',
                'wheel_vel_scale', 'clip_joint_action', 'clip_wheel_action', 'simulate_action_latency'):
        if key not in a or a[key] != b[key]:
            raise ValueError(f'incompatible action contract: {key}')
    for name in a['joint_names']:
        if a['default_joint_pos'][name] != b['default_joint_pos'][name]:
            raise ValueError(f'incompatible default joint position: {name}')
    for key in ('lin_vel', 'ang_vel', 'joint_pos', 'joint_vel', 'wheel_vel',
                'base_height', 'leg_length', 'leg_angle'):
        if source['obs_cfg']['obs_scales'][key] != target['obs_cfg']['obs_scales'][key]:
            raise ValueError(f'incompatible observation scale: {key}')
    if source['obs_cfg']['obs_scales'].get('lin_acc', 1 / 9.81) != target['obs_cfg']['obs_scales'].get('lin_acc', 1 / 9.81):
        raise ValueError('incompatible observation scale: lin_acc')
    for cfg in (source['train_cfg'], target['train_cfg']):
        if cfg.get('obs_groups', {}).get('actor') != ['policy']:
            raise ValueError("actor must consume only ['policy']")
        if cfg['actor'].get('obs_normalization', False):
            raise ValueError('normalized actors require explicit normalization-statistic migration')
    if source['train_cfg']['actor'] != target['train_cfg']['actor']:
        raise ValueError('actor architecture/distribution configuration must match')


def warm_start_actor(actor, checkpoint_path: str | Path, source_configs: Mapping,
                     target_configs: Mapping, observation_components: Mapping) -> WarmStartReport:
    """第一层按字段搬运；新命令、方向和云台输入列为零，其余层完整复制。"""
    validate_source(source_configs, target_configs)
    source_layout = LOCOMOTION_POLICY_LAYOUT
    source_tof = resolve_tof_cfg(source_configs['obs_cfg'].get('tof'))
    if source_tof['include_in_observation']:
        target_tof = resolve_tof_cfg(target_configs['obs_cfg'].get('tof'))
        if source_tof != target_tof:
            raise ValueError('source ToF observations require identical target ToF configuration')
        source_layout += (('tof_distance', len(source_tof['sensors']) * source_tof['history_frames']),)
    source_slices = layout_slices(source_layout)
    target_layout = tuple((name, value.shape[-1]) for name, value in observation_components.items())
    target_slices = layout_slices(target_layout)
    source_dim = sum(width for _, width in source_layout)
    target_dim = sum(width for _, width in target_layout)
    columns = []
    for name, width in source_layout:
        if name not in target_slices:
            raise ValueError(f'missing target observation: {name}')
        target_slice = target_slices[name]
        target_width = target_slice.stop - target_slice.start
        if target_width != width and not (name == 'commands' and width == 3 and target_width == 4):
            raise ValueError(f'incompatible observation width: {name}: {width} -> {target_width}')
        columns.extend(zip(range(source_slices[name].start, source_slices[name].stop),
                           range(target_slice.start, target_slice.start + width)))
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    source_state = checkpoint['actor_state_dict']
    target_state = actor.state_dict()
    if source_state.keys() != target_state.keys():
        raise ValueError('actor state keys differ')
    # 构造全部新张量，所有检查通过后才修改模型。
    state = {}
    for name, value in target_state.items():
        old = source_state[name].to(device=value.device, dtype=value.dtype)
        if name == 'mlp.0.weight':
            if old.shape != (value.shape[0], source_dim) or value.shape[1] != target_dim:
                raise ValueError(f'actor input dimensions do not match observation layouts: {old.shape}, {value.shape}')
            expanded = torch.zeros_like(value)
            for src, dst in columns:
                expanded[:, dst] = old[:, src]
            state[name] = expanded
        else:
            if old.shape != value.shape:
                raise ValueError(f'incompatible actor tensor: {name}: {old.shape} -> {value.shape}')
            state[name] = old
    actor.load_state_dict(state, strict=True)
    mapped = {dst for _, dst in columns}
    return WarmStartReport(source_dim, target_dim, len(columns),
                           tuple(i for i in range(target_dim) if i not in mapped))
