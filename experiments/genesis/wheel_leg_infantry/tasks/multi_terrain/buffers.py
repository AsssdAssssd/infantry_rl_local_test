"""各任务专用 buffer 集中定义；同一任务的创建和回合重置放在同一个类中。

所有张量保留 num_envs 维度，任务 mask 隔离用途；部分 reset 只清空选中的环境。
通用进度、终止、奖励与命令查表仍由 env.py 管理。
"""

from dataclasses import dataclass
import math

import torch


@dataclass
class LocomotionBuffers:
    """平地任务：完整回合的速度、转向、高度跟踪误差。"""

    tracking_errors: torch.Tensor

    @classmethod
    def create(cls, env):
        return cls(tracking_errors=env.base_pos.new_zeros((env.num_envs, 3)))

    def reset(self, env_idx=None):
        ids = slice(None) if env_idx is None else env_idx
        self.tracking_errors[ids] = 0


@dataclass
class StairsDownBuffers:
    """下台阶任务：固定任务 mask、每难度参数表和每回合落地窗口状态。"""

    # 固定任务/难度信息；reset 不清空。
    task_mask: torch.Tensor
    landing_window_steps: torch.Tensor
    landing_ang_vel_weight: torch.Tensor

    # 回合状态；全量及部分 reset 使用同一入口。
    landing_started: torch.Tensor
    landing_steps_left: torch.Tensor
    landing_gate: torch.Tensor

    @classmethod
    def create(cls, env):
        task_mask = torch.tensor([name == "stairs_down" for name in env.tasks], device=env.device)[env.task_ids]
        options = [stage.get("landing_stability", {}) for stage in env.terrain.stages]
        return cls(
            task_mask=task_mask,
            landing_window_steps=torch.tensor(
                [max(1, math.ceil(cfg.get("duration_s", 0.3) / env.dt)) for cfg in options],
                device=env.device, dtype=torch.long,
            ),
            landing_ang_vel_weight=env.base_pos.new_tensor([cfg.get("ang_vel_weight", 0.25) for cfg in options]),
            landing_started=torch.zeros_like(task_mask),
            landing_steps_left=torch.zeros_like(env.episode_length_buf),
            landing_gate=torch.zeros_like(env.base_height),
        )

    def reset(self, env_idx=None):
        ids = slice(None) if env_idx is None else env_idx
        self.landing_started[ids] = False
        self.landing_steps_left[ids] = 0
        self.landing_gate[ids] = 0
