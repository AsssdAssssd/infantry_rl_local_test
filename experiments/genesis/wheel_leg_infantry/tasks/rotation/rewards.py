"""rotate 任务奖励 mixin。

只描述“测量什么”，正负号与权重留在 config 的 reward_scales。
与 LocomotionRewards 通过 MRO 组合（RotateEnv(RotateRewards, LocomotionEnv)）：
- 新名字（如 gimbal_yaw_ang）只在 rotate 里生效；
- 同名方法覆盖 locomotion 的版本（这里把前向速度跟踪换成“云台系向量误差”）。
"""

from __future__ import annotations

import torch

class RotateRewards:
    # ---------- 覆盖：在“云台系”里跟踪整个水平速度向量（大小 + 方向） ----------
    def _rotate_lin_vel_error(self) -> torch.Tensor:
        world_vel_xy = self.robot.get_vel()[:, :2]  # (N,2) 世界系水平速度
        psi = self.gimbal_world_yaw  # 云台世界朝向 (N,)
        target = self.commands[:, 0:1] * torch.stack([torch.cos(psi), torch.sin(psi)], dim=-1)
        return torch.sum(torch.square(world_vel_xy - target), dim=-1)

    def _reward_tracking_lin_vel(self):
        return self._rotate_lin_vel_error()

    def _reward_gated_tracking_lin_vel(self):
        return self.tracking_gate * torch.exp(-self._rotate_lin_vel_error() / self._tracking_sigma())

    # ---------- 新增：云台世界朝向保持 ----------
    def _reward_gimbal_yaw_ang(self):
        err = self.gimbal_world_yaw - self.gimbal_yaw_ref
        err =  torch.atan2(torch.sin(err), torch.cos(err))
        return torch.square(err)
