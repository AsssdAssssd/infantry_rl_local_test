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
    def _base2gimbal_lin_vel(self) -> torch.Tensor:
        world_vel_xy = self.robot.get_vel()[:, :2]  # (N,2) 世界系水平速度
        psi = self.gimbal_yaw_ref # 目标云台世界朝向 (N,)（其实从设计 逻辑上我是希望追中实际的gimbal朝向的，但是好想追不上（20、0.6），，，要不要改成ref呢）
        v_forward=torch.cos(psi)*world_vel_xy[:,0]+torch.sin(psi)*world_vel_xy[:,1]
        v_vertical=torch.cos(psi)*world_vel_xy[:,1]-torch.sin(psi)*world_vel_xy[:,0]

        
        return v_forward ,v_vertical

    def _rotate_lin_vel_error(self) -> torch.Tensor:
        v_forward ,v_vertical=self._base2gimbal_lin_vel()
        # 云台系里期望速度向量 = [vx_cmd, 0]（沿云台朝向、无侧向）
        target = torch.cat(
            [self.commands[:, 0:1], torch.zeros_like(self.commands[:, 0:1])], dim=-1
        )
        base2gimbal_vel = torch.stack([v_forward, v_vertical], dim=-1)
        return torch.sum(torch.square(base2gimbal_vel - target), dim=-1)
    
    def _reward_tracking_lin_vel(self):
        return self._rotate_lin_vel_error()

    def _reward_gated_tracking_lin_vel(self):
        return self.tracking_gate * torch.exp(-self._rotate_lin_vel_error() / self._tracking_sigma())

    
    # ---------- 新增：云台世界朝向保持 ----------
    # def _reward_gimbal_yaw_ang(self):
    #     err = self.gimbal_world_yaw - self.gimbal_yaw_ref
    #     err =  torch.atan2(torch.sin(err), torch.cos(err))
    #     return torch.square(err)

    def _reward_gimbal_spin_track_lin_vel_yaw_frame(self):
        real_forward_speed,_=self._base2gimbal_lin_vel()
        target_speed=self.commands[:,0]
        return torch.square((real_forward_speed-target_speed))

    def _reward_gimbal_spin_heading_error(self):
        _, vertical_speed = self._base2gimbal_lin_vel()
        speed_sq = torch.sum(torch.square(self.robot.get_vel()[:, :2]), dim=-1)  # (N,)
        mask = ((speed_sq > 0.1) & (self.commands[:, 0].abs() > 0.0)).to(dtype=vertical_speed.dtype)  # 速度门控：只在动起来时罚侧向
        return torch.square(vertical_speed) * mask

    def _reward_gimbal_spin_track_lin_vel_yaw_lower(self): #惩罚炒熟
        real_forward_speed,_=self._base2gimbal_lin_vel()
        target_speed=self.commands[:,0]
        return torch.clamp(real_forward_speed-target_speed,min=0.0)


