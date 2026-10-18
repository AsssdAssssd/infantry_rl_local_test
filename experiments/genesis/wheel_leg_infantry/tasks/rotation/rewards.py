"""rotate 任务奖励 mixin。

只描述“测量什么”，正负号与权重留在 config 的 reward_scales。
与 LocomotionRewards 通过 MRO 组合（RotateEnv(RotateRewards, LocomotionEnv)）：
- 新名字（如 gimbal_yaw_ang）只在 rotate 里生效；
- 同名方法覆盖 locomotion 的版本（这里把前向速度跟踪换成“云台系向量误差”）。
"""

from __future__ import annotations

import os

import torch

_ABLATION = os.environ.get("ROTATE_ABLATION")

class RotateRewards:
    # ---------- 覆盖：在“云台系”里跟踪整个水平速度向量（大小 + 方向） ----------
    def _base2gimbal_lin_vel(self) -> torch.Tensor:
        """Recent world displacement velocity projected into the sampled command frame."""
        return (
            torch.clamp(self.translation_window_forward_speed, -5.0, 5.0),
            torch.clamp(self.translation_window_lateral_speed, -5.0, 5.0),
        )

    def _rotate_lin_vel_error(self) -> torch.Tensor:
        v_forward ,v_vertical=self._base2gimbal_lin_vel()
        # 目标方向坐标系中期望速度 = [speed_cmd, 0]
        target = torch.cat(
            [self.commands[:, 0:1], torch.zeros_like(self.commands[:, 0:1])], dim=-1
        )
        base2gimbal_vel = torch.stack([v_forward, v_vertical], dim=-1)
        return torch.sum(torch.square(base2gimbal_vel - target), dim=-1)
    
    def _reward_tracking_lin_vel(self):
        if _ABLATION:
            return super()._reward_tracking_lin_vel()
        return self._rotate_lin_vel_error()

    def _reward_gated_tracking_lin_vel(self):
        if _ABLATION:
            return super()._reward_gated_tracking_lin_vel()
        return self.tracking_gate * torch.exp(-self._rotate_lin_vel_error() / self._tracking_sigma())

    
    # ---------- 新增：云台世界朝向保持 ----------
    # def _reward_gimbal_yaw_ang(self):
    #     err = self.gimbal_world_yaw - self.gimbal_yaw_ref
    #     err =  torch.atan2(torch.sin(err), torch.cos(err))
    #     return torch.square(err)

    #位移

    def _reward_gimbal_spin_track_lin_vel_yaw_frame(self):
        real_forward_speed,_=self._base2gimbal_lin_vel()
        target_speed=self.commands[:,0]
        err=torch.square(real_forward_speed-target_speed)
        # 只挂姿态门（机身正不正），不挂高度门：高度没达标也别把"走"的奖励掐死
        return self.tracking_gate * torch.exp(-err/0.2)

    def _reward_gimbal_spin_heading_error(self):
        _, vertical_speed = self._base2gimbal_lin_vel()
        vertical_speed = torch.clamp(vertical_speed, -3.0, 3.0)  # 限幅
        speed_sq = torch.sum(torch.square(self.robot.get_vel()[:, :2]), dim=-1)  # (N,)
        mask = ((speed_sq > 0.001) & (self.commands[:, 0].abs() > 0.0)).to(dtype=vertical_speed.dtype)  # 速度门控：只在动起来时罚侧向
        return self.tracking_gate * torch.square(vertical_speed) * mask

    # ---------- 覆盖：高度误差改 L1（小误差也有梯度，把身高压起来更有力） ----------
    def _reward_base_height(self):
        if _ABLATION:
            return super()._reward_base_height()
        return torch.abs(self.base_height - self.commands[:, 2])

    # ---------- 覆盖：倾斜惩罚按轴分开 —— roll 重罚（治歪站），pitch 轻罚（允许伸腿时后仰） ----------
    def _reward_base_balance(self):
        if _ABLATION:
            return super()._reward_base_balance()
        roll = torch.abs(self.projected_gravity[:, 0])
        pitch = torch.abs(self.projected_gravity[:, 1])
        return roll + 0.1 * pitch

    # ---------- 新增：腿长不对称只在超过死区后才罚（高速自旋+平移允许一定不对称，但不能太离谱） ----------
    def _reward_leg_length_symmetry(self):
        diff = torch.abs(self.leg_length[:, 0] - self.leg_length[:, 1])
        deadband = float(self.reward_cfg.get("leg_length_symmetry_deadband", 0.04))
        return torch.clamp(diff - deadband, min=0.0)

    def _reward_gimbal_spin_track_lin_vel_yaw_lower(self): #惩罚炒熟
        #有一说一如果消融的话，这些个新加的reward不应该都置0吗  TODO
        real_forward_speed,_=self._base2gimbal_lin_vel()
        target_speed=self.commands[:,0]
        return torch.clamp(target_speed-real_forward_speed,min=0.0)

    def _reward_gimbal_spin_cross_track(self):
        excess = torch.clamp(
            torch.abs(self.translation_path_cross_track) - self.translation_path_deadband_m,
            min=0.0, max=self.translation_path_limit_m - self.translation_path_deadband_m,
        )
        delta = self.translation_path_huber_delta_m
        quadratic = 0.5 * excess.square()
        linear = delta * (excess - 0.5 * delta)
        moving = (self.commands[:, 0].abs() > self.reward_cfg.get("standing_lin_vel_threshold", 0.02)).to(excess.dtype)
        return torch.where(excess <= delta, quadratic, linear) * moving

    # ---------- 覆盖：给角速度误差限幅，防数值爆炸把奖励放大到百万级 ----------
    def _reward_tracking_ang_vel(self):
        if _ABLATION:
            return super()._reward_tracking_ang_vel()
        ang_vel = torch.clamp(self.base_ang_vel[:, 2], -20.0, 20.0)
        return torch.square(ang_vel - self.commands[:, 1])



    def _reward_height_gate(self):
        # 直接奖励高度门的开启程度，不依赖姿态门，避免高度还没学会时奖励消失。
        if _ABLATION:
            return super()._reward_height_gate()
        return self.height_gate*self.attitude_gate
