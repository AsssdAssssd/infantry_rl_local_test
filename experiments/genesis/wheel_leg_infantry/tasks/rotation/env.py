"""rotate 任务环境：在 LocomotionEnv 上挂一个可控云台 yaw。

思路（“只增不改”）：
- gimbal_yaw_joint 已经作为第 5 个关节放进 joint_names，所以父类的动作/PD/关节管道
  自动变成 5 关节 + 2 轮 = 7 维，无需改父类；
- 云台的“世界 yaw 保持”用父类的任务扩展点实现：
  * _apply_task_control：把 actor 输出解释成“世界 yaw 目标”，用增量式关节目标伺服；
  * _update_task_state：读关节角、算云台世界 yaw、积分参考角；
  * _get_task_observation_components：把云台朝向/相对角/误差喂给 actor；
  * _reset_task_buffers：重置云台 buffer 与参考角；
- 命令从 [vx, wz, height] 扩展为 [vx, wz, height, gimbal_yaw_rate]。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import genesis as gs
import torch

from ..locomotion.env import LocomotionEnv
from .rewards import RotateRewards



def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class RotateEnv(RotateRewards, LocomotionEnv):
    """locomotion + 云台 yaw 控制。"""

    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        curriculum_cfg=None,
        steps_per_iteration=24,
        show_viewer=False,
    ):
        # 父类 __init__ 末尾会 reset，而 reset 会调用子类 hook，所以先把云台状态准备好。
        self.gimbal_joint_names: tuple[str, ...] = tuple(env_cfg.get("gimbal_joint_names"))
        if len(self.gimbal_joint_names) != 2:
            raise ValueError("rotate currently supports exactly one gimbal yaw joint")
        self.gimbal_joint_ang_scale = float(env_cfg.get("gimbal_joint_ang_scale", math.pi))
        self.gimbal_dof_idx: Optional[torch.Tensor] = None
        self.gimbal_yaw_joint_index = int(env_cfg["joint_names"].index(self.gimbal_joint_names[0]))
        self.gimbal_pitch_joint_index = int(env_cfg["joint_names"].index(self.gimbal_joint_names[1]))


        device = gs.device
        self.gimbal_ang = torch.zeros((num_envs,), dtype=gs.tc_float, device=device)
        self.gimbal_world_yaw = torch.zeros((num_envs,), dtype=gs.tc_float, device=device)
        self.gimbal_yaw_abs_vel=torch.zeros((num_envs,), dtype=gs.tc_float, device=device)
        self.gimbal_yaw_ref = torch.zeros((num_envs,), dtype=gs.tc_float, device=device)

        super().__init__(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            curriculum_cfg=curriculum_cfg,
            steps_per_iteration=steps_per_iteration,
            show_viewer=show_viewer,
        )

        # 机器人已建好，拿云台 DOF 索引。
        self.gimbal_dof_idx = self._joint_dof_indices(self.gimbal_joint_names) #2

    # ============ 命令管道：从 3 维扩到 4 维（追加云台速率） ============

    def _build_commands_scale(self):
        return torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
                self.obs_scales["base_height"],
                self.obs_scales["gimbal_yaw_vel"],
            ],
            dtype=gs.tc_float,
            device=self.device,
        )

    def _build_command_limits(self):
        lower, upper = super()._build_command_limits()
        gimbal_range = self.command_cfg["gimbal_yaw_vel_range"]
        extra_lower = torch.tensor([gimbal_range[0]], dtype=gs.tc_float, device=self.device)
        extra_upper = torch.tensor([gimbal_range[1]], dtype=gs.tc_float, device=self.device)
        return torch.cat([lower, extra_lower]), torch.cat([upper, extra_upper])

    def _apply_command_ranges(self, values): #course
        values = dict(values)
        gimbal = values.pop("gimbal_yaw_vel_range", None)
        if gimbal is not None:
            if len(gimbal) != 2 or gimbal[0] > gimbal[1]:
                raise ValueError("gimbal_yaw_vel_range must be [lower, upper]")
            self.command_cfg["gimbal_yaw_vel_range"] = list(gimbal)
        # 父类会用（我们重写的）_build_command_limits 重建采样上下限。
        super()._apply_command_ranges(values)

    # ============ 云台控制与状态（父类任务扩展点） ============

    def _apply_task_control(self):
        """把 actor 的云台维解释为“世界 yaw 目标”，用增量式关节目标伺服。"""
        if self.gimbal_dof_idx is None:
            return
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        world_target = exec_actions[:, self.gimbal_yaw_joint_index] * self.gimbal_joint_ang_scale
        # 世界朝向最短路径误差 → 加到当前（未 wrap 的）关节角上，避免 ±π 处 2π 跳变
        err = _wrap(world_target - self.gimbal_world_yaw)
        joint_target = (self.gimbal_ang + err).unsqueeze(-1)
        pitch_joint_keep = torch.full((self.num_envs,1),math.radians(-20.0),dtype=gs.tc_float,device=gs.device)

        self.robot.control_dofs_position(torch.cat([joint_target,pitch_joint_keep],dim=-1), self.gimbal_dof_idx)

    def _update_task_state(self):
        """更新云台关节角、世界朝向与积分参考角；在奖励计算前调用。"""
        if self.gimbal_dof_idx is None:
            return
        self.gimbal_ang.copy_(self.joint_pos[:, self.gimbal_yaw_joint_index])
        base_yaw = torch.deg2rad(self.base_euler[:, 2])
        self.gimbal_world_yaw.copy_(_wrap(base_yaw + self.gimbal_ang))
        # 参考角按“云台世界 yaw 速率”命令积分；command=0 时保持不变。
        self.gimbal_yaw_ref.copy_(_wrap(self.gimbal_yaw_ref + self.commands[:, 3] * self.dt))
        self.gimbal_yaw_abs_vel.copy_(
            self.robot.get_ang()[:, 2] + self.joint_vel[:, self.gimbal_yaw_joint_index]
        )

    def _reset_task_buffers(self, env_idx):
        selected = slice(None) if env_idx is None else env_idx
        # 初始 base 姿态与云台默认角都为 0（单位四元数），故参考角从 0 起。
        self.gimbal_ang[selected] = 0.0
        self.gimbal_world_yaw[selected] = 0.0
        self.gimbal_yaw_ref[selected] = 0.0
        self.gimbal_yaw_abs_vel[selected] = 0.0
        

    # ============ 观测（父类任务扩展点） ============

    def _get_task_observation_components(self):
        base_yaw = torch.deg2rad(self.base_euler[:, 2])
        rel = _wrap(self.gimbal_world_yaw - base_yaw)  # base → gimbal 相对角
        err = _wrap(self.gimbal_yaw_ref - self.gimbal_world_yaw)  # 离目标还差多少
        return {
            "gimbal_yaw_joint": (self.gimbal_world_yaw * self.obs_scales["gimbal_yaw_angle"]).unsqueeze(-1),  # 1
            "gimbal_yaw_2_base_angle": torch.stack([torch.sin(rel), torch.cos(rel)], dim=-1),  # 2
            "gimbal_yaw_error": torch.stack([torch.sin(err), torch.cos(err)], dim=-1),  # 2
            "gimbal_yaw_abs_vel": (
                self.gimbal_yaw_abs_vel * self.obs_scales["gimbal_yaw_vel"]
            ).unsqueeze(-1),  # 1
        }

    def _get_task_privileged_observation_components(self):
        return {
            "privileged_gimbal_world_yaw": self.gimbal_world_yaw.unsqueeze(-1),
            "privileged_gimbal_yaw_ref": self.gimbal_yaw_ref.unsqueeze(-1),
        }

    # ============ 诊断 ============

    def get_rotate_diagnostics(self, env_idx=0):
        return {
            "gimbal_joint": self.gimbal_ang[env_idx].detach(),
            "gimbal_world_yaw": self.gimbal_world_yaw[env_idx].detach(),
            "gimbal_yaw_ref": self.gimbal_yaw_ref[env_idx].detach(),
            "gimbal_yaw_error": _wrap(self.gimbal_yaw_ref[env_idx] - self.gimbal_world_yaw[env_idx]).detach(),
        }
