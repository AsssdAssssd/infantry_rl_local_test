"""Rotate：四腿两轮策略 + 独立 yaw/pitch 云台伺服。

新命令为 [speed, base_wz, height, gimbal_yaw_rate, translation_direction]。
方向相对云台参考 yaw，原始弧度只用于采样，actor 读取 sin/cos。
旧四维命令配置保留向前平移语义，便于评估历史 run。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import genesis as gs
import torch

from ..locomotion.env import LocomotionEnv
from .rewards import RotateRewards
from .terrain import RotationTerrain



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

        device = gs.device
        self.gimbal_yaw_ang = torch.zeros((num_envs,), dtype=gs.tc_float, device=device)
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
        kp = torch.full((2,), 15.0, dtype=gs.tc_float, device=self.device)  #同其余joint，后写到config里面 TODO
        kd = torch.full((2,), 0.6, dtype=gs.tc_float, device=self.device)
        self.robot.set_dofs_kp(kp.tolist(), self.gimbal_dof_idx)
        self.robot.set_dofs_kv(kd.tolist(), self.gimbal_dof_idx)
        self.robot.set_dofs_force_range([-8.0, -8.0], [8.0, 8.0], self.gimbal_dof_idx)

    # ============ 地形：rotation 自己的多地形混合 ============

    def _make_terrain(self, env_cfg, num_envs):
        weights = env_cfg.get("terrain_weights")
        variants = env_cfg.get("terrain_variants")
        if weights and variants:
            raise ValueError("configure either terrain_weights or terrain_variants, not both")
        if weights or variants:
            return RotationTerrain(weights or variants, base=env_cfg.get("terrain"), num_envs=num_envs)
        return super()._make_terrain(env_cfg, num_envs)

    def _register_task_curriculum(self):
        super()._register_task_curriculum()
        self.curriculum.register_target("terrain_weights", self._apply_terrain_weights)

    def _apply_terrain_weights(self, values):
        """混合地形只改权重；各环境在下一次 reset 时切换到新分配。"""
        if not isinstance(self.terrain, RotationTerrain):
            raise ValueError("terrain_weights requires terrain_weights or terrain_variants in env_cfg")
        self.terrain.set_weights(values)

    def _apply_terrain_curriculum(self, values):
        if isinstance(self.terrain, RotationTerrain):
            raise ValueError("terrain curriculum cannot be combined with terrain_weights")
        super()._apply_terrain_curriculum(values)

    def _activate_pending_terrain(self):
        if isinstance(self.terrain, RotationTerrain):
            return False
        return super()._activate_pending_terrain()

    # ============ 命令管道：五维命令，方向以 sin/cos 单独提供给 actor ============

    def _has_direction_command(self):
        # 旧 run 的四维命令仍可用于评估；新训练使用五维命令。
        return self.command_cfg["num_commands"] == 5

    def _build_commands_scale(self):
        scales = [
            self.obs_scales["lin_vel"],
            self.obs_scales["ang_vel"],
            self.obs_scales["base_height"],
            self.obs_scales["gimbal_yaw_vel"],
        ]
        if self._has_direction_command():
            scales.append(self.obs_scales.get("translation_direction", 1.0))
        return torch.tensor(
            scales,
            dtype=gs.tc_float,
            device=self.device,
        )

    def _build_command_limits(self):
        lower, upper = super()._build_command_limits()
        if self._has_direction_command():
            ranges = [self.command_cfg["gimbal_yaw_vel_range"],self.command_cfg["translation_direction_range"]]
        else:
            ranges = [self.command_cfg["gimbal_yaw_vel_range"]]
        extra_lower = torch.tensor([r[0] for r in ranges], dtype=gs.tc_float, device=self.device)
        extra_upper = torch.tensor([r[1] for r in ranges], dtype=gs.tc_float, device=self.device)
        return torch.cat([lower, extra_lower]), torch.cat([upper, extra_upper])

    def _apply_command_ranges(self, values):
        values = dict(values)
        direction = values.pop("translation_direction_range", None)
        if direction is not None:
            if (len(direction) != 2 or not all(math.isfinite(float(x)) for x in direction)
                or not -math.pi <= direction[0] <= direction[1] <= math.pi):
                raise ValueError("translation_direction_range must lie within [-pi, pi]")
            self.command_cfg["translation_direction_range"] = list(direction)
        gimbal = values.pop("gimbal_yaw_vel_range", None)
        if gimbal is not None:
            if len(gimbal) != 2 or gimbal[0] > gimbal[1]:
                raise ValueError("gimbal_yaw_vel_range must be [lower, upper]")
            self.command_cfg["gimbal_yaw_vel_range"] = list(gimbal)
        super()._apply_command_ranges(values)

    def _get_command_observation_components(self):
        # 不把有 ±pi 接缝的原始角度喂给网络，也不挪动原有四条命令。
        return {"commands": self.commands[:, :4] * self.commands_scale[:4]}

    def get_translation_direction(self):
        """相对云台参考 yaw：0 前、+pi/2 左、-pi/2 右、pi 后。"""
        if self._has_direction_command():
            return self.commands[:, 4]
        return torch.zeros_like(self.commands[:, 0])

    def _initialize_task_buffers(self):
        self.translation_window_s = float(self.env_cfg.get("translation_window_s", 0.5))
        if not math.isfinite(self.translation_window_s) or self.translation_window_s <= 0.0:
            raise ValueError("translation_window_s must be positive and finite")
        self.translation_window_steps = max(1, round(self.translation_window_s / self.dt))
        self.translation_history_length = self.translation_window_steps + 1
        self.translation_resample_steps = max(1, round(self.resample_step))
        self.translation_position_history = torch.zeros(
            (self.num_envs, self.translation_history_length, 2), dtype=gs.tc_float, device=self.device
        )
        self.translation_window_age = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.translation_path_origin = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=self.device)
        self.translation_path_yaw = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device)
        self.translation_path_progress = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device)
        self.translation_path_cross_track = torch.zeros_like(self.translation_path_progress)
        self.translation_window_forward_speed = torch.zeros_like(self.translation_path_progress)
        self.translation_window_lateral_speed = torch.zeros_like(self.translation_path_progress)
        self.translation_distance_scale = float(self.obs_scales.get("translation_distance", 1.0))
        self.translation_path_deadband_m = float(self.reward_cfg.get("translation_path_deadband_m", 0.3))
        self.translation_path_limit_m = float(self.reward_cfg.get("translation_path_limit_m", 1.0))
        self.translation_path_huber_delta_m = float(self.reward_cfg.get("translation_path_huber_delta_m", 0.7))
        if self.translation_distance_scale <= 0.0:
            raise ValueError("translation_distance observation scale must be positive")
        if self.translation_path_deadband_m < 0.0 or self.translation_path_huber_delta_m <= 0.0:
            raise ValueError("translation path deadband must be nonnegative and Huber delta positive")
        if self.translation_path_limit_m <= self.translation_path_deadband_m:
            raise ValueError("translation path limit must exceed the deadband")

    def _resample_commands(self, envs_idx):
        super()._resample_commands(envs_idx)
        if envs_idx is not None and not torch.any(envs_idx):
            return
        if isinstance(self.terrain, RotationTerrain):
            self._resample_mixed_base_height(envs_idx)
        selected = slice(None) if envs_idx is None else envs_idx
        position = self.base_pos[selected, :2]
        self.translation_path_origin[selected] = position
        self.translation_path_yaw[selected] = self.gimbal_world_yaw[selected] + self.get_translation_direction()[selected]
        self.translation_window_age[selected] = 0
        self.translation_position_history[selected] = position[:, None, :].expand(
            -1, self.translation_history_length, -1
        )
        self.translation_path_progress[selected] = 0.0
        self.translation_path_cross_track[selected] = 0.0
        self.translation_window_forward_speed[selected] = 0.0
        self.translation_window_lateral_speed[selected] = 0.0

    def _resample_mixed_base_height(self, envs_idx):
        """按 env 当前地形重采样 base_height，避免生成腿长做不到的指令。

        ang/lin 仍用全局 ranges；这里只处理第 2 维 base_height。
        优先使用「全局范围 ∩ 地形可行范围」；若课程固定 0.22 而坡面需要
        0.25，交集为空，则退回该地形自身的可行范围。
        """
        selected = slice(None) if envs_idx is None else envs_idx
        tiles = self.terrain_tile_index[selected]
        variant_low, variant_high = self.terrain.base_height_bounds(tiles)
        global_low = self.commands_limit[0][2]
        global_high = self.commands_limit[1][2]
        low = torch.maximum(variant_low, global_low)
        high = torch.minimum(variant_high, global_high)
        fallback = low > high
        if fallback.any():
            low = torch.where(fallback, variant_low, low)
            high = torch.where(fallback, variant_high, high)
        if (low > high).any():
            raise ValueError("no feasible base_height command for some terrain variants")
        self.commands[selected, 2] = low + (high - low) * torch.rand_like(low)

    def _update_translation_metrics(self):
        """Update rolling displacement velocity and command-segment path error."""
        age = self.translation_window_age + 1
        self.translation_window_age.copy_(age)
        env_ids = torch.arange(self.num_envs, device=self.device)
        current_index = age.remainder(self.translation_history_length)
        position = self.base_pos[:, :2]
        self.translation_position_history[env_ids, current_index] = position

        window_steps = age.clamp(max=self.translation_window_steps)
        old_index = (age - window_steps).remainder(self.translation_history_length)
        old_position = self.translation_position_history[env_ids, old_index]
        elapsed = window_steps.to(gs.tc_float) * self.dt
        average_velocity = (position - old_position) / elapsed.unsqueeze(-1)

        forward = torch.stack(
            [torch.cos(self.translation_path_yaw), torch.sin(self.translation_path_yaw)], dim=-1
        )
        lateral = torch.stack([-forward[:, 1], forward[:, 0]], dim=-1)
        self.translation_window_forward_speed.copy_(torch.sum(average_velocity * forward, dim=-1))
        self.translation_window_lateral_speed.copy_(torch.sum(average_velocity * lateral, dim=-1))

        displacement = position - self.translation_path_origin
        self.translation_path_progress.copy_(torch.sum(displacement * forward, dim=-1))
        self.translation_path_cross_track.copy_(torch.sum(displacement * lateral, dim=-1))

    # ============ 云台控制与状态（父类任务扩展点） ============

    def _apply_task_control(self):
        """把 actor 的云台维解释为“世界 yaw 目标”，用增量式关节目标伺服。"""
        if self.gimbal_dof_idx is None:
            return

        world_target = self.gimbal_yaw_ref #更改，不依靠策略的输出
        # 世界朝向最短路径误差 → 加到当前（未 wrap 的）关节角上，避免 ±π 处 2π 跳变
        err = _wrap(world_target - self.gimbal_world_yaw)
        joint_target = (self.gimbal_yaw_ang + err).unsqueeze(-1)
        pitch_joint_keep = torch.full((self.num_envs,1),math.radians(-5.0),dtype=gs.tc_float,device=gs.device)

        self.robot.control_dofs_position(torch.cat([joint_target,pitch_joint_keep],dim=-1), self.gimbal_dof_idx)

    def _update_task_state(self):
        """更新云台关节角、世界朝向与积分参考角；在奖励计算前调用。"""
        if self.gimbal_dof_idx is None:
            return
        gimbal_pos = self.robot.get_dofs_position(self.gimbal_dof_idx)
        gimbal_vel = self.robot.get_dofs_velocity(self.gimbal_dof_idx)
        self.gimbal_yaw_ang.copy_(gimbal_pos[:, 0])
        base_yaw = torch.deg2rad(self.base_euler[:, 2])
        self.gimbal_world_yaw.copy_(_wrap(base_yaw + self.gimbal_yaw_ang))
        # 参考角按“云台世界 yaw 速率”命令积分；command=0 时保持不变。
        self.gimbal_yaw_ref.copy_(_wrap(self.gimbal_yaw_ref + self.commands[:, 3] * self.dt))
        self.gimbal_yaw_abs_vel.copy_(self.robot.get_ang()[:, 2] + gimbal_vel[:, 0])
        self._update_translation_metrics()

    def _reset_task_buffers(self, env_idx):
        selected = slice(None) if env_idx is None else env_idx
        # 初始 base 姿态与云台默认角都为 0（单位四元数），故参考角从 0 起。
        self.gimbal_yaw_ang[selected] = 0.0
        self.gimbal_world_yaw[selected] = 0.0
        self.gimbal_yaw_ref[selected] = 0.0
        self.gimbal_yaw_abs_vel[selected] = 0.0
        self.translation_window_age[selected] = 0
        self.translation_path_origin[selected] = self.base_pos[selected, :2]
        self.translation_path_yaw[selected] = self.gimbal_world_yaw[selected] + self.get_translation_direction()[selected]
        self.translation_position_history[selected] = self.base_pos[selected, None, :2].expand(
            -1, self.translation_history_length, -1
        )
        self.translation_path_progress[selected] = self.translation_path_cross_track[selected] = 0.0
        

    # ============ 观测（父类任务扩展点） ============

    def _get_task_observation_components(self):
        base_yaw = torch.deg2rad(self.base_euler[:, 2])
        rel = _wrap(self.gimbal_world_yaw - base_yaw)  # base → gimbal 相对角
        err = _wrap(self.gimbal_yaw_ref - self.gimbal_world_yaw)  # 离目标还差多少
        direction = self.get_translation_direction()
        return {
            **({"translation_direction": torch.stack(
                [torch.sin(direction), torch.cos(direction)], dim=-1)
            } if self._has_direction_command() else {}), #前面只味4个            # "gimbal_yaw_joint": (self.gimbal_world_yaw * self.obs_scales["gimbal_yaw_angle"]).unsqueeze(-1),  # 1
            "gimbal_yaw_2_base_angle": torch.stack([torch.sin(rel), torch.cos(rel)], dim=-1),  # 2
            # "gimbal_yaw_error": torch.stack([torch.sin(err), torch.cos(err)], dim=-1),  # 2
            # "gimbal_yaw_abs_vel": (
            #     self.gimbal_yaw_abs_vel * self.obs_scales["gimbal_yaw_vel"]
            # ).unsqueeze(-1),  # 1
            # 前向速度估计（IMU 比力 + 轮速互补滤波，赛博真机接口）。base 系 x 分量；
            # 本车非完整约束，base 横向≈0，故策略可用 cos(rel) 投到云台系前向。
            # "estimated_forward_vel": (
            #     self.estimated_base_lin_vel * self.obs_scales["lin_vel"]
            # ).unsqueeze(-1),  # 1       ？ TODO这个后续好像都不用了
        }

    def _get_task_privileged_observation_components(self):
        return {
            "privileged_gimbal_world_yaw": torch.stack(
                [torch.sin(self.gimbal_world_yaw), torch.cos(self.gimbal_world_yaw)], dim=-1
            ),
            # "privileged_gimbal_yaw_ref": self.gimbal_yaw_ref.unsqueeze(-1),#有必要给吗，这之前是为了给云台闭环的。现在都是开换了，，其实执行成什么样子和现在是什么样子总是有个差距的，，其实更严的应该说是pd控制器就是有个稳态误差的,,,算了我觉得这种没必要
            "privileged_base_height": (self.base_height * self.obs_scales["base_height"]).unsqueeze(-1), #确实还是给，我看基础的里面好像只有分别的两条腿的长度
            "privileged_translation_path_progress": (
                self.translation_path_progress * self.translation_distance_scale
            ).unsqueeze(-1),
            "privileged_translation_path_cross_track": (
                self.translation_path_cross_track * self.translation_distance_scale
            ).unsqueeze(-1),
            "privileged_translation_segment_fraction": (
                self.translation_window_age.to(gs.tc_float) / self.translation_resample_steps
            ).clamp(0.0, 1.0).unsqueeze(-1),

            #我们来思考一下需要哪些
            #1.多个action ：1.速度吗，还是位移呢，，，保险起见（如果重心不在中心的话）我觉得先不给2.相对于出生位置的平移世界系 3.command指令（有了） 4.云台方向（需要吗，，同1）
            #2.云台绝对世界方向+command：世界位移下的平移方向？是要分别给还是处理好给呢，其实command已经给了，考虑实车的话
            #3.base 的速度（说真的给速度吗，reward给位移算是板上钉钉但是这里给reward吗），但反正这个给的就是last_n_action里面需要包括的

            #我觉得就给“出生坐标系”下我在哪里就可以了，因为不同于eval可能会有连续方向的变化，目前是训练环境里面只有一个采样，，后面或许会试一试多采样模拟连续速度方向命令吧
            #方向是不可能一直对的，但是critical本质不也是按照reward行事吗，我觉得给给也没事，但是这版本是保险的，先不给
            #给一个相对于世界坐标位移，是否要给个近期时间段的，，，还有就是世界位移关键

            #那我现在 觉得可以给1.给gimbal_abs_yaw+command
            #2.base_Height
            #3. 相对于初始时刻云台系方向base原点的base位移（给大小还是给方向）（这个目前是比较模糊的，因为如果给世界位移就不能不给时间，但是如果是规定了某种“时间段/方向段”的位移，其实就可以不给，具体看reward如何实现）
            #4. last_5_action（主要包含3的位移/command+云台abs/速度大小，先固定给5帧，考虑提config）

        #方向和速度的质量都给critical来评测
        }

    # ============ 诊断 ============

    def get_rotate_diagnostics(self, env_idx=0):
        return {
            "gimbal_joint": self.gimbal_yaw_ang[env_idx].detach(),
            "gimbal_world_yaw": self.gimbal_world_yaw[env_idx].detach(),
            "gimbal_yaw_ref": self.gimbal_yaw_ref[env_idx].detach(),
            "gimbal_yaw_error": _wrap(self.gimbal_yaw_ref[env_idx] - self.gimbal_world_yaw[env_idx]).detach(),
        }
