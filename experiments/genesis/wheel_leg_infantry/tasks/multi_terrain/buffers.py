"""各任务专用 buffer 集中定义；同一任务的创建和回合重置放在同一个类中。

所有张量保留 num_envs 维度，任务 mask 隔离用途；部分 reset 只清空选中的环境。
通用进度、终止、奖励与命令查表仍由 env.py 管理。
"""

from dataclasses import dataclass
import math

import torch
from genesis.utils.geom import inv_quat, transform_by_quat


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
    """下台阶任务：当前回合任务 mask、每难度参数表和每回合落地窗口状态。"""

    # 任务 mask 在 reset 时随分组更新；难度参数表保持不变。
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


@dataclass
class StairsUpBuffers:
    """上台阶：触台收腿进度、首次轮子落点、双轮落台及站立确认窗口。"""

    task_mask: torch.Tensor
    tables: dict
    support_steps: torch.Tensor
    success_steps: torch.Tensor
    retract_start_length: torch.Tensor
    retract_peak: torch.Tensor
    support_awarded: torch.Tensor
    retract_awarded: torch.Tensor
    support_event: torch.Tensor
    lower_ground_seen: torch.Tensor
    takeoff_awarded: torch.Tensor
    takeoff_event: torch.Tensor
    retract_progress: torch.Tensor
    wheels_landed: torch.Tensor
    first_landing_seen: torch.Tensor
    first_landing_distance: torch.Tensor
    landing_distance_event: torch.Tensor
    fallen_off: torch.Tensor
    approach_start_height: torch.Tensor
    approach_peak_height: torch.Tensor
    approach_lift_progress: torch.Tensor
    base_collision_vertices: torch.Tensor
    top_contact_seen: torch.Tensor
    backswing_peak: torch.Tensor
    backswing_progress: torch.Tensor

    @classmethod
    def create(cls, env):
        mask = torch.tensor([name == "stairs_up" for name in env.tasks], device=env.device)[env.task_ids]
        # 兼容未保存 ascent 参数的旧运行配置；新配置在 stairs_up task 中集中设置。
        defaults = dict(
            approach_distance_m=0.50, approach_clearance_m=0.01,
            backswing_angle_deg=20.0,
            support_duration_s=0.10, support_force_n=5.0, support_normal_z=0.9,
            support_surface_tolerance_m=0.015, support_margin_m=0.005,
            retract_shortening_m=0.02, wheel_edge_margin_m=0.01, wheel_bottom_tolerance_m=0.005,
            fall_height_tolerance_m=0.03,
            success_duration_s=0.20, standing_height_m=0.20,
            max_vertical_speed_m_s=0.3, max_roll_pitch_speed_rad_s=1.5,
        )
        tables = {key: env.base_pos.new_tensor([stage.get("ascent", {}).get(key, value)
                  for stage in env.terrain.stages]) for key, value in defaults.items()}
        for name in ("support", "success"):
            tables[name + "_steps"] = (tables[name + "_duration_s"] / env.dt).ceil().long().clamp_min(1)
        tables["top_height"] = env.base_pos.new_tensor([
            track.geometry.origin[2] + track.geometry.num_steps * track.geometry.step_height
            if track.kind == "stairs_up" else 0.0 for track in env.terrain.tracks])
        tables["approach_edge"] = env.base_pos.new_tensor([
            track.geometry.start_x if track.kind == "stairs_up" else 0.0 for track in env.terrain.tracks])
        tables["lower_height"] = env.base_pos.new_tensor([
            track.geometry.origin[2] if track.kind == "stairs_up" else 0.0 for track in env.terrain.tracks])
        tables["approach_top_height"] = env.base_pos.new_tensor([
            track.geometry.origin[2] + track.geometry.step_height
            if track.kind == "stairs_up" else 0.0 for track in env.terrain.tracks])
        vertices = env.base_pos.new_zeros((1, 3))
        if "stairs_up" in env.tasks:
            # 缓存实际碰撞体在机体系中的顶点；排除视觉网格，并包含碰撞体自身偏移/旋转。
            link = env.robot.get_link(env.env_cfg.get("base_link_name", "base_link"))
            world_vertices = link.get_verts()[0]
            vertices = transform_by_quat(
                world_vertices - env.robot.get_pos()[0], inv_quat(env.robot.get_quat()[0]))
        return cls(
            task_mask=mask, tables=tables,
            support_steps=torch.zeros_like(env.episode_length_buf),
            success_steps=torch.zeros_like(env.episode_length_buf),
            retract_start_length=env.base_pos.new_zeros((env.num_envs, 2)),
            retract_peak=env.base_pos.new_zeros((env.num_envs, 2)),
            support_awarded=torch.zeros_like(mask), retract_awarded=torch.zeros_like(mask),
            support_event=torch.zeros_like(mask),
            lower_ground_seen=torch.zeros_like(mask),
            takeoff_awarded=torch.zeros_like(mask), takeoff_event=torch.zeros_like(mask),
            retract_progress=torch.zeros_like(env.base_height),
            wheels_landed=torch.zeros_like(mask), fallen_off=torch.zeros_like(mask),
            first_landing_seen=torch.zeros_like(mask),
            first_landing_distance=torch.zeros_like(env.base_height),
            landing_distance_event=torch.zeros_like(env.base_height),
            approach_start_height=torch.zeros_like(env.base_height),
            approach_peak_height=torch.zeros_like(env.base_height),
            approach_lift_progress=torch.zeros_like(env.base_height),
            base_collision_vertices=vertices,
            top_contact_seen=torch.zeros_like(mask),
            backswing_peak=env.base_pos.new_zeros((env.num_envs, 2)),
            backswing_progress=torch.zeros_like(env.base_height),
        )

    def bottom_height(self, env):
        """真实底盘碰撞体最低点的世界高度，随 roll/pitch 改变；reset 不依赖延迟的 FK。"""
        if "stairs_up" not in env.tasks:
            return env.base_pos[:, 2]
        rotated = transform_by_quat(self.base_collision_vertices[None, :, :], env.base_quat[:, None, :])
        return env.base_pos[:, 2] + rotated[..., 2].amin(dim=1)

    def reset(self, env_idx=None):
        ids = slice(None) if env_idx is None else env_idx
        self.support_steps[ids] = 0
        self.success_steps[ids] = 0
        self.retract_start_length[ids] = 0
        self.retract_peak[ids] = 0
        self.support_awarded[ids] = False
        self.retract_awarded[ids] = False
        self.support_event[ids] = False
        self.lower_ground_seen[ids] = False
        self.takeoff_awarded[ids] = False
        self.takeoff_event[ids] = False
        self.retract_progress[ids] = 0
        self.wheels_landed[ids] = False
        self.first_landing_seen[ids] = False
        self.first_landing_distance[ids] = 0
        self.landing_distance_event[ids] = 0
        self.fallen_off[ids] = False
        self.approach_start_height[ids] = 0
        self.approach_peak_height[ids] = 0
        self.approach_lift_progress[ids] = 0
        self.top_contact_seen[ids] = False
        self.backswing_peak[ids] = 0
        self.backswing_progress[ids] = 0
