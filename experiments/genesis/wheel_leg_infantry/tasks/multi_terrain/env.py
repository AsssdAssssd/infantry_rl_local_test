"""共享策略，按固定任务分组应用课程、命令、奖励和终止。"""

from copy import deepcopy
import math

import torch

from ..locomotion.env import LocomotionEnv
from .buffers import LocomotionBuffers, StairsDownBuffers, StairsUpBuffers
from .curriculum import allocate_counts, initial_task_counts, resolve_tasks
from .rewards import MultiTerrainRewards
from .terrain import TaskTerrain


class MultiTerrainEnv(MultiTerrainRewards, LocomotionEnv):
    EVENT_REWARDS = {"task_progress", "task_success", "death", "stairs_up_support", "stairs_up_retract",
                     "stairs_up_takeoff", "stairs_up_landing_distance", "stairs_up_approach_lift", "stairs_up_backswing"}

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, *, evaluation=False, **kwargs):
        self.tasks = resolve_tasks(env_cfg["multi_terrain"], require_locomotion=not evaluation)
        self.task_terrain = TaskTerrain(self.tasks, initial_task_counts(self.tasks, num_envs, kwargs.get("curriculum_cfg")))
        reward_cfg = deepcopy(reward_cfg)
        reward_cfg["terrain_reward_scales"] = {}
        names = set().union(*(stage["rewards"] for stage in self.task_terrain.stages))
        for name in names:
            if not callable(getattr(self, "_reward_" + name, None)):
                raise ValueError(f"unknown task reward: {name}")
        reward_cfg["reward_scales"] = dict.fromkeys(sorted(names), 0.0)
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, **kwargs)

    def _register_task_curriculum(self):
        # 在 Scene 选定碰撞容量前替换地形；几何/任务快照保存在 multi_terrain。
        self.terrain = self.task_terrain
        self.curriculum.register_target("task_weights", self._apply_task_weights)

    def _apply_task_weights(self, weights):
        self.terrain.set_counts(allocate_counts(self.tasks, self.num_envs, weights))

    def _add_terrain(self):
        self.terrain_entity = self.terrain.add_to_scene(self.scene)
        self.friction_terrain_entities = self.terrain_entity

    def _initialize_task_buffers(self):
        # 所有任务共用：分组、进度、计时、回合结果。
        self.task_ids = self.terrain.fixed_types
        self.traversal = torch.tensor([name != "locomotion" for name in self.tasks], device=self.device)[self.task_ids]
        self.episode_start_x = torch.zeros_like(self.base_height)
        self.progress = torch.zeros_like(self.base_height)
        self.farthest_x = torch.zeros_like(self.base_height)
        self.progress_anchor = torch.zeros_like(self.base_height)
        self.stall_steps = torch.zeros_like(self.episode_length_buf)
        self.task_steps = torch.zeros_like(self.episode_length_buf)
        self.success = torch.zeros_like(self.traversal)
        self.outcome_success = torch.zeros_like(self.traversal)
        # 按任务集中创建；每类内部定义其全部状态、专用参数表与 reset 行为。
        self.locomotion_buffers = LocomotionBuffers.create(self)
        self.stairs_down_buffers = StairsDownBuffers.create(self)
        self.stairs_up_buffers = StairsUpBuffers.create(self)

        # 所有任务共用：按当前地形/难度查表的奖励、命令及终止参数。
        self.reward_tables = {name: self.base_pos.new_tensor([
            stage["rewards"].get(name, 0.0) * (1.0 if name in self.EVENT_REWARDS else self.dt)
            for stage in self.terrain.stages]) for name in self.reward_functions}
        self.stage_commands = [(stage["commands"], self._build_command_limits(stage["commands"]))
                               for stage in self.terrain.stages]
        self.tables = {}
        for name, default in (("corridor_half_width", 1.0), ("stall_timeout_s", 5.0), ("progress_epsilon", 0.01),
                              ("episode_length_s", 20.0), ("termination_if_roll_greater_than", 10.0),
                              ("termination_if_pitch_greater_than", 10.0), ("tilt_termination_duration_s", 0.3),
                              ("base_contact_termination_duration_s", 0.25)):
            self.tables[name] = self.base_pos.new_tensor([s["termination"].get(name, default) for s in self.terrain.stages])
        for name, default in (("max_roll_deg", 12.0), ("max_pitch_deg", 15.0),
                              ("wheel_clearance_m", 0.15), ("wheel_surface_tolerance_m", 0.08)):
            self.tables[name] = self.base_pos.new_tensor([s.get("success", {}).get(name, default) for s in self.terrain.stages])
        self.tables["direction"] = self.base_pos.new_tensor([t.direction for t in self.terrain.tracks])
        self.tables["goal_x"] = self.base_pos.new_tensor([t.goal_x for t in self.terrain.tracks])
        self.tables["obstacle_end"] = self.base_pos.new_tensor([t.obstacle_end for t in self.terrain.tracks])
        self.tables["resample_steps"] = self.base_pos.new_tensor([
            max(1, math.ceil(s["resampling_time_s"] / self.dt)) for s in self.terrain.stages])
        self.tables["min_x"] = self.base_pos.new_tensor([-t.tile_size[0] / 2 + t.boundary_margin for t in self.terrain.tracks])
        self.tables["max_x"] = self.base_pos.new_tensor([t.tile_size[0] / 2 - t.boundary_margin for t in self.terrain.tracks])
        # 基类每拍回调，具体重采样周期由每项任务决定。
        self.resample_step = 1

    def _value(self, name):
        return self.tables[name][self.terrain_tile_index]

    def _reset_task_buffers(self, env_idx):
        # 通用回合状态。
        ids = slice(None) if env_idx is None else env_idx
        self.traversal[ids] = torch.tensor([name != "locomotion" for name in self.tasks], device=self.device)[self.task_ids[ids]]
        self.stairs_down_buffers.task_mask[ids] = torch.tensor(
            [name == "stairs_down" for name in self.tasks], device=self.device)[self.task_ids[ids]]
        self.stairs_up_buffers.task_mask[ids] = torch.tensor(
            [name == "stairs_up" for name in self.tasks], device=self.device)[self.task_ids[ids]]
        self.farthest_x[ids] = (self.base_pos[ids, 0] - self.terrain_spawn_centers[ids, 0]) * self._value("direction")[ids]
        self.episode_start_x[ids] = self.farthest_x[ids]
        self.progress_anchor[ids] = self.farthest_x[ids]
        self.progress[ids] = 0
        self.stall_steps[ids] = 0
        self.task_steps[ids] = 0
        self.success[ids] = False
        self.outcome_success[ids] = False
        self.tilt_termination_steps = (self._value("tilt_termination_duration_s") / self.dt).ceil().long().clamp_min(1)
        self.base_contact_termination_steps = (self._value("base_contact_termination_duration_s") / self.dt).ceil().long().clamp_min(1)

        # 专用状态：与初始化保持相同的任务顺序。
        self.locomotion_buffers.reset(env_idx)
        self.stairs_down_buffers.reset(env_idx)
        self.stairs_up_buffers.reset(env_idx)
        state = self.stairs_up_buffers
        tiles = self.terrain_tile_index[ids]
        state.approach_start_height[ids] = state.bottom_height(self)[ids]
        state.approach_peak_height[ids] = torch.minimum(
            state.approach_start_height[ids],
            state.tables["approach_top_height"][tiles] + state.tables["approach_clearance_m"][tiles])
        target_angle = torch.deg2rad(state.tables["backswing_angle_deg"][tiles])
        state.backswing_peak[ids] = (self.leg_angle[ids].clamp_min(0) / target_angle[:, None]).clamp_max(1)

    def _resample_commands(self, envs_idx):
        selected = torch.ones_like(self.traversal) if envs_idx is None else envs_idx
        selected = selected & (self.task_steps % self._value("resample_steps").long() == 0)
        if not selected.any():
            return
        for i, (cfg, limits) in enumerate(self.stage_commands):
            mask = selected & (self.terrain_tile_index == i)
            if not mask.any():
                continue
            # 采样支持平地静止概率和高速转向限制，写回仅限本任务、本回合。
            commands = self._sample_commands(cfg, limits)
            self.commands[mask] = commands[mask]

    def _update_task_state(self):
        self.task_steps += 1
        local = self.base_pos[:, :2] - self.terrain_spawn_centers
        direction = self._value("direction")
        x = torch.minimum(local[:, 0] * direction, self._value("goal_x") * direction)
        self.progress.copy_((x - self.farthest_x).clamp_min(0) * self.traversal)
        self.farthest_x.copy_(torch.maximum(self.farthest_x, x))
        advanced = self.farthest_x - self.progress_anchor >= self._value("progress_epsilon")
        self.progress_anchor.copy_(torch.where(advanced, self.farthest_x, self.progress_anchor))
        self.stall_steps.copy_(torch.where(advanced, 0, self.stall_steps + 1))
        wheels = self.robot.get_links_pos(self.wheel_links_idx)
        wheel_heights = self.terrain.height_at(wheels[:, :, :2].reshape(-1, 2)).reshape(self.num_envs, 2)
        wheel_clear = ((wheels[:, :, 0] - self.terrain_spawn_centers[:, :1]) * direction[:, None]
                       >= (self._value("obstacle_end") * direction + self._value("wheel_clearance_m"))[:, None]).all(dim=1)
        wheel_supported = ((wheels[:, :, 2] - wheel_heights - self.wheel_radius).abs()
                           <= self._value("wheel_surface_tolerance_m")[:, None]).all(dim=1)
        self.success.copy_(
            self.traversal & (local[:, 0] * direction >= self._value("goal_x") * direction) & wheel_clear & wheel_supported
            & (local[:, 1].abs() <= self._value("corridor_half_width"))
            & (self.wheel_contact > 0.5).all(dim=1) & (self.base_contact < 0.5)
            & (self.base_euler[:, 0].abs() <= self._value("max_roll_deg"))
            & (self.base_euler[:, 1].abs() <= self._value("max_pitch_deg")))
        # 专用状态：与初始化/重置保持相同的任务顺序。
        self._update_locomotion_tracking()
        self._update_stairs_down_landing(wheels, wheel_heights)
        self._update_stairs_up_ascent(wheels)

    def _update_stairs_up_ascent(self, wheels):
        state = self.stairs_up_buffers
        if "stairs_up" not in self.tasks:
            return
        cfg = {key: value[self.terrain_tile_index] for key, value in state.tables.items()}
        lower_contact = self._stairs_up_lower_contact(wheels, cfg)
        # 先确认双轮实际站在台下，避免 reset 后接触信号尚未更新或空中出生误发奖。
        state.lower_ground_seen |= state.task_mask & lower_contact.all(dim=1)
        lifted = (wheels[..., 2] - self.wheel_radius >
                  (cfg["lower_height"] + cfg["wheel_bottom_tolerance_m"])[:, None]).all(dim=1)
        state.takeoff_event.copy_(state.task_mask & state.lower_ground_seen & lifted & ~state.takeoff_awarded)
        state.takeoff_awarded |= state.takeoff_event
        # 轮子仍可抵着立面；离地看实际高度，重新落回则必须有台下地面接触。
        state.fallen_off |= state.task_mask & state.takeoff_awarded & lower_contact.any(dim=1)
        self._update_stairs_up_approach(cfg, lower_contact.all(dim=1))
        contacts = self.robot.get_contacts(exclude_self_contact=True)
        self._update_stairs_up_landing_distance(contacts, cfg)
        self._update_stairs_up_wheel_wall_force(contacts, cfg)
        # Genesis 的接触 link 索引是全场景索引；base_link_idx 是机器人局部索引。
        base_link = self.robot.link_start + self.base_link_idx
        is_a = contacts["link_a"] == base_link
        is_b = contacts["link_b"] == base_link
        force_z = torch.where(is_a, contacts["force_a"][..., 2], contacts["force_b"][..., 2])
        position = contacts["position"]
        local_x = position[..., 0] - self.terrain_spawn_centers[:, :1]
        local_y = position[..., 1] - self.terrain_spawn_centers[:, 1:2]
        surface_contact = (
            contacts["valid_mask"] & (is_a | is_b)
            & (local_x >= self._value("obstacle_end")[:, None])
            & (local_x <= self._value("max_x")[:, None])
            & (local_y.abs() <= self._value("corridor_half_width")[:, None])
            & ((position[..., 2] - cfg["top_height"][:, None]).abs()
               <= cfg["support_surface_tolerance_m"][:, None])
            & (contacts["normal"][..., 2].abs() >= cfg["support_normal_z"][:, None])
            & (self.base_pos[:, 2:3] > position[..., 2])
        )
        # 首次触台就切换到收腿引导，不等稳定支撑奖励；此拍之前的缩腿不补奖。
        touching_top = state.task_mask & (surface_contact & (force_z > 0)).any(dim=1)
        first_contact = touching_top & ~state.top_contact_seen
        state.retract_start_length.copy_(torch.where(first_contact[:, None], self.leg_length, state.retract_start_length))
        state.top_contact_seen |= touching_top
        self._update_stairs_up_backswing(wheels, cfg)
        top_contact = surface_contact & (
            local_x >= (self._value("obstacle_end") + cfg["support_margin_m"])[:, None])
        upward_force = torch.where(top_contact, force_z.clamp_min(0), 0).sum(dim=1)
        vertical_speed = (self.base_lin_vel * self.projected_gravity).sum(dim=1).abs()
        quiet = ((vertical_speed <= cfg["max_vertical_speed_m_s"])
                 & (self.base_ang_vel[:, :2].norm(dim=1) <= cfg["max_roll_pitch_speed_rad_s"]))
        supported = state.task_mask & quiet & (upward_force >= cfg["support_force_n"])
        state.support_steps.copy_(torch.where(supported, state.support_steps + 1, 0))
        state.support_event.copy_(supported & (state.support_steps >= cfg["support_steps"]) & ~state.support_awarded)
        state.support_awarded |= state.support_event

        wheel_x = wheels[..., 0] - self.terrain_spawn_centers[:, :1]
        wheel_y = wheels[..., 1] - self.terrain_spawn_centers[:, 1:2]
        wheel_bottom = wheels[..., 2] - self.wheel_radius
        # 首次双轮接触台面即可关闭收腿，不等整轮后缘越沿或最终站稳。
        on_top = (
            (wheel_x >= (self._value("obstacle_end") + cfg["wheel_edge_margin_m"])[:, None])
            & (wheel_x <= self._value("max_x")[:, None])
            & (wheel_y.abs() <= self._value("corridor_half_width")[:, None])
            & ((wheel_bottom - cfg["top_height"][:, None]).abs() <= cfg["wheel_bottom_tolerance_m"][:, None])
            & (self.wheel_contact > 0.5)
        ).all(dim=1)
        state.wheels_landed |= state.task_mask & on_top
        eligible = state.task_mask & state.top_contact_seen & ~state.wheels_landed
        score = ((state.retract_start_length - self.leg_length) / cfg["retract_shortening_m"][:, None]).clamp(0, 1)
        state.retract_progress.copy_((score - state.retract_peak).clamp_min(0).mean(dim=1) * eligible)
        state.retract_peak.copy_(torch.where(eligible[:, None], torch.maximum(state.retract_peak, score), state.retract_peak))
        state.retract_awarded |= eligible & (state.retract_peak >= 1 - 1e-6).all(dim=1)
        # 落台后允许向上弹跳；任一轮明显跌到台面以下即失败，无需等落到低处地面。
        below_top = (wheel_bottom < (cfg["top_height"] - cfg["fall_height_tolerance_m"])[:, None]).any(dim=1)
        state.fallen_off |= state.task_mask & state.wheels_landed & below_top

        # 最终成功独立于中间奖励：直接登上台阶的策略也可以完成任务。
        standing = state.task_mask & self.success & quiet & (self.base_height >= cfg["standing_height_m"])
        state.success_steps.copy_(torch.where(standing, state.success_steps + 1, 0))
        self.success &= ~state.task_mask | (state.success_steps >= cfg["success_steps"])

    def _update_stairs_up_landing_distance(self, contacts, cfg):
        """第一次任一轮真实触台时，按接触点距台阶入口沿的水平距离发一次奖励。"""
        state = self.stairs_up_buffers
        state.landing_distance_event.zero_()
        if contacts["valid_mask"].shape[1] == 0:
            return
        wheel_links = torch.as_tensor(self.wheel_links_idx, device=self.device) + self.robot.link_start
        is_a = (contacts["link_a"][..., None] == wheel_links).any(dim=-1)
        is_b = (contacts["link_b"][..., None] == wheel_links).any(dim=-1)
        force_z = torch.where(is_a, contacts["force_a"][..., 2], contacts["force_b"][..., 2])
        position = contacts["position"]
        local_x = position[..., 0] - self.terrain_spawn_centers[:, :1]
        local_y = position[..., 1] - self.terrain_spawn_centers[:, 1:2]
        distance = local_x - self._value("obstacle_end")[:, None]
        on_top = (
            contacts["valid_mask"] & (is_a | is_b) & (force_z > 0)
            & (distance >= 0) & (local_x <= self._value("max_x")[:, None])
            & (local_y.abs() <= self._value("corridor_half_width")[:, None])
            & ((position[..., 2] - cfg["top_height"][:, None]).abs()
               <= cfg["support_surface_tolerance_m"][:, None])
            & (contacts["normal"][..., 2].abs() >= cfg["support_normal_z"][:, None])
        )
        first = state.task_mask & on_top.any(dim=1) & ~state.first_landing_seen
        # 同拍多个轮/接触点落台时取最近沿的落点，避免用较远轮掩盖贴沿落地。
        nearest = torch.where(on_top, distance, torch.inf).amin(dim=1)
        state.landing_distance_event.copy_(torch.where(first, nearest, 0))
        state.first_landing_distance.copy_(torch.where(first, nearest, state.first_landing_distance))
        # 即使第一次恰好落在边沿（距离为 0），也必须锁存，后续滚动不能补领。
        state.first_landing_seen |= first

    def _update_stairs_up_wheel_wall_force(self, contacts, cfg):
        """只累计轮子与台阶竖直立面的法向接触力，排除台面支撑及切向摩擦。"""
        state = self.stairs_up_buffers
        wheel_links = torch.as_tensor(self.wheel_links_idx, device=self.device) + self.robot.link_start
        is_a = (contacts["link_a"][..., None] == wheel_links).any(dim=-1)
        is_b = (contacts["link_b"][..., None] == wheel_links).any(dim=-1)
        force = torch.where(is_a[..., None], contacts["force_a"], contacts["force_b"])
        position = contacts["position"]
        local_x = position[..., 0] - self.terrain_spawn_centers[:, :1]
        local_y = position[..., 1] - self.terrain_spawn_centers[:, 1:2]
        tolerance = cfg["wall_surface_tolerance_m"][:, None]
        on_wall = (
            contacts["valid_mask"] & (is_a | is_b)
            & (contacts["normal"][..., 2].abs() <= cfg["wall_normal_z_max"][:, None])
            & (local_x >= cfg["approach_edge"][:, None] - tolerance)
            & (local_x <= self._value("obstacle_end")[:, None] + tolerance)
            & (local_y.abs() <= self._value("corridor_half_width")[:, None])
            & (position[..., 2] >= cfg["lower_height"][:, None] - tolerance)
            & (position[..., 2] <= cfg["top_height"][:, None] + tolerance)
        )
        # 逐接触点投影后求和，双轮/多个接触点的力不会因方向相反而抵消。
        normal_force = (force * contacts["normal"]).sum(dim=-1).abs()
        state.wheel_wall_force.copy_(torch.where(on_wall, normal_force, 0).sum(dim=1) * state.task_mask)

    def _stairs_up_lower_contact(self, wheels, cfg):
        """逐轮确认台下地面接触，排除台面、腾空和立面上的高处接触。"""
        wheel_x = wheels[..., 0] - self.terrain_spawn_centers[:, :1]
        wheel_y = wheels[..., 1] - self.terrain_spawn_centers[:, 1:2]
        return (
            (wheel_x < cfg["approach_edge"][:, None])
            & (wheel_y.abs() <= self._value("corridor_half_width")[:, None])
            & ((wheels[..., 2] - self.wheel_radius - cfg["lower_height"][:, None]).abs()
               <= cfg["wheel_bottom_tolerance_m"][:, None])
            & (self.wheel_contact > 0.5)
        )

    def _update_stairs_up_approach(self, cfg, on_lower_ground):
        """近台阶伸腿抬升的有界增量；全回合最高值防止反复升降或离开区域刷分。"""
        state = self.stairs_up_buffers
        local = self.base_pos[:, :2] - self.terrain_spawn_centers
        distance = cfg["approach_edge"] - local[:, 0]
        eligible = (
            state.task_mask & ~state.support_awarded & (self.base_contact < 0.5)
            & (distance > 0) & (distance <= cfg["approach_distance_m"])
            & (local[:, 1].abs() <= self._value("corridor_half_width")) & on_lower_ground
            & (self.base_euler[:, 0].abs() <= self._value("max_roll_deg"))
            & (self.base_euler[:, 1].abs() <= self._value("max_pitch_deg"))
        )
        # 底盘下底面略高于眼前台阶即可，不要求固定机体原点高度，也不奖励继续抬高。
        target = cfg["approach_top_height"] + cfg["approach_clearance_m"]
        height = torch.minimum(state.bottom_height(self), target)
        remaining_height = (target - state.approach_start_height).clamp_min(1e-6)
        state.approach_lift_progress.copy_(
            (height - state.approach_peak_height).clamp_min(0) / remaining_height * eligible)
        # 区域外或腾空抬升也更新最高值，随后进入区域/落地不能补领那段奖励。
        state.approach_peak_height.copy_(torch.maximum(state.approach_peak_height, height))

    def _update_stairs_up_backswing(self, wheels, cfg):
        """底盘接触台面前的后摆增量，每条腿独立记录最高进度，合计最多 1。"""
        state = self.stairs_up_buffers
        local = self.base_pos[:, :2] - self.terrain_spawn_centers
        distance = cfg["approach_edge"] - local[:, 0]
        wheel_x = wheels[..., 0] - self.terrain_spawn_centers[:, :1]
        wheel_y = wheels[..., 1] - self.terrain_spawn_centers[:, 1:2]
        eligible = (
            state.task_mask & ~state.top_contact_seen & ~state.support_awarded
            & (self.base_contact < 0.5) & ~self._tilt_failure()
            & (distance.abs() <= cfg["approach_distance_m"])
            & (local[:, 1].abs() <= self._value("corridor_half_width"))
            & (wheel_x < cfg["approach_edge"][:, None]).all(dim=1)
            & (wheel_y.abs() <= self._value("corridor_half_width")[:, None]).all(dim=1)
        )
        # 正腿角代表轮子相对机体向 -x 后摆；与机体世界俯仰角无关。
        target = torch.deg2rad(cfg["backswing_angle_deg"])
        score = (self.leg_angle.clamp_min(0) / target[:, None]).clamp_max(1)
        state.backswing_progress.copy_((score - state.backswing_peak).clamp_min(0).mean(dim=1) * eligible)
        # 区域外的后摆也记录，不能进入区域补领，也不能前摆后再后摆刷分。
        state.backswing_peak.copy_(torch.maximum(state.backswing_peak, score))

    def _update_locomotion_tracking(self):
        self.locomotion_buffers.tracking_errors += torch.stack((
            self._reward_tracking_lin_vel(), self._reward_tracking_ang_vel(), self._reward_base_height()), dim=1)

    def _update_stairs_down_landing(self, wheels, wheel_heights):
        state = self.stairs_down_buffers
        # 不要求曾经双轮腾空，兼容单轮先落地或保持接触的平顺下台阶。
        wheel_x = wheels[:, :, 0] - self.terrain_spawn_centers[:, :1]
        beyond_edge = wheel_x > self._value("obstacle_end")[:, None]
        on_surface = ((wheels[:, :, 2] - wheel_heights - self.wheel_radius).abs()
                      <= self._value("wheel_surface_tolerance_m")[:, None])
        lower_contact = (beyond_edge & on_surface & (self.wheel_contact > 0.5)).any(dim=1)
        landed = state.task_mask & lower_contact & ~state.landing_started & (self.task_steps > 1)
        state.landing_steps_left.sub_(1).clamp_min_(0)
        state.landing_steps_left.copy_(torch.where(
            landed, state.landing_window_steps[self.terrain_tile_index], state.landing_steps_left))
        state.landing_started |= landed
        # 窗口内再次弹起或机身触地也继续计入振荡；每回合只启动一次。
        state.landing_gate.copy_((state.landing_steps_left > 0).to(self.base_pos.dtype))

    def _tilt_failure(self):
        return ((self.base_euler[:, 0].abs() > self._value("termination_if_roll_greater_than"))
                | (self.base_euler[:, 1].abs() > self._value("termination_if_pitch_greater_than")))

    def _episode_timeout(self):
        # 独立计时不受 runner.init_at_random_ep_len 修改的 episode_length_buf 干扰。
        return self.task_steps >= (self._value("episode_length_s") / self.dt).ceil().long()

    def _task_termination(self):
        local = self.base_pos[:, :2] - self.terrain_spawn_centers
        outside = ((local[:, 1].abs() > self._value("corridor_half_width"))
                   | (local[:, 0] < self._value("min_x")) | (local[:, 0] > self._value("max_x")))
        stalled = self.stall_steps * self.dt >= self._value("stall_timeout_s")
        return self.traversal & (outside | (stalled & ~self.success) | self.stairs_up_buffers.fallen_off)

    def _task_success(self):
        return self.success

    def _get_reward_scale(self, name):
        return self.reward_tables[name][self.terrain_tile_index]

    def _record_task_outcomes(self):
        ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if not len(ids):
            return
        passed = self.success & ~self.terminated_buf
        for i, stage in enumerate(self.terrain.stages):
            if stage["terrain"]["preset"] != "plane":
                continue
            cfg = stage["tracking_success"]
            limits = self.base_pos.new_tensor([cfg["lin_vel_rmse"], cfg["ang_vel_rmse"], cfg["height_rmse"]]).square()
            mse = self.locomotion_buffers.tracking_errors / self.task_steps.clamp_min(1)[:, None]
            passed |= ((self.terrain_tile_index == i) & ~self.terminated_buf & self._episode_timeout()
                       & (mse <= limits).all(dim=1))
        rows = torch.stack((self.terrain_tile_index[ids], self.terrain.assignment_epoch[ids], passed[ids].long()), dim=1).cpu().tolist()
        groups = {}
        for tile, epoch, success in rows:
            task, level = self.terrain.variants[tile]
            wins, count = groups.get((task, level, epoch), (0, 0))
            groups[task, level, epoch] = wins + success, count + 1
        for (task, level, epoch), (wins, count) in groups.items():
            self.terrain.controller.observe(task, level, epoch, wins, count)
        self.outcome_success.copy_(passed)

    def _reset_idx(self, env_idx=None):
        metrics = {}
        if env_idx is not None and env_idx.any():
            for i, name in enumerate(self.tasks):
                mask = env_idx & (self.task_ids == i)
                metrics[f"task/{name}/env_fraction"] = (self.task_ids == i).float().mean().item()
                metrics[f"task/{name}/difficulty"] = self.terrain.controller.levels[i]
                metrics[f"task/{name}/curriculum_success_rate"] = self.terrain.controller.rates[i]
                metrics[f"task/{name}/window_episodes"] = self.terrain.controller.counts[i]
                if mask.any():
                    metrics[f"task/{name}/success"] = self.outcome_success[mask].float()
                    metrics[f"task/{name}/failure"] = self.terminated_buf[mask].float()
                    metrics[f"task/{name}/duration_s"] = self.task_steps[mask] * self.dt
                    if name != "locomotion":
                        metrics[f"task/{name}/progress_m"] = self.farthest_x[mask] - self.episode_start_x[mask]
                    if name == "stairs_up":
                        state = self.stairs_up_buffers
                        metrics[f"task/{name}/support_reached"] = state.support_awarded[mask].float()
                        metrics[f"task/{name}/retract_reached"] = state.retract_awarded[mask].float()
                        metrics[f"task/{name}/takeoff_reached"] = state.takeoff_awarded[mask].float()
                        metrics[f"task/{name}/first_landing_reached"] = state.first_landing_seen[mask].float()
                        metrics[f"task/{name}/first_landing_distance_m"] = state.first_landing_distance[mask]
                    if name == "locomotion":
                        mse = self.locomotion_buffers.tracking_errors[mask] / self.task_steps[mask].clamp_min(1)[:, None]
                        for j, key in enumerate(("lin_vel", "ang_vel", "height")):
                            metrics[f"task/{name}/{key}_mse"] = mse[:, j]
        super()._reset_idx(env_idx)
        if metrics:
            self.extras["episode"].update(metrics)

    def terrain_state_dict(self):
        return self.terrain.controller.state_dict()

    def reset(self):
        saved = getattr(self, "_pending_terrain_state", None)
        if saved is not None:
            self.terrain.controller.load_state_dict(saved)
            self._pending_terrain_state = None
        return super().reset()
