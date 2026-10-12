"""共享策略，按固定任务分组应用课程、命令、奖励和终止。"""

from copy import deepcopy
import math

import torch

from ..locomotion.env import LocomotionEnv
from .buffers import LocomotionBuffers, StairsDownBuffers
from .curriculum import allocate_counts, resolve_tasks
from .rewards import MultiTerrainRewards
from .terrain import TaskTerrain


class MultiTerrainEnv(MultiTerrainRewards, LocomotionEnv):
    EVENT_REWARDS = {"task_progress", "task_success", "death"}

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, *, evaluation=False, **kwargs):
        self.tasks = resolve_tasks(env_cfg["multi_terrain"], require_locomotion=not evaluation)
        self.task_terrain = TaskTerrain(self.tasks, allocate_counts(self.tasks, num_envs))
        reward_cfg = deepcopy(reward_cfg)
        reward_cfg["terrain_reward_scales"] = {}
        names = set().union(*(stage["rewards"] for stage in self.task_terrain.stages))
        for name in names:
            if not callable(getattr(self, "_reward_" + name, None)):
                raise ValueError(f"unknown task reward: {name}")
        reward_cfg["reward_scales"] = dict.fromkeys(sorted(names), 0.0)
        allowed_targets = {"domain_rand", "sensor_noise"}
        for stage in (kwargs.get("curriculum_cfg") or {}).get("stages", []):
            if set(stage.get("targets", {})) - allowed_targets:
                raise ValueError("multi_terrain global curriculum only controls domain_rand/sensor_noise; configure task stages separately")
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, **kwargs)

    def _register_task_curriculum(self):
        # 在 Scene 选定碰撞容量前替换地形；几何/任务快照保存在 multi_terrain。
        self.terrain = self.task_terrain

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
        self.farthest_x[ids] = self.base_pos[ids, 0] - self.terrain_spawn_centers[ids, 0]
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
        x = torch.minimum(local[:, 0], self._value("goal_x"))
        self.progress.copy_((x - self.farthest_x).clamp_min(0) * self.traversal)
        self.farthest_x.copy_(torch.maximum(self.farthest_x, x))
        advanced = self.farthest_x - self.progress_anchor >= self._value("progress_epsilon")
        self.progress_anchor.copy_(torch.where(advanced, self.farthest_x, self.progress_anchor))
        self.stall_steps.copy_(torch.where(advanced, 0, self.stall_steps + 1))
        wheels = self.robot.get_links_pos(self.wheel_links_idx)
        wheel_heights = self.terrain.height_at(wheels[:, :, :2].reshape(-1, 2)).reshape(self.num_envs, 2)
        wheel_clear = (wheels[:, :, 0] - self.terrain_spawn_centers[:, :1]
                       >= (self._value("obstacle_end") + self._value("wheel_clearance_m"))[:, None]).all(dim=1)
        wheel_supported = ((wheels[:, :, 2] - wheel_heights - self.wheel_radius).abs()
                           <= self._value("wheel_surface_tolerance_m")[:, None]).all(dim=1)
        self.success.copy_(
            self.traversal & (local[:, 0] >= self._value("goal_x")) & wheel_clear & wheel_supported
            & (local[:, 1].abs() <= self._value("corridor_half_width"))
            & (self.wheel_contact > 0.5).all(dim=1) & (self.base_contact < 0.5)
            & (self.base_euler[:, 0].abs() <= self._value("max_roll_deg"))
            & (self.base_euler[:, 1].abs() <= self._value("max_pitch_deg")))
        # 专用状态：与初始化/重置保持相同的任务顺序。
        self._update_locomotion_tracking()
        self._update_stairs_down_landing(wheels, wheel_heights)

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
        return self.traversal & (outside | (stalled & ~self.success))

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
                metrics[f"task/{name}/env_fraction"] = self.terrain.counts[i] / self.num_envs
                metrics[f"task/{name}/difficulty"] = self.terrain.controller.levels[i]
                metrics[f"task/{name}/curriculum_success_rate"] = self.terrain.controller.rates[i]
                metrics[f"task/{name}/window_episodes"] = self.terrain.controller.counts[i]
                if mask.any():
                    metrics[f"task/{name}/success"] = self.outcome_success[mask].float()
                    metrics[f"task/{name}/failure"] = self.terminated_buf[mask].float()
                    metrics[f"task/{name}/duration_s"] = self.task_steps[mask] * self.dt
                    if name != "locomotion":
                        metrics[f"task/{name}/progress_m"] = self.farthest_x[mask] - self.episode_start_x[mask]
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
