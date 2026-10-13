"""多地形任务奖励；各 task 在 config.py 中独立选择和设置权重。"""

import torch


class MultiTerrainRewards:
    # 各通关任务共用。
    def _reward_task_progress(self):
        return self.progress * ~self.terminated_buf

    def _reward_task_success(self):
        return self.success * ~self.terminated_buf

    def _reward_stairs_up_support(self):
        return self.stairs_up_buffers.support_event & ~self.terminated_buf

    def _reward_stairs_up_retract(self):
        return self.stairs_up_buffers.retract_progress * ~self.terminated_buf

    def _reward_stairs_up_takeoff(self):
        return self.stairs_up_buffers.takeoff_event & ~self.terminated_buf

    def _reward_stairs_up_landing_distance(self):
        return self.stairs_up_buffers.landing_distance_event * ~self.terminated_buf

    def _reward_stairs_up_approach_lift(self):
        return self.stairs_up_buffers.approach_lift_progress * ~self.terminated_buf

    def _reward_stairs_up_backswing(self):
        return self.stairs_up_buffers.backswing_progress * ~self.terminated_buf

    def _reward_task_lateral(self):
        return (self.base_pos[:, 1] - self.terrain_spawn_centers[:, 1]).square()

    def _reward_task_heading(self):
        return 1 - torch.cos(torch.deg2rad(self.base_euler[:, 2])) * self._value("direction")

    def _reward_stairs_down_landing_stability(self):
        """下台阶后的振荡代价，配置负权重；不惩罚水平前进或 yaw 转动。"""
        # 速度和重力方向都在机体系，点积得到世界竖直方向速度（符号不影响平方）。
        state = self.stairs_down_buffers
        vertical_velocity = (self.base_lin_vel * self.projected_gravity).sum(dim=1)
        angular_weight = state.landing_ang_vel_weight[self.terrain_tile_index]
        oscillation = vertical_velocity.square() + angular_weight * self.base_ang_vel[:, :2].square().sum(dim=1)
        return state.landing_gate * oscillation
