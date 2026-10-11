"""rotate（旋转平移）任务配置：在 locomotion 基础上加入可控云台 yaw。

设计：
- 复用 locomotion 的全部机器人/地形/随机化/奖励/课程配置；
- 机器人换成带 gimbal_yaw_joint 的模型；
- 把 gimbal_yaw_joint 追加进 joint_names（这样父类的动作管道自动变成 7 维）；
- 增加第 4 条命令 [vx, wz, base_height, gimbal_yaw_rate]；
- 课程后期逐步放开云台速率，前期锁 0 先学“云台保持世界朝向”和多向平移走。
"""

from __future__ import annotations

import math
from copy import deepcopy

from ..locomotion.config_locomotion import get_cfgs as get_locomotion_cfgs

GIMBAL_YAW_JOINT = "gimbal_yaw_joint"
GIMBAL_PITCH_JOINT="gimbal_pitch_joint"


def get_cfgs(locomotion_cfgs=None):
    """返回 env / obs / reward / command / curriculum 五组配置。"""
    if locomotion_cfgs is None:
        locomotion_cfgs = get_locomotion_cfgs()

    env_cfg = deepcopy(locomotion_cfgs[0])
    obs_cfg = deepcopy(locomotion_cfgs[1])
    reward_cfg = deepcopy(locomotion_cfgs[2])
    command_cfg = deepcopy(locomotion_cfgs[3])
    # curriculum_cfg = deepcopy(locomotion_cfgs[4])

    # ---------- 机器人：换成带 yaw 关节的模型 ----------
    # TODO 加入pitch，先不给控制，就是让他稳定到平头
    env_cfg["num_actions"] = 6
    env_cfg["num_joints"] = 4
    env_cfg["robot_mjcf"] = "assets/robot/wheelbipeV14_2_yaw_joint/mjcf/wheelbipeV14_2.xml"
    env_cfg["gimbal_joint_names"] = [GIMBAL_YAW_JOINT,GIMBAL_PITCH_JOINT] #可惜这个模型没有pitch倒是
    # if GIMBAL_YAW_JOINT not in env_cfg["joint_names"]:
    #     env_cfg["joint_names"] = [*env_cfg["joint_names"], GIMBAL_YAW_JOINT]
    # if GIMBAL_PITCH_JOINT not in env_cfg["joint_names"]:
    #         env_cfg["joint_names"] = [*env_cfg["joint_names"], GIMBAL_PITCH_JOINT]
    env_cfg["default_joint_pos"][GIMBAL_YAW_JOINT] = 0.0
    env_cfg["default_joint_pos"][GIMBAL_PITCH_JOINT] = 0.0


    # actor 输出的云台“世界 yaw 目标”的物理量程：action(-1,1) * scale
    env_cfg["gimbal_joint_ang_scale"] = math.pi
    # env_cfg["clip_gimbal_joint_action"]= 1.0  #不是动作，不clip了；
    # ---------- 观测缩放 ----------
    obs_cfg["obs_scales"]["gimbal_yaw_angle"] = 1.0 / math.pi # TODO 后续如果变化的比较大，可以换成2唯，不会跳变
    obs_cfg["obs_scales"]["gimbal_yaw_vel"] = 1.0 / math.pi

    obs_cfg["obs_scales"]["gimbal_pitch_angle"] = 1.0 / math.pi

    # ---------- 命令：追加云台速率 ----------
    command_cfg["num_commands"] = 4
    command_cfg["gimbal_yaw_vel_range"] = [-1.0, 1.0] #TODO 这里其他的参数还是原来的，但小陀螺差异比较大，后面记得显示更改

    # ---------- 奖励 ----------
    # reward_cfg["reward_scales"]["gimbal_yaw_ang"] = -10.0 #不需要奖励了
    reward_cfg["reward_scales"]["tracking_lin_vel"] = -0.0
    reward_cfg["reward_scales"]["gated_tracking_lin_vel"] = -0.0
    reward_cfg["reward_scales"]["gimbal_spin_track_lin_vel_yaw_frame"] = -10.0
    reward_cfg["reward_scales"]["gimbal_spin_heading_error"] = -30.0
    reward_cfg["reward_scales"]["gimbal_spin_track_lin_vel_yaw_lower"] = -3.0

    # ---------- 课程：每阶段补上云台命令范围与奖励权重；默认锁 0 ----------
    #TODO 我认为还是覆盖一个比较好，课程差异还是比较大的
    # for stage in curriculum_cfg["stages"]:
    #     targets = stage.setdefault("targets", {})
    #     targets.setdefault("command_ranges", {}).setdefault("gimbal_yaw_vel_range", [0.0, 0.0])
    #     targets.setdefault("reward_scales", {}).setdefault("gimbal_yaw_ang", -10.0)
    # # 最后一个阶段放开云台速率，让策略学习“跟着 ref 转”
    # curriculum_cfg["stages"][-1]["targets"]["command_ranges"]["gimbal_yaw_vel_range"] = [-1.0, 1.0]

    return env_cfg, obs_cfg, reward_cfg, command_cfg, _curriculum_cfg()




def _curriculum_cfg() -> dict:
    # 阶段采用累计覆盖：这里只写相对上一阶段发生变化的字段。
    return {
        "enabled": True,
        "stages": [
            {
                "name": "stand",
                "start_iteration": 0,
                "targets": {
                    "domain_rand": {"strength": 0.0},
                    "sensor_noise": {"strength": 0.0},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    },
                    "command_ranges": {
                        "standing_probability": 1.0,
                        "lin_vel_range": [0.0, 0.0],
                        "ang_vel_range": [0.0, 0.0],
                        "base_height_range": [0.22, 0.22],
                        "gimbal_yaw_vel_range": [-0.0, 0.0],
                    },
                    "termination_limits": {
                        "termination_if_roll_greater_than": 20.0,
                        "termination_if_pitch_greater_than": 20.0,
                        "tilt_termination_duration_s": 0.30,
                        "base_contact_termination_duration_s": 0.25,
                    },
                },
            },
            {
                "name": "locomotion",
                "start_iteration": 500,
                "targets": {
                    "standing_reward": {"tracking_sigma": 0.10},
                    "domain_rand": {"strength": 0.0},
                    "sensor_noise": {"strength": 0.0},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    },
                    "command_ranges": {
                        "standing_probability": 0.1,
                        "lin_vel_range": [-0.1, 0.1],
                        "ang_vel_range": [-3.0, 3.0],
                        "base_height_range": [0.20, 0.30],
                        "gimbal_yaw_vel_range": [-0.0, 0.0],

                    },
                    "tracking_gate": {
                        "height_full_error": 0.03,
                        "height_zero_error": 0.09,
                        "attitude_full_angle_deg": 1.0,
                        "attitude_zero_angle_deg": 4.0,
                        "floor": 0.10,
                    },
                    "reward_scales": {
                        "standing_drift": -1.0,
                        # "tracking_lin_vel": -2.0,
                        "tracking_ang_vel": -3.0,
                        "leg_symmetry": -10.0,
                        "base_balance": -10.0,
                        # "gimbal_yaw_ang": -10.0,
                    },
                },
            },
            {
                "name": "locomotion2",
                "start_iteration": 1500,
                "targets": {
                    "standing_reward": {"tracking_sigma": 0.05},
                    "domain_rand": {"strength": 0.0},
                    "sensor_noise": {"strength": 0.0},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.22]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    },
                    "command_ranges": {
                        "standing_probability": 0.1,
                        "lin_vel_range": [-0.25, 0.25],
                        "ang_vel_range": [-6.0, 6.0],
                        "base_height_range": [0.20, 0.34],
                        "gimbal_yaw_vel_range": [-0.2, 0.2],

                    },
                    "tracking_gate": {
                        "height_full_error": 0.02,
                        "height_zero_error": 0.05,
                        "attitude_full_angle_deg": 0.8,
                        "attitude_zero_angle_deg": 3.0,
                        "floor": 0.05,
                    },
                    "reward_scales": {
                        "standing_drift": -2.0,
                        # "gated_tracking_lin_vel": 5.0,
                        # "gated_tracking_ang_vel": 7.0,
                        "base_balance": -15.0,
                        "base_contact": -15.0,
                        "landing_base_oscillation": -0.3,
                        "landing_joint_vel": -0.01,
                        # "gimbal_yaw_ang": -10.0,

                    },
                },
            },
            {
                "name": "locomotion_rand",
                "start_iteration": 3000,
                "targets": {
                    "standing_reward": {"tracking_sigma": 0.02},
                    "domain_rand": {"strength": 0.4},
                    "sensor_noise": {"strength": 0.4},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.24]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [-1.0, 1.0]],
                    },
                    "command_ranges": {
                        "standing_probability": 0.25,
                        "lin_vel_range": [-0.50, 0.50],
                        "ang_vel_range": [-10.0, 10.0],
                        "base_height_range": [0.20, 0.36],
                        "gimbal_yaw_vel_range": [-0.4, 0.4],

                    },
                    "tracking_gate": {
                        "height_full_error": 0.02,
                        "height_zero_error": 0.04,
                        "attitude_full_angle_deg": 0.6,
                        "attitude_zero_angle_deg": 2.0,
                        "floor": 0.05,
                    },
                    "reward_scales": {
                        "standing_drift": -3.0,
                        "joint_vel": -0.01,
                        "wheel_action_rate": -1.0,
                        "leg_action_rate": -0.15,
                        # "gimbal_yaw_ang": -10.0,
                        "gimbal_spin_track_lin_vel_yaw_frame": -15.0,
                        "gimbal_spin_heading_error":-20.0

                    },
                },
            },
            {
                "name": "full_range",
                "start_iteration": 5000,
                "targets": {
                    "domain_rand": {"strength": 1.0},
                    "sensor_noise": {"strength": 1.0},
                    "reset_ranges": {
                        "base_init_pos_range": [[0.0, 0.0], [0.0, 0.0], [0.22, 0.24]],
                        "base_init_rpy_offset_range_deg": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_lin_vel_range": [[-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
                        "base_init_ang_vel_range": [[0.0, 0.0], [0.0, 0.0], [-1.0, 1.0]],
# TODO 后续域随机话需要在这里增加云台角度
                    },
                    "command_ranges": {
                        "standing_probability": 0.25,
                        "lin_vel_range": [-0.75, 0.75],
                        "ang_vel_range": [-11.0, 11.0],
                        "base_height_range": [0.20, 0.36],
                        "gimbal_yaw_vel_range": [-0.4, 0.4],
                    },
                    "reward_scales": {},
                },
            },
        ],
    }
