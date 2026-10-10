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

GIMBAL_JOINT = "gimbal_yaw_joint"


def get_cfgs(locomotion_cfgs=None):
    """返回 env / obs / reward / command / curriculum 五组配置。"""
    if locomotion_cfgs is None:
        locomotion_cfgs = get_locomotion_cfgs()

    env_cfg = deepcopy(locomotion_cfgs[0])
    obs_cfg = deepcopy(locomotion_cfgs[1])
    reward_cfg = deepcopy(locomotion_cfgs[2])
    command_cfg = deepcopy(locomotion_cfgs[3])
    curriculum_cfg = deepcopy(locomotion_cfgs[4])

    # ---------- 机器人：换成带 yaw 关节的模型 ----------
    env_cfg["num_actions"] = 7
    env_cfg["num_joints"] = 5
    env_cfg["robot_mjcf"] = "assets/robot/wheelbipeV14_2_yaw_joint/mjcf/wheelbipeV14_2.xml"
    env_cfg["gimbal_joint_names"] = [GIMBAL_JOINT] #可惜这个模型没有pitch倒是
    if GIMBAL_JOINT not in env_cfg["joint_names"]:
        env_cfg["joint_names"] = [*env_cfg["joint_names"], GIMBAL_JOINT]
    env_cfg["default_joint_pos"][GIMBAL_JOINT] = 0.0

    # actor 输出的云台“世界 yaw 目标”的物理量程：action(-1,1) * scale
    env_cfg["gimbal_joint_ang_scale"] = math.pi
    env_cfg["clip_gimbal_joint_action"]= 1.0  #不过其实被joint一起clip了；
    # ---------- 观测缩放 ----------
    obs_cfg["obs_scales"]["gimbal_yaw_angle"] = 1.0 / math.pi # TODO 后续如果变化的比较大，可以换成2唯，不会跳变
    obs_cfg["obs_scales"]["gimbal_yaw_vel"] = 1.0 / math.pi

    # ---------- 命令：追加云台速率 ----------
    command_cfg["num_commands"] = 4
    command_cfg["gimbal_yaw_vel_range"] = [-1.0, 1.0] #TODO 这里其他的参数还是原来的，但小陀螺差异比较大，后面记得显示更改

    # ---------- 奖励 ----------
    reward_cfg["reward_scales"]["gimbal_yaw_ang"] = -10.0

    # ---------- 课程：每阶段补上云台命令范围与奖励权重；默认锁 0 ----------
    #TODO 我认为还是覆盖一个比较好，课程差异还是比较大的
    for stage in curriculum_cfg["stages"]:
        targets = stage.setdefault("targets", {})
        targets.setdefault("command_ranges", {}).setdefault("gimbal_yaw_vel_range", [0.0, 0.0])
        targets.setdefault("reward_scales", {}).setdefault("gimbal_yaw_ang", -10.0)
    # 最后一个阶段放开云台速率，让策略学习“跟着 ref 转”
    curriculum_cfg["stages"][-1]["targets"]["command_ranges"]["gimbal_yaw_vel_range"] = [-1.0, 1.0]

    return env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg
