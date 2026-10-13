"""独立配置各任务的环境占比、几何、命令、奖励和成功率课程。"""

from ..locomotion.config_locomotion import get_cfgs as locomotion_cfgs
from ..locomotion.config_common import get_final_command_cfg
from ...terrains.registry import level_catalog

TASK_NAMES = ("locomotion", "stairs_down", "stairs_up", "slope_up", "slope_down", "rough", "square_wave", "platform_ridge")


def _get_curriculum_cfg():
    """按阶段配置随机化、传感器噪声与任务权重"""
    return {
        "enabled": True,
        "stages": [
            {
                "name": "clean",
                "start_iteration": 0,
                "targets": {
                    "domain_rand": {"strength": 0.0},
                    "sensor_noise": {"strength": 0.0},
                    "task_weights": {
                        "platform_ridge": 0.0,
                    },
                },
            },
            {
                "name": "randomized",
                "start_iteration": 1000,
                "targets": {
                    "domain_rand": {"strength": 0.4},
                    "sensor_noise": {"strength": 0.4},
                    "task_weights": {
                        "square_wave": 0.0,
                        "platform_ridge": 1.0,
                        "stairs_down": 0.5,
                        "slope_down": 0.5,
                    },
                },
            },
            {
                "name": "full",
                "start_iteration": 1200,
                "targets": {
                    "domain_rand": {"strength": 1.0},
                    "sensor_noise": {"strength": 1.0},
                },
            },
        ],
    }


def get_cfgs():
    env, obs, reward, commands, flat_course = locomotion_cfgs()
    commands = get_final_command_cfg(commands, flat_course)
    # 合并原平地课程的全局奖励参数；各 task 的可调配置独立列在下方。
    for stage in flat_course["stages"]:
        targets = stage["targets"]
        reward["reward_scales"].update(targets.get("reward_scales", {}))
        reward["tracking_gate"].update(targets.get("tracking_gate", {}))
        reward.update({"standing_" + k: v for k, v in targets.get("standing_reward", {}).items()})
        env.update(targets.get("termination_limits", {}))
    # 每个任务集中配置；地形几何和难度参数统一由 terrains 提供。
    # 所有地形保留线速度/偏航角速度跟踪，并加大机身倾斜惩罚。
    tasks = {}

    # 平地运动
    tasks["locomotion"] = dict(
        enabled=True,
        weight=2.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[-3.0, 3.0],
            ang_vel_range=[-5.0, 5.0],
            base_height_range=[0.2, 0.36],
            high_speed_ang_vel=dict(lin_vel_threshold=1.5, max_abs_ang_vel=2.0),
            standing_probability=0.3,
        ),
        rewards=dict(
            tracking_lin_vel=-2.0,
            tracking_ang_vel=-3.0,
            standing_drift=-3.0,
            gated_tracking_lin_vel=5.0,
            gated_tracking_ang_vel=7.0,
            base_balance=-20.0,
            leg_symmetry=-10.0,
            leg_angle_limits=-5.0,
            leg_symmetry_bonus=1.0,
            base_height=-10.0,
            height_gate=10.0,
            joint_vel=-0.02,
            wheel_action_rate=-2.0,
            leg_action_rate=-0.15,
            landing_base_oscillation=-0.3,
            landing_joint_vel=-0.01,
            wheel_airborne=-10.0,
            base_contact=-15.0,
            alive=5.0,
            death=-100.0,
        ),
        terrain=dict(preset="plane", tile_size=[80.0, 80.0]),
        termination=dict(
            termination_if_roll_greater_than=10.0,
            termination_if_pitch_greater_than=10.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=0.1,
            episode_length_s=10.0,
        ),
        resampling_time_s=3.0,
        # 难度参数读取 terrains 对应模块的 LEVELS；None 表示使用最高已定义等级。
        curriculum=dict(
            enabled=False,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.8,
            demote_threshold=0.3,
        ),
        # 平地按完整回合的跟踪误差统计成功率。
        tracking_success=dict(
            lin_vel_rmse=0.5,
            ang_vel_rmse=1.0,
            height_rmse=0.1,
        ),
    )

    # 下台阶
    tasks["stairs_down"] = dict(
        enabled=True,
        weight=1.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[1.0, 2.5],
            ang_vel_range=[0.0, 0.0],
            base_height_range=[0.20, 0.30],
            standing_probability=0.0,
        ),
        rewards=dict(
            tracking_lin_vel=-2.0,
            tracking_ang_vel=-3.0,
            gated_tracking_lin_vel=5.0,
            gated_tracking_ang_vel=7.0,
            task_progress=5.0,
            task_success=20.0,
            task_lateral=-1.0,
            task_heading=-0.5,
            base_balance=-20.0,
            leg_angle_limits=-5.0,
            joint_vel=-0.001,
            wheel_action_rate=-0.05,
            leg_action_rate=-0.05,
            base_contact=-10.0,
            stairs_down_landing_stability=-15.0,
            death=-10.0,
        ),
        resampling_time_s=10.0,
        terrain=dict(preset="stairs", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.8),
        termination=dict(
            episode_length_s=10.0,
            termination_if_roll_greater_than=10.0,
            termination_if_pitch_greater_than=10.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=0.5,
            corridor_half_width=1.0,
            stall_timeout_s=1.0,
            progress_epsilon=0.1,
        ),
        success=dict(
            max_roll_deg=1.0,
            max_pitch_deg=2.0,
            wheel_clearance_m=0.1,
            wheel_surface_tolerance_m=0.08,
        ),
        # 难度参数读取 terrains 对应模块的 LEVELS；None 表示使用最高已定义等级。
        curriculum=dict(
            enabled=True,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.95,
            demote_threshold=0.2,
        ),
        # 首次落到台阶下方后，短时抑制竖直振荡及 roll/pitch 角速度。
        landing_stability=dict(duration_s=0.3, ang_vel_weight=0.25),
    )

    # 上台阶
    tasks["stairs_up"] = dict(
        enabled=True,
        weight=1.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[1.0, 2.5],
            ang_vel_range=[0.0, 0.0],
            base_height_range=[0.31, 0.31],
            standing_probability=0.0,
        ),
        rewards=dict(
            task_progress=10.0,
            task_success=20.0,
            stairs_up_support=10.0,
            stairs_up_retract=20.0,
            stairs_up_takeoff=10.0,  # 双轮首次离开台下地面，一回合只奖励一次。
            stairs_up_landing_distance=10.0,  # 首轮触台距离奖励，单位为奖励/m，仅首次触台发放。
            stairs_up_wheel_wall_force=-0.2,  # 轮子撞立面的法向力总和（N），按 dt 积分。
            stairs_up_approach_lift=10.0,
            stairs_up_backswing=30.0,
            task_lateral=-2.0,
            task_heading=-0.5,
            base_balance=-20.0,
            joint_vel=-0.001,
            wheel_action_rate=-0.05,
            leg_action_rate=-0.02,
            death=-10.0,
        ),
        resampling_time_s=3.0,
        terrain=dict(preset="stairs_up", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.8),
        termination=dict(
            episode_length_s=4.0,
            termination_if_roll_greater_than=70.0,
            termination_if_pitch_greater_than=70.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=10.0,
            corridor_half_width=1.0,
            stall_timeout_s=2.0,
            progress_epsilon=0.01,
        ),
        success=dict(
            max_roll_deg=10.0,
            max_pitch_deg=10.0,
            wheel_clearance_m=0.15,
            wheel_surface_tolerance_m=0.08,
        ),
        ascent=dict(
            approach_distance_m=0.35,
            approach_clearance_m=0.01,  # 底盘碰撞体下底面高出台阶顶面的余量。
            backswing_angle_deg=20.0,  # 正腿角为向后摆，低于当前 30° 摆角限幅。
            support_duration_s=0.10,
            support_force_n=5.0,
            support_normal_z=0.9,
            support_surface_tolerance_m=0.015,
            support_margin_m=0.1,
            retract_shortening_m=0.05,  # 首次底盘触台后，每腿新增缩短量的奖励上限。
            wheel_edge_margin_m=0.01,
            wheel_bottom_tolerance_m=0.005,
            fall_height_tolerance_m=0.03,  # 双轮落台后，任一轮底低于台面此距离即失败。
            wall_normal_z_max=0.3,  # 接触法向近似水平，区分立面和台面/地面。
            wall_surface_tolerance_m=0.015,
            success_duration_s=0.20,
            standing_height_m=0.20,
            max_vertical_speed_m_s=0.3,
            max_roll_pitch_speed_rad_s=1.5,
        ),
        # 难度参数读取 terrains 对应模块的 LEVELS；None 表示使用最高已定义等级。
        curriculum=dict(
            enabled=True,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.95,
            demote_threshold=0.2,
        ),
    )

    # 上坡
    tasks["slope_up"] = dict(
        enabled=True,
        weight=1.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[1.0, 2.5],
            ang_vel_range=[0.0, 0.0],
            base_height_range=[0.20, 0.30],
            standing_probability=0.0,
        ),
        rewards=dict(
            tracking_lin_vel=-2.0,
            tracking_ang_vel=-3.0,
            gated_tracking_lin_vel=5.0,
            gated_tracking_ang_vel=7.0,
            task_progress=5.0,
            task_success=20.0,
            task_lateral=-2.0,
            task_heading=-1.0,
            base_balance=-20.0,
            leg_angle_limits=-5.0,
            joint_vel=-0.001,
            wheel_action_rate=-0.05,
            leg_action_rate=-0.02,
            leg_symmetry=-5.0,
            leg_symmetry_bonus=0.2,
            base_contact=-10.0,
            death=-10.0,
        ),
        resampling_time_s=10.0,
        terrain=dict(preset="slope_up", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.8),
        termination=dict(
            episode_length_s=10.0,
            termination_if_roll_greater_than=10.0,
            termination_if_pitch_greater_than=10.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=0.5,
            corridor_half_width=1.0,
            stall_timeout_s=1.0,
            progress_epsilon=0.1,
        ),
        success=dict(
            max_roll_deg=10.0,
            max_pitch_deg=10.0,
            wheel_clearance_m=0.15,
            wheel_surface_tolerance_m=0.08,
        ),
        # 难度参数读取 terrains 对应模块的 LEVELS；None 表示使用最高已定义等级。
        curriculum=dict(
            enabled=True,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.95,
            demote_threshold=0.2,
        ),
    )

    # 下坡
    tasks["slope_down"] = dict(
        enabled=True,
        weight=1.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[1.0, 2.5],
            ang_vel_range=[0.0, 0.0],
            base_height_range=[0.20, 0.30],
            standing_probability=0.0,
        ),
        rewards=dict(
            tracking_lin_vel=-2.0,
            tracking_ang_vel=-3.0,
            gated_tracking_lin_vel=5.0,
            gated_tracking_ang_vel=7.0,
            task_progress=5.0,
            task_success=20.0,
            task_lateral=-2.0,
            task_heading=-0.5,
            base_balance=-20.0,
            leg_angle_limits=-5.0,
            joint_vel=-0.001,
            wheel_action_rate=-0.05,
            leg_action_rate=-0.02,
            base_contact=-10.0,
            death=-10.0,
        ),
        resampling_time_s=10.0,
        terrain=dict(preset="slope_down", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.8),
        termination=dict(
            episode_length_s=10.0,
            termination_if_roll_greater_than=10.0,
            termination_if_pitch_greater_than=10.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=0.5,
            corridor_half_width=1.0,
            stall_timeout_s=1.0,
            progress_epsilon=0.1,
        ),
        success=dict(
            max_roll_deg=10.0,
            max_pitch_deg=10.0,
            wheel_clearance_m=0.15,
            wheel_surface_tolerance_m=0.08,
        ),
        # 难度参数读取 terrains 对应模块的 LEVELS；None 表示使用最高已定义等级。
        curriculum=dict(
            enabled=True,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.95,
            demote_threshold=0.2,
        ),
    )

    # 起伏路面
    tasks["rough"] = dict(
        enabled=False,
        weight=1.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[1.0, 2.5],
            ang_vel_range=[0.0, 0.0],
            base_height_range=[0.22, 0.36],
            standing_probability=0.0,
        ),
        rewards=dict(
            tracking_lin_vel=-2.0,
            tracking_ang_vel=-3.0,
            gated_tracking_lin_vel=5.0,
            gated_tracking_ang_vel=7.0,
            task_progress=5.0,
            task_success=20.0,
            task_lateral=-2.0,
            task_heading=-0.5,
            base_balance=-20.0,
            leg_angle_limits=-5.0,
            joint_vel=-0.001,
            wheel_action_rate=-0.05,
            leg_action_rate=-0.02,
            base_contact=-3.0,
            death=-10.0,
        ),
        resampling_time_s=10.0,
        terrain=dict(preset="random_rough", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.8, distance=3.0),
        termination=dict(
            episode_length_s=10.0,
            termination_if_roll_greater_than=10.0,
            termination_if_pitch_greater_than=10.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=0.5,
            corridor_half_width=1.0,
            stall_timeout_s=1.0,
            progress_epsilon=0.1,
        ),
        success=dict(
            max_roll_deg=10.0,
            max_pitch_deg=10.0,
            wheel_clearance_m=0.15,
            wheel_surface_tolerance_m=0.08,
        ),
        # 难度参数读取 terrains 对应模块的 LEVELS；None 表示使用最高已定义等级。
        curriculum=dict(
            enabled=True,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.95,
            demote_threshold=0.2,
        ),
    )

    tasks["square_wave"] = dict(
        enabled=True,
        weight=1.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[0.5, 2.5],
            ang_vel_range=[0.0, 0.0],
            base_height_range=[0.20, 0.30],
            standing_probability=0.0,
        ),
        rewards=dict(
            tracking_lin_vel=-2.0,
            tracking_ang_vel=-3.0,
            gated_tracking_lin_vel=5.0,
            gated_tracking_ang_vel=7.0,
            task_progress=5.0,
            task_success=20.0,
            task_lateral=-2.0,
            task_heading=-0.5,
            base_balance=-20.0,
            leg_angle_limits=-5.0,
            joint_vel=-0.001,
            wheel_action_rate=-0.05,
            leg_action_rate=-0.02,
            base_contact=-3.0,
            death=-10.0,
        ),
        resampling_time_s=10.0,
        terrain=dict(preset="square_wave", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.8),
        termination=dict(
            episode_length_s=10.0,
            termination_if_roll_greater_than=10.0,
            termination_if_pitch_greater_than=10.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=0.5,
            corridor_half_width=1.0,
            stall_timeout_s=1.0,
            progress_epsilon=0.1,
        ),
        success=dict(
            max_roll_deg=10.0,
            max_pitch_deg=10.0,
            wheel_clearance_m=0.15,
            wheel_surface_tolerance_m=0.08,
        ),
        # 难度参数读取 terrains 对应模块的 LEVELS；None 表示使用最高已定义等级。
        curriculum=dict(
            enabled=True,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.95,
            demote_threshold=0.2,
        ),
    )
    # 后期凸台任务：从第二平台朝 -x 越过窄凸台，再经第一平台下到地面。
    # enabled 控制是否构建；全局课程 task_weights 控制何时分配训练环境。
    tasks["platform_ridge"] = dict(
        enabled=True,
        weight=1.0,
        commands=dict(
            num_commands=3,
            lin_vel_range=[0.5, 2.5],
            ang_vel_range=[0.0, 0.0],
            base_height_range=[0.20, 0.30],
            standing_probability=0.0,
        ),
        rewards=dict(
            tracking_lin_vel=-2.0,
            tracking_ang_vel=-3.0,
            gated_tracking_lin_vel=5.0,
            gated_tracking_ang_vel=7.0,
            task_progress=5.0,
            task_success=20.0,
            task_lateral=-2.0,
            task_heading=-0.5,
            base_balance=-2.0,
            joint_vel=-0.001,
            wheel_action_rate=-0.05,
            leg_action_rate=-0.02,
            base_contact=-1.0,
            death=-10.0,
        ),
        resampling_time_s=10.0,
        terrain=dict(preset="platform_ridge", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.8),
        termination=dict(
            episode_length_s=15.0,
            termination_if_roll_greater_than=45.0,
            termination_if_pitch_greater_than=45.0,
            tilt_termination_duration_s=0.3,
            base_contact_termination_duration_s=0.5,
            corridor_half_width=1.0,
            stall_timeout_s=5.0,
            progress_epsilon=0.01,
        ),
        success=dict(
            max_roll_deg=10.0,
            max_pitch_deg=10.0,
            wheel_clearance_m=0.15,
            wheel_surface_tolerance_m=0.08,
        ),
        curriculum=dict(
            enabled=True,
            difficulty=0,
            min_difficulty=0,
            max_difficulty=None,
            window_episodes=100,
            promote_threshold=0.95,
            demote_threshold=0.2,
        ),
    )
    env["multi_terrain"] = dict(tasks=tasks, level_catalog=level_catalog())
    env.update(episode_length_s=25.0, resampling_time_s=0.01, base_init_pos_range=[[-0.03, 0.03], [-0.03, 0.03], [0.22, 0.22]])
    # 仅注册奖励函数，实际系数按各环境本回合所处的任务/课程阶段查表。
    reward["reward_scales"] = dict.fromkeys(set().union(*(task["rewards"] for task in tasks.values())), 0.0)
    return env, obs, reward, commands, _get_curriculum_cfg()
