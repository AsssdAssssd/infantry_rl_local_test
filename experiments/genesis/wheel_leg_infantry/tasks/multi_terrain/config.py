"""独立配置各任务的环境占比、几何、命令、奖励和成功率课程。"""

from ..locomotion.config_locomotion import get_cfgs as locomotion_cfgs
from ..locomotion.config_common import get_final_command_cfg
from ...terrains.registry import level_catalog

TASK_NAMES = ("locomotion", "stairs_down", "stairs_up", "slope_up", "slope_down", "rough", "square_wave")


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
            joint_vel=-0.01,
            wheel_action_rate=-1.5,
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
            base_contact=-1.0,
            death=-10.0,
        ),
        resampling_time_s=10.0,
        terrain=dict(preset="stairs_up", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.8),
        termination=dict(
            episode_length_s=10.0,
            termination_if_roll_greater_than=10.0,
            termination_if_pitch_greater_than=10.0,
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

    # 方波路面
    tasks["square_wave"] = dict(
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
            base_contact=-3.0,
            death=-10.0,
        ),
        resampling_time_s=10.0,
        terrain=dict(preset="square_wave", tile_size=[16.0, 6.0]),
        traversal=dict(landing_length=0.2),
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
    env["multi_terrain"] = dict(tasks=tasks, level_catalog=level_catalog())
    env.update(episode_length_s=25.0, resampling_time_s=0.01, base_init_pos_range=[[-0.03, 0.03], [-0.03, 0.03], [0.22, 0.22]])
    # 仅注册奖励函数，实际系数按各环境本回合所处的任务/课程阶段查表。
    reward["reward_scales"] = dict.fromkeys(set().union(*(task["rewards"] for task in tasks.values())), 0.0)
    curriculum = dict(
        enabled=True,
        stages=[
            dict(name="clean", start_iteration=0, targets=dict(domain_rand=dict(strength=0.0), sensor_noise=dict(strength=0.0))),
            dict(name="randomized", start_iteration=5000, targets=dict(domain_rand=dict(strength=0.4), sensor_noise=dict(strength=0.4))),
            dict(name="full", start_iteration=10000, targets=dict(domain_rand=dict(strength=1.0), sensor_noise=dict(strength=1.0))),
        ],
    )
    return env, obs, reward, commands, curriculum
