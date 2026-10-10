"""rotate checkpoint 的交互式可视化评估。

结构与 locomotion/eval.py 一致，环境换成 RotateEnv、键盘换成带云台的版本。
"""

import argparse
from copy import deepcopy

import genesis as gs
import torch

from ...terrains import TERRAIN_PRESETS, default_terrain_cfg, resolve_terrain_cfg
from ...core.randomization_config import add_randomization_arguments, apply_randomization_arguments
from ...tools.tof_viewer import prepare_tof_for_viewer
from ...tools.run_utils import (
    load_run_configs,
    load_runner_class,
    resolve_checkpoint,
    resolve_run_dir,
)
from ..locomotion.config_common import get_final_command_cfg
from .env import RotateEnv
from .interactive_viewer import KeyboardCommand, format_tensor


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    add_randomization_arguments(parser, evaluation=True)
    parser.add_argument("-e", "--exp_name", type=str, default="rotation")
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--terrain", choices=TERRAIN_PRESETS, default=None)
    parser.add_argument("--difficulty", type=int, default=None)
    parser.add_argument("--terrain-size", type=float, nargs=2, metavar=("LENGTH", "WIDTH"), default=None)
    parser.add_argument("--print-interval", type=int, default=50)
    args = parser.parse_args(argv)

    if args.print_interval < 0:
        parser.error("--print-interval cannot be negative")
    if args.terrain_size is not None and any(value <= 0.0 for value in args.terrain_size):
        parser.error("--terrain-size values must be positive")
    return args


def _apply_terrain_overrides(env_cfg, args):
    preset = args.terrain or env_cfg.get("terrain", {}).get("preset", "plane")
    if preset == "flat":
        preset = "plane"
    env_cfg["terrain"] = default_terrain_cfg(preset)
    if args.terrain_size is not None:
        env_cfg["terrain"]["tile_size"] = list(args.terrain_size)
    if args.difficulty is not None:
        env_cfg["terrain"]["difficulty"] = args.difficulty
        env_cfg["terrain"] = resolve_terrain_cfg(env_cfg["terrain"])


def print_evaluation_diagnostics(env, step):
    print(
        f"\n[eval step {step}] scaled observations "
        f"(policy={env.obs_buf.shape[-1]}, critic={env.critic_obs_buf.shape[-1]})"
    )
    for name, value in env.get_observation_components().items():
        print(f"  obs.{name:<20} {format_tensor(value)}")

    velocity = env.get_velocity_estimator_diagnostics()
    print(f"  velocity source          {velocity['source']}")
    print(f"  velocity true            {velocity['true_forward_velocity'].item():+.4f} m/s")
    print(f"  velocity estimated       {velocity['estimated_forward_velocity'].item():+.4f} m/s")
    print(f"  IMU specific force       {format_tensor(velocity['imu_specific_force'])} m/s^2")

    gimbal = env.get_rotate_diagnostics()
    print(
        "  gimbal                   "
        f"joint={gimbal['gimbal_joint'].item():+.4f} rad  "
        f"world={gimbal['gimbal_world_yaw'].item():+.4f} rad  "
        f"ref={gimbal['gimbal_yaw_ref'].item():+.4f} rad  "
        f"err={gimbal['gimbal_yaw_error'].item():+.4f} rad"
    )

    pd = env.get_pd_diagnostics()
    print("  joint names             " + ", ".join(pd["joint_names"]))
    print(f"  joint target position   {format_tensor(pd['joint_target_pos'])}")
    print(f"  joint position          {format_tensor(pd['joint_pos'])}")
    print("  wheel names             " + ", ".join(pd["wheel_names"]))
    print(f"  wheel target velocity   {format_tensor(pd['wheel_target_vel'])}", flush=True)


def main():
    args = _parse_args()
    OnPolicyRunner = load_runner_class()

    run_dir = resolve_run_dir(args.log_root, args.exp_name, args.version, require_checkpoint=True)
    checkpoint_path = resolve_checkpoint(run_dir, args.ckpt)
    configs = load_run_configs(run_dir)
    env_cfg = deepcopy(configs["env_cfg"])
    _apply_terrain_overrides(env_cfg, args)
    obs_cfg = deepcopy(configs["obs_cfg"])
    apply_randomization_arguments(env_cfg, obs_cfg, args)
    prepare_tof_for_viewer(obs_cfg)
    reward_cfg = deepcopy(configs["reward_cfg"])
    command_cfg = get_final_command_cfg(configs["command_cfg"], configs["curriculum_cfg"])
    train_cfg = configs["train_cfg"]
    reward_cfg["reward_scales"] = {}

    print(f"[eval] run:        {run_dir}")
    print(f"[eval] checkpoint: {checkpoint_path}")
    print(
        "[eval] commands:   "
        f"vx={command_cfg['lin_vel_range']} wz={command_cfg['ang_vel_range']} "
        f"base_height={command_cfg['base_height_range']} gimbal={command_cfg['gimbal_yaw_vel_range']}"
    )

    gs.init(backend=gs.gpu, seed=args.seed)

    env = RotateEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg={"enabled": False, "stages": []},
        steps_per_iteration=train_cfg["num_steps_per_env"],
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, str(run_dir), device=gs.device)
    runner.load(str(checkpoint_path), map_location=gs.device)
    policy = runner.get_inference_policy(device=gs.device)

    obs_dict = env.reset()
    env.focus_viewer()
    keyboard = KeyboardCommand(env)
    keyboard.update_caption()
    print("[eval] keys: UP/DOWN=vx  LEFT/RIGHT=wz  PAGEUP/DOWN=height  [ / ]=gimbal rate  SPACE=stop")
    print_evaluation_diagnostics(env, step=0)
    step = 0
    with torch.no_grad():
        while env.scene.viewer.is_alive():
            keyboard.write_to_env()
            env._update_observations()
            obs_dict = env.get_observations()
            actions = policy(obs_dict)
            obs_dict, rews, dones, infos = env.step(actions)
            keyboard.update_caption()
            step += 1
            if args.print_interval and step % args.print_interval == 0:
                print_evaluation_diagnostics(env, step)


if __name__ == "__main__":
    main()
