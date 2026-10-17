"""rotate checkpoint 离屏渲染评估，包含视频录制。"""

import argparse
import json
import csv
from copy import deepcopy
from pathlib import Path

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


DEFAULT_COMMAND_SEQUENCE = [
    {"step": 0, "lin_vel": 0.5, "ang_vel": 0.0, "base_height": 0.22, "gimbal_yaw_vel": 0.0, "translation_direction": 0.0},
    {"step": 500, "ang_vel": -3.0},
    {"step": 1167, "ang_vel": 0.0},
    {"step": 1333, "ang_vel": 3.0,"base_height": 0.30},
    {"step": 2000, "ang_vel": 0.0},
    {"step": 2167, "ang_vel": 6.0,"base_height": 0.25},
    {"step": 2833, "ang_vel": 0.0},
    {"step": 3000, "ang_vel": -6.0},
    {"step": 3667, "ang_vel": 0.0},
    {"step": 3833, "ang_vel": 10.0},
    {"step": 4500, "ang_vel": 0.0},
    {"step": 4667, "ang_vel": -10.0},
    {"step": 5333, "ang_vel": 0.0},
    {"step": 5500, "ang_vel": 13.0},
    {"step": 6167, "ang_vel": 0.0},
    {"step": 6333, "ang_vel": -13.0},
    {"step": 7000, "ang_vel": 0.0},
]


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
    parser.add_argument("--output-dir", type=str, default="eval_renders", help="output directory for videos and metrics")
    parser.add_argument("--total-steps", type=int, default=7000, help="total evaluation steps")
    parser.add_argument("--fps", type=int, default=None, help="video fps (default: derived from env dt for real-time playback)")
    args = parser.parse_args(argv)
    
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


class CommandSequenceExecutor:
    def __init__(self, sequence: list[dict]):
        if not sequence:
            raise ValueError("command sequence cannot be empty")
        
        first = sequence[0]
        required = {"lin_vel", "ang_vel", "base_height", "gimbal_yaw_vel", "translation_direction"}
        missing = required - set(first.keys())
        if missing:
            raise ValueError(f"first command must contain all fields, missing: {missing}")
        
        self.sequence = sorted(sequence, key=lambda x: x["step"])
        self.current = dict(first)
        self.current.pop("step", None)
        self.sequence_idx = 1
    
    def update(self, step: int):
        while self.sequence_idx < len(self.sequence) and self.sequence[self.sequence_idx]["step"] <= step:
            cmd = dict(self.sequence[self.sequence_idx])
            cmd.pop("step")
            self.current.update(cmd)
            self.sequence_idx += 1
    
    def get_command_dict(self) -> dict:
        return dict(self.current)


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

    print(f"[eval_render_video] run:        {run_dir}")
    print(f"[eval_render_video] checkpoint: {checkpoint_path}")
    print(f"[eval_render_video] output:     {args.output_dir}")
    print(
        "[eval_render_video] commands:   "
        f"vx={command_cfg['lin_vel_range']} wz={command_cfg['ang_vel_range']} "
        f"base_height={command_cfg['base_height_range']} gimbal={command_cfg['gimbal_yaw_vel_range']}"
    )

    gs.init(backend=gs.gpu, seed=args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 配置相机前初始化环境
    env = RotateEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg={"enabled": False, "stages": []},
        steps_per_iteration=train_cfg["num_steps_per_env"],
        show_viewer=False,
    )

    runner = OnPolicyRunner(env, train_cfg, str(run_dir), device=gs.device)
    runner.load(str(checkpoint_path), map_location=gs.device)
    policy = runner.get_inference_policy(device=gs.device)

    # 场景在 RotateEnv 构造时已 build，scene.add_camera 要求未 build，
    # 因此通过 visualizer 直接创建相机并手动 build。
    video_path = output_dir / f"eval_{run_dir.name}.mp4"
    fps = args.fps if args.fps is not None else max(1, round(1.0 / float(env.dt)))
    cam_offset = (1.2, -2.0, 1.0)
    cam_lookat_z = 0.4
    print(f"[eval_render_video] adding camera and starting recording... (fps={fps})")
    cam = env.scene.visualizer.add_camera(
        res=(1280, 720),
        pos=cam_offset,
        lookat=(0.0, 0.0, cam_lookat_z),
        up=(0.0, 0.0, 1.0),
        model="pinhole",
        fov=30,
        aperture=2.0,
        focus_dist=None,
        GUI=False,
        spp=256,
        denoise=None,
        near=0.1,
        far=20.0,
        env_idx=0,
        debug=True,
    )
    cam.build()
    cam.start_recording()
    print(f"[eval_render_video] recording to {video_path}")

    cmd_executor = CommandSequenceExecutor(DEFAULT_COMMAND_SEQUENCE)
    
    metrics = {
        "step": [],
        "lin_vel_cmd": [],
        "ang_vel_cmd": [],
        "base_height_cmd": [],
        "gimbal_yaw_vel_cmd": [],
        "translation_direction_cmd": [],
        "gimbal_world_yaw": [],
        "gimbal_yaw_error": [],
        "base_height_actual": [],
        "base_roll": [],
        "base_pitch": [],
        "estimated_forward_velocity": [],
        "wheel_forward_velocity": [],
        "true_forward_velocity": [],
    }

    obs_dict = env.reset()
    step = 0
    
    print("[eval_render_video] starting evaluation...")
    with torch.no_grad():
        while step < args.total_steps:
            cmd_executor.update(step)
            cmd = cmd_executor.get_command_dict()
            
            env.commands[0, 0] = cmd["lin_vel"]
            env.commands[0, 1] = cmd["ang_vel"]
            env.commands[0, 2] = cmd["base_height"]
            env.commands[0, 3] = cmd["gimbal_yaw_vel"]
            if env._has_direction_command():
                env.commands[0, 4] = cmd["translation_direction"]
            
            env._update_observations()
            obs_dict = env.get_observations()
            actions = policy(obs_dict)
            obs_dict, rews, dones, infos = env.step(actions)
            
            base_pos = env.base_pos[0].detach().cpu().tolist()
            cam.set_pose(
                pos=(base_pos[0] + cam_offset[0], base_pos[1] + cam_offset[1], base_pos[2] + cam_offset[2]),
                lookat=(base_pos[0], base_pos[1], base_pos[2] + cam_lookat_z),
            )
            cam.render()
            
            metrics["step"].append(step)
            metrics["lin_vel_cmd"].append(cmd["lin_vel"])
            metrics["ang_vel_cmd"].append(cmd["ang_vel"])
            metrics["base_height_cmd"].append(cmd["base_height"])
            metrics["gimbal_yaw_vel_cmd"].append(cmd["gimbal_yaw_vel"])
            metrics["translation_direction_cmd"].append(cmd["translation_direction"])
            
            gimbal_diag = env.get_rotate_diagnostics()
            metrics["gimbal_world_yaw"].append(gimbal_diag["gimbal_world_yaw"].item())
            metrics["gimbal_yaw_error"].append(gimbal_diag["gimbal_yaw_error"].item())
            
            metrics["base_height_actual"].append(env.base_height[0].item())
            
            roll, pitch, yaw = env.base_euler[0]
            metrics["base_roll"].append(roll.item())
            metrics["base_pitch"].append(pitch.item())
            
            vel_diag = env.get_velocity_estimator_diagnostics()
            metrics["estimated_forward_velocity"].append(vel_diag["estimated_forward_velocity"].item())
            metrics["wheel_forward_velocity"].append(vel_diag.get("wheel_forward_velocity", torch.tensor(0.0)).item())
            metrics["true_forward_velocity"].append(vel_diag["true_forward_velocity"].item())
            
            step += 1
            if step % 500 == 0:
                print(f"  step {step}/{args.total_steps}")
    
    if cam is not None:
        cam.stop_recording(save_to_filename=str(video_path), fps=fps)
        print(f"[eval_render_video] video saved: {video_path}")
    
    metrics_path = output_dir / f"eval_{run_dir.name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval_render_video] metrics JSON: {metrics_path}")
    
    csv_path = output_dir / f"eval_{run_dir.name}_metrics.csv"
    keys = list(metrics.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for i in range(len(metrics["step"])):
            row = {k: metrics[k][i] for k in keys}
            writer.writerow(row)
    print(f"[eval_render_video] metrics CSV: {csv_path}")
    
    cmd_seq_path = output_dir / f"eval_{run_dir.name}_commands.json"
    with open(cmd_seq_path, "w") as f:
        json.dump(DEFAULT_COMMAND_SEQUENCE, f, indent=2)
    print(f"[eval_render_video] command sequence: {cmd_seq_path}")
    
    print(f"[eval_render_video] done!")


if __name__ == "__main__":
    main()
