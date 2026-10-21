"""rotate（旋转平移）任务 PPO 入口。

结构与 locomotion/train.py 一致，只是把环境换成 RotateEnv、配置换成 rotation/config.py。
"""

import argparse
from copy import deepcopy
from pathlib import Path

import genesis as gs

from ...core.train_config import get_train_cfg
from ...terrains import TERRAIN_PRESETS, default_terrain_cfg
from ...terrains.curriculum import terrain_course_configs
from ...core.randomization_config import add_randomization_arguments, apply_randomization_arguments
from ...tools.run_utils import (
    add_resume_arguments,
    create_or_broadcast_run_dir,
    init_distributed,
    load_runner_class,
    load_run_configs,
    resolve_resume_plan,
    restore_training_state,
    save_run_artifacts,
    skip_duplicate_process_group_init,
)
from .config import get_cfgs
from .env import RotateEnv
from .warm_start import validate_source, warm_start_actor
from ...terrains.registry import level_catalog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-e", "--exp_name", type=str, default="rotation")
    parser.add_argument("-B", "--num_envs", type=int, default=8196)
    parser.add_argument("--max_iterations", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--terrain", choices=TERRAIN_PRESETS, default=None)
    parser.add_argument("--difficulty", type=int, default=None,
                        help="terrain level for --terrain (see terrains/<preset>.py LEVELS)")
    parser.add_argument("--terrain-curriculum", choices=("slope_up", "slope_down", "tilted_plane"), default=None,
                        help="rotation-local slope curriculum; keeps the implementation inside tasks/rotation")
    parser.add_argument("--terrain-schedule", type=str, default=None,
                        help="start:difficulty,... for --terrain-curriculum (default: 7000:0,10000:2,13500:4)")
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument("--load-weights", type=Path, default=None)
    parser.add_argument("--load-actor-only", action="store_true",
                        help="with --load-weights, skip critic when its observation dimension changed")
    parser.add_argument("--warm-start", type=Path, default=None,
                        help="locomotion model_*.pt; cfgs.pkl must be in the same directory; actor only")
    add_randomization_arguments(parser, evaluation=False)
    add_resume_arguments(parser)
    args = parser.parse_args()

    if args.warm_start is not None:
        if args.resume is not None or args.load_weights is not None:
            parser.error("--warm-start cannot be combined with --resume or --load-weights")
        args.warm_start = args.warm_start.expanduser().resolve()
        if not args.warm_start.is_file():
            parser.error(f"Warm-start checkpoint does not exist: {args.warm_start}")
    if args.load_weights is not None and args.resume is not None:
        parser.error("--load-weights cannot be combined with --resume")
    if args.resume is None:
        if args.checkpoint is not None:
            parser.error("--checkpoint requires --resume")
        if args.resume_config != "saved":
            parser.error("--resume-config requires --resume")
    if args.load_weights is not None:
        args.load_weights = args.load_weights.expanduser().resolve()
        if not args.load_weights.is_file():
            parser.error(f"Weights checkpoint does not exist: {args.load_weights}")
    if args.terrain is not None and args.terrain_curriculum is not None:
        parser.error("--terrain and --terrain-curriculum cannot be combined")
    if args.terrain_schedule is not None and args.terrain_curriculum is None:
        parser.error("--terrain-schedule requires --terrain-curriculum")
    if args.difficulty is not None and args.difficulty < 0:
        parser.error("--difficulty must be non-negative")

    dist_info = init_distributed()
    if dist_info["is_distributed"]:
        print(f"[rotate] distributed training: rank {dist_info['global_rank']}/{dist_info['world_size']} on {dist_info['device']}")
    OnPolicyRunner = load_runner_class()
    resume_plan = None
    if args.resume is None or args.resume_config == "current":
        env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg = get_cfgs()
        train_cfg = get_train_cfg(args.exp_name)
        train_cfg["algorithm"]["learning_rate"] = 2e-4 #TODO
        train_cfg["algorithm"]["schedule"] = "fixed"

    if args.resume is not None:
        resume_plan = resolve_resume_plan(args.log_root, args.exp_name, args.resume, args.checkpoint)
        if args.resume_config == "saved":
            saved_configs = load_run_configs(resume_plan.source_run_dir)
            env_cfg = deepcopy(saved_configs["env_cfg"])
            obs_cfg = deepcopy(saved_configs["obs_cfg"])
            reward_cfg = deepcopy(saved_configs["reward_cfg"])
            command_cfg = deepcopy(saved_configs["command_cfg"])
            curriculum_cfg = deepcopy(saved_configs["curriculum_cfg"])
            train_cfg = deepcopy(saved_configs["train_cfg"])

    # 单地形参数优先：显式指定 --terrain / --difficulty / --terrain-curriculum 时关闭配置里的混合地形。
    if args.terrain is not None or args.difficulty is not None or args.terrain_curriculum is not None:
        env_cfg.pop("terrain_weights", None)
        env_cfg.pop("terrain_variants", None)
        for stage in curriculum_cfg.get("stages", []):
            targets = stage.get("targets", {})
            targets.pop("terrain_weights", None)
            targets.pop("terrain_variants", None)

    if args.terrain_curriculum is not None:
        from .terrain_curriculum import (
            DEFAULT_TERRAIN_SCHEDULES,
            merge_curriculum_stages,
            parse_schedule,
            slope_schedule_stages,
        )

        schedule = DEFAULT_TERRAIN_SCHEDULES[args.terrain_curriculum] if args.terrain_schedule is None else parse_schedule(args.terrain_schedule)
        curriculum_cfg = merge_curriculum_stages(
            curriculum_cfg, slope_schedule_stages(args.terrain_curriculum, schedule)
        )

    env_cfg.setdefault("terrain", default_terrain_cfg())
    env_cfg["terrain"].setdefault("level_catalog", level_catalog())
    if args.terrain is not None or args.difficulty is not None:
        targets = [env_cfg["terrain"]] + [
            stage["targets"]["terrain"] for stage in curriculum_cfg.get("stages", [])
            if "terrain" in stage.get("targets", {})
        ]
        for target in targets:
            if args.terrain is not None:
                target["preset"] = args.terrain
                for key in ("mixture", "adaptive"):
                    target.pop(key, None)
            if args.difficulty is not None:
                target["difficulty"] = args.difficulty
            elif args.terrain is not None:
                target.pop("difficulty", None)
    terrain_course_configs(env_cfg.get("terrain"), curriculum_cfg)
    apply_randomization_arguments(env_cfg, obs_cfg, args)
    remaining_iterations = args.max_iterations

    run_dir = create_or_broadcast_run_dir(args.log_root, args.exp_name, dist_info)
    train_cfg["run_name"] = f"{args.exp_name}/{run_dir.name}"
    print(f"[rotate] saving this run to: {run_dir}")

    configs = {
        "env_cfg": env_cfg,
        "obs_cfg": obs_cfg,
        "reward_cfg": reward_cfg,
        "command_cfg": command_cfg,
        "curriculum_cfg": curriculum_cfg,
        "train_cfg": train_cfg,
    }
    source_configs = None
    if args.warm_start is not None:
        source_configs = load_run_configs(args.warm_start.parent)
        validate_source(source_configs, configs)
        configs["source_locomotion"] = {
            "run_dir": str(args.warm_start.parent), "checkpoint": str(args.warm_start),
        }
    run_arguments = vars(args).copy()
    if resume_plan is not None:
        run_arguments["resume_from"] = str(resume_plan.checkpoint_path.resolve())
    if args.load_weights is not None:
        run_arguments["weights_from"] = str(args.load_weights)
    if not dist_info["is_distributed"] or dist_info["global_rank"] == 0:
        save_run_artifacts(run_dir, configs, run_arguments)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env = RotateEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg=curriculum_cfg,
        steps_per_iteration=train_cfg["num_steps_per_env"],
        show_viewer=args.vis,
    )

    with skip_duplicate_process_group_init():
        runner = OnPolicyRunner(env, deepcopy(train_cfg), str(run_dir), device=dist_info["device"])
    if resume_plan is not None:
        print(f"[rotate] resuming from: {resume_plan.checkpoint_path}")
        remaining_iterations = restore_training_state(runner, env, resume_plan, args.max_iterations)
        print(f"[rotate] continuing at iteration {resume_plan.next_iteration}; {remaining_iterations} iterations remain")
    elif args.warm_start is not None:
        report = warm_start_actor(runner.alg.get_policy(), args.warm_start,
                                  source_configs, configs, env.obs_components)
        configs["actor_warm_start"] = report.to_dict()
        save_run_artifacts(run_dir, configs, run_arguments)
        print(f"[rotate] actor warm-start: {args.warm_start}; {report}")
        print("[rotate] critic and optimizer start fresh; command frame transfer is learned by PPO")
    elif args.load_weights is not None:
        runner.load(
            str(args.load_weights),
            load_cfg={"actor": True, "critic": not args.load_actor_only, "optimizer": False, "iteration": False, "rnd": False},
            strict=True,
            map_location=gs.device,
        )
        runner.current_learning_iteration = 0
        print(f"[rotate] loaded actor weights from: {args.load_weights}" + (" (critic fresh)" if args.load_actor_only else " (critic loaded)"))
    runner.learn(num_learning_iterations=remaining_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
