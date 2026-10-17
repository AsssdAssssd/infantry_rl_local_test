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
    create_versioned_run_dir,
    load_runner_class,
    load_run_configs,
    resolve_resume_plan,
    restore_training_state,
    save_run_artifacts,
)
from .config import get_cfgs
from .env import RotateEnv
from .warm_start import validate_source, warm_start_actor
from ...terrains.registry import level_catalog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-e", "--exp_name", type=str, default="rotation")
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--terrain", choices=TERRAIN_PRESETS, default=None)
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument("--load-weights", type=Path, default=None)
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

    env_cfg.setdefault("terrain", default_terrain_cfg())
    env_cfg["terrain"].setdefault("level_catalog", level_catalog())
    if args.terrain is not None:
        targets = [env_cfg["terrain"]] + [
            stage["targets"]["terrain"] for stage in curriculum_cfg.get("stages", [])
            if "terrain" in stage.get("targets", {})
        ]
        for target in targets:
            target["preset"] = args.terrain
            for key in ("mixture", "adaptive", "difficulty"):
                target.pop(key, None)
    terrain_course_configs(env_cfg.get("terrain"), curriculum_cfg)
    apply_randomization_arguments(env_cfg, obs_cfg, args)
    remaining_iterations = args.max_iterations

    run_dir = create_versioned_run_dir(args.log_root, args.exp_name)
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

    runner = OnPolicyRunner(env, deepcopy(train_cfg), str(run_dir), device=gs.device)
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
            load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
            strict=True,
            map_location=gs.device,
        )
        runner.current_learning_iteration = 0
        print(f"[rotate] loaded actor/critic weights from: {args.load_weights}")
    runner.learn(num_learning_iterations=remaining_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
