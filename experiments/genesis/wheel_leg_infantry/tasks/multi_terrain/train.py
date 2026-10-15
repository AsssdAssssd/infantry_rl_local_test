"""python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.train"""

import argparse
from copy import deepcopy
from pathlib import Path

import genesis as gs

from ...core.train_config import get_train_cfg
from ...core.randomization_config import add_randomization_arguments, apply_randomization_arguments
from ...tools.run_utils import (add_resume_arguments, create_versioned_run_dir, load_runner_class,
                               load_run_configs, resolve_resume_plan, restore_training_state, save_run_artifacts)
from .runner import task_runner_class
from .config import get_cfgs
from .curriculum import initial_task_counts, resolve_tasks
from .env import MultiTerrainEnv


def main(argv=None):
    parser = argparse.ArgumentParser(description="Task-driven multi-terrain training; edit multi_terrain/config.py for per-task settings")
    parser.add_argument("-e", "--exp_name", default="multi_terrain_v3")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=3001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("--load-weights", type=Path, default=None)
    add_resume_arguments(parser)
    add_randomization_arguments(parser, evaluation=False)
    args = parser.parse_args(argv)
    if args.load_weights is not None and args.resume is not None:
        parser.error("--load-weights cannot be combined with --resume")
    if args.resume is None and (args.checkpoint is not None or args.resume_config != "saved"):
        parser.error("--checkpoint and --resume-config require --resume")
    if args.max_iterations < 1:
        parser.error("--max_iterations must be positive")
    if args.load_weights is not None and not args.load_weights.expanduser().is_file():
        parser.error(f"weights checkpoint does not exist: {args.load_weights}")
    cfgs = get_cfgs()
    configs = dict(zip(("env_cfg", "obs_cfg", "reward_cfg", "command_cfg", "curriculum_cfg"), cfgs))
    configs["train_cfg"] = get_train_cfg(args.exp_name)
    resume_plan = None
    if args.resume is not None:
        resume_plan = resolve_resume_plan(args.log_root, args.exp_name, args.resume, args.checkpoint)
        if args.resume_config == "saved":
            configs = deepcopy(load_run_configs(resume_plan.source_run_dir))
    if resume_plan is not None and args.resume_config == "saved":
        print(f"[train] config: saved in {resume_plan.source_run_dir}")
    else:
        print("[train] config: config.py")
    env_cfg, obs_cfg = configs["env_cfg"], configs["obs_cfg"]
    if "multi_terrain" not in env_cfg:
        parser.error("saved run is not multi_terrain; use --load-weights to transfer locomotion weights")
    apply_randomization_arguments(env_cfg, obs_cfg, args)
    try:
        tasks = resolve_tasks(env_cfg["multi_terrain"])
        counts = initial_task_counts(tasks, args.num_envs, configs["curriculum_cfg"])
    except ValueError as exc:
        parser.error(str(exc))
    for (name, task), count in zip(tasks.items(), counts):
        print(f"[train] {name}: {count}/{args.num_envs} envs ({count / args.num_envs:.1%}), "
              f"{len(task['resolved_stages'])} stages")
    run_dir = create_versioned_run_dir(args.log_root, args.exp_name)
    configs["train_cfg"]["run_name"] = f"{args.exp_name}/{run_dir.name}"
    arguments = vars(args).copy()
    if resume_plan is not None:
        arguments["resume_from"] = str(resume_plan.checkpoint_path.resolve())
    save_run_artifacts(run_dir, configs, arguments)
    print(f"[train] saving to {run_dir}")
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)
    env = MultiTerrainEnv(args.num_envs, env_cfg, obs_cfg, configs["reward_cfg"], configs["command_cfg"],
                          curriculum_cfg=configs["curriculum_cfg"],
                          steps_per_iteration=configs["train_cfg"]["num_steps_per_env"], show_viewer=args.vis)
    runner = task_runner_class(load_runner_class())(env, configs["train_cfg"], str(run_dir), device=gs.device)
    remaining = args.max_iterations
    if resume_plan is not None:
        remaining = restore_training_state(runner, env, resume_plan, args.max_iterations)
    elif args.load_weights is not None:
        runner.load(str(args.load_weights.expanduser().resolve()),
                    load_cfg=dict(actor=True, critic=True, optimizer=False, iteration=False, rnd=False),
                    strict=True, map_location=gs.device)
        runner.current_learning_iteration = 0
    runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)


if __name__ == "__main__":
    main()
