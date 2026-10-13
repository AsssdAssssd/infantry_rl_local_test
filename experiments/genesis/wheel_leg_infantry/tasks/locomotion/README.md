# Locomotion 训练

本目录只维护经典行走训练：速度和高度跟踪、姿态保持、静止站立及随机化课程。
多地形通关任务统一位于 [`../multi_terrain/`](../multi_terrain/README.md)。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `config_locomotion.py` | Locomotion 奖励、命令和迭代课程 |
| `config_common.py` | 机器人、动作、观测配置与命令合并工具 |
| `env.py` / `rewards.py` | 仿真环境、通用扩展接口与经典奖励 |
| `tracking_gate.py` / `velocity_estimator.py` | 跟踪门控与速度估计 |
| `train.py` / `eval.py` | 训练、续训与交互评估 |
| `interactive_viewer.py` | 评估键盘和显示辅助 |

## 训练和续训

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  -e locomotion_v4 -B 64 --max_iterations 10001

.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.train \
  -e locomotion_v4 --resume --max_iterations 15001
```

只有 `config_locomotion.py` 一套配置，不再需要 `--config`。
默认环境数为 8192；用 `-B` 根据可用显存调整。`--vis` 显示训练环境。
`--resume` 恢复保存配置、优化器及迭代；`--resume-config current` 显式改用当前 locomotion 配置。
`--load-weights CHECKPOINT` 只加载策略与价值网络，优化器和课程从零开始。
动力学与传感器随机化可分别通过 `--domain-rand/--no-domain-rand`、
`--sensor-noise/--no-sensor-noise` 覆盖。

`config_locomotion.py` 的课程先练站立，再逐步扩大速度、转向和高度范围，最后增加随机化。
默认地面为 plane；`--terrain` 可指定单一测试地形，仍使用 locomotion 的跟踪目标和奖励。
每个训练阶段使用单一地形，不包含混合分配或按成功率升降级；这些功能统一由 `multi_terrain` 管理。
地形几何参数见 [`terrains/`](../../terrains/README.md)。

## 评估

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.eval \
  -e locomotion_v4

.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.locomotion.eval \
  -e locomotion_v4 --terrain square_wave --difficulty 2
```

评估使用课程最终命令范围，支持键盘命令、ToF 显示以及周期性观测和控制器诊断。
`--version` / `--ckpt` 选择版本与 checkpoint；`--terrain-size LENGTH WIDTH` 覆盖地形尺寸。

## 转入多任务训练

上/下 20 cm 台阶、上/下 23° 坡、变摩擦起伏路和小高度单级上台阶，全部在
[`multi_terrain/config.py`](../multi_terrain/config.py) 的统一任务表中配置。
各任务共用环境与课程实现，没有单独的台阶专项环境或配置文件。

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.train \
  -e multi_terrain_v1 -B 64 --load-weights /path/to/locomotion/model_N.pt
```

旧多地形/上台阶专项 checkpoint 可以作为相同观测接口的权重来源，
其配置和旧专项课程不会通过 locomotion 入口继续运行。
多任务完整续训与单任务评估使用 `multi_terrain.train` / `multi_terrain.eval`。
