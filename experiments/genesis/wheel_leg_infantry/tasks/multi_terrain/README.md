# 任务驱动多地形训练

独立入口 `tasks.multi_terrain`，一套策略同时训练六种通关任务和保留用的 locomotion。
机器人、控制器、IMU/ToF 观测及三维命令接口复用 locomotion；不额外向 actor 输入任务 ID 或终点真值。
相同观测配置的 locomotion 权重可通过 `--load-weights` 迁移。

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.train \
  -e multi_terrain_v1 -B 64 --max_iterations 15001

# 迁移已有策略；优化器和所有任务课程从头开始。
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.train \
  -e multi_terrain_v1 -B 64 --load-weights logs/locomotion_v4/version_0000/model_10000.pt

# 保存/恢复各任务等级、成功率窗口及 epoch，同时恢复 PPO 状态。
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.train \
  -e multi_terrain_v1 --resume --max_iterations 20001
```

训练需要可用的 CUDA/Genesis 环境。默认 64 个环境；所有任务的课程几何预先构建为独立区域，
增大环境数量前应先测显存与吞吐。示例 checkpoint 路径需替换为实际文件。

当前 Genesis ToF Raycaster 会为每个环境分配整个场景的三角面 BVH，即使地形使用
`batch_fixed_verts=False`，该 BVH 也不会随静态顶点一起共享。以 127196 个碰撞面为例，
4096 个环境仅 AABB、BVH 节点和主要排序缓存就约需 63 GiB，尚未包含物理仿真和 PPO。
24 GiB GPU 请先使用 `-B 64`；不要直接沿用平地训练的 4096。构建时若出现字段 shape
异常或显存不足，先降低 `-B` 验证；增加任务难度等级或网格精度也会增加面数与显存。

## 默认任务与配比

配置入口：`config.py` 的 `get_cfgs()`，最终保存在 `env_cfg["multi_terrain"]["tasks"]`。

| 任务 | 最终目标 | 默认课程 | 权重 |
| --- | --- | --- | --- |
| `locomotion` | 平地速度、高度跟踪、姿态保持与静止采样 | 保留经典最终命令与奖励，默认不升降级 | 2 |
| `stairs_down` | 下一级台阶 | 高度 15 → 20 → 25 cm | 1 |
| `stairs_up` | 上一级 20 cm 台阶 | `stairs_up.LEVELS` 当前仅有 0 级 | 1，当前关闭 |
| `slope_up` | 上一个独立坡面 | 坡角等级：5 → 10 → 15 → 20 → 23 → 25°，高差 25 cm | 1 |
| `slope_down` | 下一个独立坡面 | 与上坡相同，但独立统计成功率 | 1 |
| `rough` | 通过二维随机起伏地面 | `random_rough` 高度参数 1 → 2 → 3 → 4 → 5 cm | 1 |
| `square_wave` | 越过下一完整高平台并落到低平台 | `square_wave` 高度 2 → 2.5 → 3 → 3.5 → 4 → 4.5 → 5 cm | 1 |

几何和难度唯一来源是 `terrains` 注册模块的 `PARAMETERS` 与 `LEVELS`。
上述表格描述当前默认值；修改模块难度表会作用于新训练，已有训练使用保存的 `level_catalog` 快照。
上台阶目前没有更低难度，若需渐进训练，应在 `terrains/stairs_up.py` 中增加等级。

任务层不再重建单级楼梯、单个方波或分段变摩擦起伏。
台阶、坡面、方波模块提供 `traversal_end`，任务终点在其后 `landing_length` 处；台阶和坡面默认 0.8 m，方波 0.2 m。
随机起伏没有固定障碍末端，使用 `traversal.distance=3.0 m`，终点在 x=3.8 m。
下台阶默认末沿 x=0.8 m，终点 x=1.6 m，地块长 16 m，回合上限 10 s。
方波从高平台中央出生，高、低平台各长 0.5 m；经过下一完整高平台后，末沿 x=1.25 m，终点 x=1.45 m，回合上限 10 s。
独立坡面的参数按注册等级读取，水平坡长由 `height / tan(angle)` 决定。
随机起伏复用二维非凸网格与统一材质，不再具有旧 task 几何的左右分区摩擦差。

`enabled=False` 或 `weight=0` 关闭某任务，其他任务按权重重新归一化。
环境按最大余数法固定分组，每次部分 reset 不重抽任务，只更新该任务的课程等级。
因此通关回合较短也不会改变每个 PPO 批次的环境占比。
由于环境数量为整数，比例存在最多一个环境的取整误差；任何启用任务分不到环境时直接报错。
训练必须保留 `locomotion.enabled=True` 且 `weight>0`。

## 独立课程、奖励与命令

每个任务的主要字段：

- `enabled`、`weight`：是否启用、环境数量权重。
- `commands`：前进速度、角速度、高度范围及静止概率。
- `resampling_time_s`：本任务自己的命令重采样周期。
- `rewards`：本任务完整奖励权重表，未指定的奖励为零。
- `terrain`：注册地形的 `preset`、地块尺寸等公共参数；高度等难度参数由 `LEVELS` 提供。
- `traversal`：末沿后的落地距离；无固定末沿的随机地面另设通行距离。
- `termination`：最大时长、姿态与接触容忍时间、赛道宽度、停滞超时。
- `success`：终点允许姿态、轮子越过障碍的余量与落地高度误差。
- `curriculum`：本任务初始/最低/最高难度、成功率窗口和升降级阈值。

每个 task 在 `config.py` 中独立集中配置，例如方波：

```python
terrain=dict(preset="square_wave", tile_size=[16.0, 6.0]),
traversal=dict(landing_length=0.2),
curriculum=dict(
    enabled=True,
    difficulty=0,           # 初始 LEVELS 等级
    min_difficulty=0,
    max_difficulty=None,    # 当前快照中的最高等级；也可指定整数上限
    window_episodes=100,
    promote_threshold=0.8,
    demote_threshold=0.3,
),
```

每项默认统计至少 100 个同等级、同 epoch 的结束回合，成功率 ≥0.8 升一级、≤0.3 降一级。
升降级由本目录 `curriculum.py` 的 `TaskCurriculum` 统一实现，各 task 独立设置窗口和阈值。
只在该环境下一次 reset 应用新难度；旧等级仍在运行的回合保留原来的地形、命令和奖励，
不污染新课程窗口。关闭课程后固定在 `difficulty`。例如直接练方波最高级，设 `difficulty=6`。

若某些难度需要调整命令或奖励，可在该 task 的 `curriculum` 中添加：

```python
overrides={
    1: dict(commands=dict(lin_vel_range=[0.25, 0.45]), rewards=dict(task_progress=7.0)),
},
```

这些行为覆盖项按难度累计继承，支持 `commands`、`rewards`、`termination`、`success`、
`resampling_time_s`、`tracking_success` 和 `landing_stability`。不允许覆盖几何；几何只在 `terrains` 中维护。

locomotion 固定使用 plane 的 0 级，完整时长回合内的线速度、角速度、高度 RMSE 达标记为成功，
阈值在 `tracking_success` 中。全局 `curriculum_cfg` 仅控制动力学随机化和传感器噪声强度。
`--resume` 默认读保存配置及难度表快照；变更任务集合、难度表或权重应新建训练并 `--load-weights`。
旧的 `geometry` / `curriculum.stages` 格式及其课程状态不再兼容，旧策略可以通过 `--load-weights` 迁移。

## 通关奖励与结束条件

六种通关任务禁止非零 `alive` 奖励；locomotion 保留经典奖励。
通关任务默认使用：

- `task_progress=5`：每前进到本回合新的最远位置奖励 5 × 新增米数，最多计算到终点。
- `task_success=20`：成功终止一次奖励 20，不乘 `dt`。
- 横向偏离、朝向偏离、动作变化、姿态、关节限位、接触和失败惩罚。

进度奖励也不再乘 `dt`；来回倒车不能重复获得进度。其他连续奖励按 `dt` 积分，
`death` 是一次性失败惩罚。默认通关任务不启用持续速度/高度跟踪正奖励。

下台阶额外配置 `stairs_down_landing_stability=-2.0`，专用参数集中在该 task 内：

```python
landing_stability=dict(duration_s=0.3, ang_vel_weight=0.25),
```

任一轮首次越过台阶末沿并接触下方地面后，启动 0.3 s 的落地窗口，每回合只启动一次。
窗口内每拍奖励为 `-2 × dt × (竖直速度² + 0.25 × (横滚角速度² + 俯仰角速度²))`。
竖直速度由机体系速度沿重力方向投影得到，水平前进速度不计入此项。
上平台、空中不会触发；触发后再次弹跳仍继续计入，回合提前结束时窗口随之结束。
该项衡量触地后的振荡，不直接测量撞击瞬间的峰值冲击力，也不持续发放静止正奖励。

任务专用奖励函数集中在 `rewards.py`。任务专用 buffer 集中在 `buffers.py`，
每个任务各有一个类，按固定参数表、回合状态分组，并在同一类中实现 `create()` 和 `reset()`。
目前 `LocomotionBuffers` 保存跟踪误差，`StairsDownBuffers` 保存落地触发状态、剩余步数和窗口开关。
`env.py` 按相同任务顺序调用创建、更新和重置；部分重置只清空选中环境，不清空其他回合或固定参数表。

到终点且双轮已越过最后障碍、落地接触正常、无机身接触、姿态合格时，立即结束并发成功奖励。
方波平台上不会提前结束。停滞或驶出任务走廊按失败处理；最大时长按截断处理。
失败优先于成功，成功不发 death，也不作为 timeout bootstrap。
同一拍既超时又失败时同样不进行 timeout bootstrap。

TensorBoard 的 `task/<name>/...` 分别记录环境占比、当前难度（`difficulty`）、课程成功率窗口、
实际回合成功/失败、时长、通关进度，以及 locomotion 三项跟踪误差。

## 单任务评估与测试

```bash
# 默认评估所选任务的最终难度，可通过 --difficulty 0 查看初始等级（--stage 为兼容别名）。
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.eval \
  -e multi_terrain_v1 --task square_wave

# 自动命令、自动终点 reset，打印成功数；无需 GUI。
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.eval \
  -e multi_terrain_v1 --task slope_down --headless --steps 3000

# 无需 GPU 的几何、配置、课程、奖励、命令及重置测试。
.venv/bin/python -m unittest tests.genesis.test_multi_terrain -v

# 可选：真实 Genesis CPU 构建、步进和状态恢复测试。
NGXY_MULTI_TERRAIN_SMOKE=1 .venv/bin/python -m unittest \
  tests.genesis.test_multi_terrain.SimulationTests -v
```

课程和奖励是初始训练配置，各地形最终等级的成功率需要训练后验证。
