# 任务驱动多地形训练

独立入口 `tasks.multi_terrain`，一套策略同时训练七种通关任务和保留用的 locomotion。
机器人、控制器、IMU/ToF 观测及三维命令接口复用 locomotion；不额外向 actor 输入任务 ID 或终点真值。
相同观测配置的 locomotion 权重可通过 `--load-weights` 迁移。

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.train \
  -e multi_terrain_v1 -B 64 --max_iterations 25001

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
| `stairs_up` | 上一级 20 cm 台阶 | 高度 15 → 17 → 20 cm | 1，已启用 |
| `slope_up` | 上一个独立坡面 | 坡角等级：5 → 10 → 15 → 20 → 23 → 25°，高差 25 cm | 1 |
| `slope_down` | 下一个独立坡面 | 与上坡相同，但独立统计成功率 | 1 |
| `rough` | 通过二维随机起伏地面 | `random_rough` 高度参数 1 → 2 → 3 → 4 → 5 cm | 1 |
| `square_wave` | 登上一级小高度台阶 | `square_wave` 高度 2 → 2.5 → 3 → 3.5 → 4 → 4.5 → 5 cm | 1 |
| `platform_ridge` | 朝 −x 越过凸台，经第一平台下到地面 | 第二平台与凸台高差 1 → 2 → 3 → 4 → 5 cm | 15000 轮前为 0，之后为 1 |

几何和难度唯一来源是 `terrains` 注册模块的 `PARAMETERS` 与 `LEVELS`。
上述表格描述当前默认值；修改模块难度表会作用于新训练，已有训练使用保存的 `level_catalog` 快照。
上台阶从 15 cm 开始，按该任务的独立成功率课程推进到 17 cm、20 cm。

任务层不再重建单级楼梯、小高度上台阶或分段变摩擦起伏。
台阶、坡面模块提供 `traversal_end`，任务终点沿通行方向位于其后 `landing_length` 处；台阶和坡面默认 0.8 m。
随机起伏没有固定障碍末端，使用 `traversal.distance=3.0 m`，终点在 x=3.8 m。
下台阶默认末沿 x=0.8 m，终点 x=1.6 m，地块长 16 m，回合上限 10 s。
`square_wave` 保留原入口名，现为单级上台阶：从低平地出生，台阶沿 x=0.5 m，高平台延伸至地块末端，终点 x=1.3 m，回合上限 10 s。
独立坡面的参数按注册等级读取，水平坡长由 `height / tan(angle)` 决定。
随机起伏复用二维非凸网格与统一材质，不再具有旧 task 几何的左右分区摩擦差。

`enabled=False` 或 `weight=0` 关闭某任务，其他任务按权重重新归一化。
环境按最大余数法分组，同一全局阶段内不重抽任务。全局阶段改变任务权重时，重新计算目标配额；
每个环境在下一次 reset 时才采用新任务，正在运行的回合保留原来的奖励、命令和成功率归属。
所有环境完成一次 reset 后达到新配比；稳定阶段内，回合长短不改变环境占比。
由于环境数量为整数，比例存在最多一个环境的取整误差；任何阶段权重大于 0 的任务分不到环境时直接报错。
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

每个 task 在 `config.py` 中独立集中配置，例如小高度上台阶：

```python
terrain=dict(preset="square_wave", tile_size=[16.0, 6.0]),
traversal=dict(landing_length=0.8),
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
不污染新课程窗口。关闭课程后固定在 `difficulty`。例如直接练小高度上台阶最高级，设 `difficulty=6`。

若某些难度需要调整命令或奖励，可在该 task 的 `curriculum` 中添加：

```python
overrides={
    1: dict(commands=dict(lin_vel_range=[0.25, 0.45]), rewards=dict(task_progress=7.0)),
},
```

这些行为覆盖项按难度累计继承，支持 `commands`、`rewards`、`termination`、`success`、
`resampling_time_s`、`tracking_success`、`landing_stability` 和 `ascent`。不允许覆盖几何；几何只在 `terrains` 中维护。

locomotion 固定使用 plane 的 0 级，完整时长回合内的线速度、角速度、高度 RMSE 达标记为成功，
阈值在 `tracking_success` 中。全局 `curriculum_cfg` 由 `config.py` 的 `_get_curriculum_cfg()` 返回，控制动力学随机化、传感器噪声强度和任务配额：

```python
def _get_curriculum_cfg():
    """按阶段配置随机化、传感器噪声与任务权重；15000 轮起加入凸台任务。"""
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
                },
            },
            {
                "name": "full",
                "start_iteration": 1500,
                "targets": {
                    "domain_rand": {"strength": 1.0},
                    "sensor_noise": {"strength": 1.0},
                },
            },
            {
                "name": "platform_ridge",
                "start_iteration": 15000,
                "targets": {
                    # 保留已有任务及其权重，新增凸台任务；随机化和噪声继承 full 阶段。
                    "task_weights": {
                        "platform_ridge": 1.0,
                    },
                },
            },
        ],
    }
```

后续阶段继承此前的目标值；凸台阶段保持随机化和噪声强度 1.0，其他任务保留各自权重。
凸台几何及 5 个难度在初始化时预建，前期不分配环境；启用后从 0 级开始独立统计成功率并升降级。
`enabled=False` 或任务初始 `weight=0` 会禁用该任务，全局 `task_weights` 不会重新启用未构建的任务。
关闭全局课程时使用任务原始权重（凸台为 1），因此会从训练开始就分配凸台环境。
训练入口默认上限为 25001 轮，给 15000 轮后新增的任务留出训练时间。

`platform_ridge` 从第二平台中央朝 −x 出发；越过 35 cm 凸台、20 cm 第一平台后下到地面，
继续前进 0.8 m 到终点。进度、双轮越沿和航向奖励按 −x 方向计算；其他通关任务仍沿 +x。

`--resume` 默认读保存配置及难度表快照；变更任务集合、难度表或权重应新建训练并 `--load-weights`。
旧的 `geometry` / `curriculum.stages` 格式及其课程状态不再兼容，旧策略可以通过 `--load-weights` 迁移。

## 通关奖励与结束条件

七种通关任务禁止非零 `alive` 奖励；locomotion 保留经典奖励。
通关任务默认使用：

- `task_progress=5`：每前进到本回合新的最远位置奖励 5 × 新增米数，最多计算到终点。
- `task_success=20`：成功终止一次奖励 20，不乘 `dt`。
- 横向偏离、朝向偏离、动作变化、姿态、关节限位、接触和失败惩罚。

进度奖励也不再乘 `dt`；来回倒车不能重复获得进度。其他连续奖励按 `dt` 积分，
`death` 是一次性失败惩罚。默认通关任务不启用持续速度/高度跟踪正奖励。

### 上台阶三阶段奖励

`stairs_up` 在现有水平进度奖励之外增加靠近时抬升、搭接前后摆、触台后缩腿的增量奖励，以及稳定搭台事件奖励。
参数集中在 `config.py` 的 `tasks["stairs_up"]["ascent"]`。

`stairs_up_approach_lift=2`：机体原点在第一级台阶前 `approach_distance_m=0.35 m` 内，
双轮仍在下方地面接触、底盘没有接触且尚未获得搭接奖励、roll/pitch 满足成功姿态门限时，
对底盘下底面向“眼前台阶顶面高度 + `approach_clearance_m=0.01 m`”的新增抬升发奖。
下底面使用 baselink 实际碰撞体的最低点，包含碰撞体自身偏移和机身 roll/pitch 的影响，
不使用视觉网格或固定机体原点高度。15 / 17 / 20 cm 台阶对应下底面目标为 16 / 18 / 21 cm
（相对下方地面）；达到余量后继续抬高不再增加奖励。每拍奖励为：

`2 × max(0, min(当前下底面高度, 台阶顶面高度 + 余量) − 此前最高下底面高度) / max(台阶顶面高度 + 余量 − 出生下底面高度, 1e-6)`。

此前最高下底面高度在 reset 时初始化为出生下底面高度（截断到目标高度），随后每拍更新。
奖励不乘 `dt`，每回合累计最多 +2；保持高度、反复升降、离开再进入奖励区都不能重复领奖。
区域外或腾空获得的高度也会更新最高值，不能在进入奖励区或落地时补领奖励。
出生时已达到目标高度则不发该项；同拍失败时不发正奖励。
越沿余量和 +2 系数是初始训练设置，可在 `ascent` 和 `rewards` 中分别调整。

`stairs_up_backswing=2`：底盘首次接触台面之前，在台阶沿前后 `approach_distance_m` 范围内、
双轮仍位于立面后方、机身没有接触且未超过失败姿态门限时，鼓励左右腿向后摆。
使用实测关节角计算的虚拟腿角，当前模型正值为后摆，`backswing_angle_deg=20` 为初始目标，
低于当前 ±30° 动作摆角限幅。无需等待抬升完成，也不要求双轮一直接地，允许腿/轮抵住立面时后摆。

每条腿的进度为 `clamp(腿角 / 目标后摆角, 0, 1)`，只对超过本回合该腿历史最高进度的增量发奖，
两腿增量取平均后乘 2，不乘 `dt`。左右腿各最多贡献 +1，每回合合计最多 +2；
保持姿态、反复前后摆、在远处摆好后进入范围都不能重复或补领奖励，同拍失败不发奖励。
出生已有的后摆进度不计奖。首次真实台面接触（向上力 >0）当拍立即关闭该奖励，并锁存至 reset，
不等待搭接奖励的 5 N 支撑力、搭接深度或 0.10 s 稳定时间；仅碰侧壁不会锁存台面接触。
此项只提供奖励引导，不直接修改动作，也不要求最终成功必须完成后摆。

`stairs_up_retract` 在 baselink 首次真实接触台面（向上力 >0）时记录左右腿长，随后对每条腿
相对该基准的新增缩短量发奖，无需等待稳定搭接奖励、轮子越沿或水平前进。
每腿进度为 `clamp((触台时腿长 − 当前腿长) / retract_shortening_m, 0, 1)`；
只奖励超过该腿历史最高进度的增量，两腿取平均后乘奖励权重，不乘 `dt`。
当前 `retract_shortening_m=0.05 m`、权重为 10，因此左右腿各缩短 5 cm 时累计最多 +10；
保持腿长、反复伸缩不能重复领奖，首次触台之前的缩短量不补奖。

双轮第一次同时落在顶部台面，当拍即永久关闭该回合缩腿奖励，不等稳定站立或最终成功。
判定要求双轮都有接触、轮心进入台面至少 `wheel_edge_margin_m=0.01 m`、位于赛道范围内，
且轮底距顶部台面不超过 `wheel_bottom_tolerance_m=0.005 m`；单轮落台、腾空越沿均不触发。
双轮落台后，任一轮底低于台面超过 `fall_height_tolerance_m=0.03 m` 就当拍按失败终止，
无需等待落到下方地面或停滞超时；向上弹跳不会触发，也不会重启缩腿奖励。
即使没有 baselink 搭台、直接双轮登台，跌台终止也生效。

`stairs_up_takeoff=10`：本回合确认双轮都曾接触台下地面后，双轮轮底第一次同时高出台下地面
`wheel_bottom_tolerance_m=0.005 m` 就发放一次 +10，不乘 `dt`，不要求 baselink 已搭台。
单轮抬起或只有接触信号消失不发奖；轮子离地后抵住立面仍可触发。
空中出生不会直接领奖，需先实际双轮接触台下地面。离地状态锁存至 reset，之后任一轮
再次接触台下地面就当拍按失败终止，无需先登上台面；落到上台面不会触发此失败。
同拍失败时不发离地正奖励。

`stairs_up_landing_distance=10`：任一轮第一次真实接触顶部台面时，按接触点沿 +x 距台阶入口边缘
的水平距离（m）乘权重发奖，不乘 `dt`。例如首次落点深入台面 10 cm 得 +1，30 cm 得 +3；
距离越大奖励越大，不作额外饱和截断，落点必须位于平台赛道范围内。
使用 Genesis 真实轮子接触点，要求向上受力、接触高度距台面 ≤15 mm、法向竖直分量绝对值 ≥0.9；
baselink 搭台、侧壁和台下地面接触不算。单轮先落台就立即发放，不等另一轮或稳定站立；
同拍多个轮/接触点触台时取离入口边缘最近的一个距离。
即使第一次恰好触到边缘得到 0，也会锁存本回合机会；之后滚动前进、另一轮落台或弹起重落均不补奖。
同拍失败时不发该正奖励，部分 reset 仅清空所选环境的落点记录。

以下两个事件同样不乘 `dt`，每回合各最多发放一次：

- `stairs_up_support=10`：baselink 的真实接触点进入顶部台面至少 10 cm，接触点高度距台面不超过
  15 mm，接触法向竖直分量绝对值 ≥0.9，向上支撑力合计 ≥5 N，并连续稳定支撑 0.10 s。
  侧壁、低处地面、腿/轮接触和瞬间撞击不计入。
- `task_success=20`：保留到达上平台终点、双轮越沿并接地、底盘脱离接触、roll/pitch 各不超过 10°
  的条件；底盘离台面高度还需 ≥0.20 m，并连续稳定站立 0.20 s。直接登上台阶也可以成功，
  不强制先领取搭台或缩腿奖励。

支撑和站立窗口均要求竖直速度 ≤0.3 m/s、roll/pitch 角速度合范数 ≤1.5 rad/s；
水平前进速度不受此稳定门限限制。窗口中断会重新计时，已发奖励不会因失去接触或倒车重发。
部分环境 reset 只清空对应回合的状态；同拍失败时不发上述正奖励。

TensorBoard 新增 `task/stairs_up/support_reached`、`task/stairs_up/retract_reached`、
`task/stairs_up/takeoff_reached`、`task/stairs_up/first_landing_reached` 回合到达率，
以及 `task/stairs_up/first_landing_distance_m` 首次轮子触台距离（未触台回合记为 0），
其中 `retract_reached` 表示落台前两腿都曾达到目标缩短量，
以及 `reward_stairs_up_support`、`reward_stairs_up_retract`、`reward_stairs_up_approach_lift`、
`reward_stairs_up_backswing`、`reward_stairs_up_takeoff`、`reward_stairs_up_landing_distance` 奖励统计。

使用当前任务配比开启新训练（保留其他已启用任务及 locomotion）：

```bash
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.train \
  -e stairsup_v1 -B 64 --max_iterations 25001 \
  --load-weights logs/locomotion_v4/version_0002/model_10000.pt
```

该命令读取当前代码中的奖励和 15 → 17 → 20 cm 课程。`--resume` 默认读取旧运行保存的配置，
不会自动启用新增奖励；迁移已有多地形策略时也应使用 `--load-weights` 指向所选 checkpoint。

```bash
.venv/bin/python -m unittest tests.genesis.test_stairs_up_rewards -v
NGXY_STAIRS_UP_SMOKE=1 .venv/bin/python -m unittest \
  tests.genesis.test_stairs_up_rewards.StairsUpSimulationTests -v
```

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
目前 `LocomotionBuffers` 保存跟踪误差，`StairsDownBuffers` 保存落地触发状态、剩余步数和窗口开关，
`StairsUpBuffers` 保存支撑/站立计时、首次触台腿长、缩腿最高进度、台下接地/离地、双轮首次落台及跌台状态。
`env.py` 按相同任务顺序调用创建、更新和重置；部分重置只清空选中环境，不清空其他回合或固定参数表。

到终点且双轮已越过最后障碍、落地接触正常、无机身接触、姿态合格时，结束并发成功奖励；
`stairs_up` 还需完成上述 0.20 s 稳定站立窗口。
小高度上台阶需到达高平台上的终点后才会成功。停滞或驶出任务走廊按失败处理；最大时长按截断处理。
失败优先于成功，成功不发 death，也不作为 timeout bootstrap。
同一拍既超时又失败时同样不进行 timeout bootstrap。

TensorBoard 的 `task/<name>/...` 分别记录环境占比、当前难度（`difficulty`）、课程成功率窗口、
实际回合成功/失败、时长、通关进度，以及 locomotion 三项跟踪误差。

## 单任务评估与测试

```bash
# 默认评估所选任务的最终难度，可通过 --difficulty 0 查看初始等级（--stage 为兼容别名）。
.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.eval \
  -e multi_terrain_v1 --task square_wave

.venv/bin/python -m experiments.genesis.wheel_leg_infantry.tasks.multi_terrain.eval \
  -e multi_terrain_v1 --task platform_ridge --difficulty 0

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
