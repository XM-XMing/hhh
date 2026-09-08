# XMflight 三维航迹规划项目交接文档

更新日期：2026-08-09  
代码包版本：`planning 0.10.2`  
项目目录：`/home/xm/XM/xm_ws/src/planning`  
工作空间：`/home/xm/XM/xm_ws`  

## 当前 RL owner（2026-09-03）

`AWAC_ONLY_ARCHITECTURE_MIGRATION_V1` 已完成。当前正式 RL owner 是
`planning.awac`，训练入口是 `scripts/train_awac.py`；`planning/rl/`、旧
SAC 生产模块和旧 guarded runner 已删除，不提供兼容包装。下面保留的旧
SAC/AWAC 混合描述属于迁移前历史交接证据，不是当前执行命令；当前的
owner、checkpoint、replay、runtime 和 evaluation 边界以
[`AWAC_ONLY_ARCHITECTURE_MIGRATION_V1.md`](AWAC_ONLY_ARCHITECTURE_MIGRATION_V1.md)
为准。

## 1. 交接结论

本项目的目标是在 Unity 森林环境中，仅使用深度图、无人机状态、最终目标和上一动作，完成无人机三维航迹规划。允许飞行高度为 1～3 m，不允许部署策略读取全局地图、全局路线、教师代价或未来专家轨迹。

当前已经打通并留有证据的主链路是：

```text
105 个运动基元
  -> 安全 mission 生成
  -> 专家 mission 审计
  -> 20 Unity 并行执行
  -> 离线软标签和深度 action mask
  -> 数据集审计与 mmap
  -> Soft BC
  -> 固定 dev100 闭环评估
  -> 受控 AWAC 在线实验
```

截至本次交接：

- 当前应保留的基线是 60,011 条专家成功轨迹训练出的 BC；checkpoint SHA256 为 `188e41301b67cd8c53ffdd45d55684c5f7626b4a4208eca599e322870e8c31f7`。
- 该 BC 在新的、与旧/新 BC 数据均零重合的 dev100 上取得 67% 成功、7% 碰撞、24% dead-end、2% timeout。它优于旧 BC，但仍未满足正式碰撞门槛 5%，因此是“当前已接受训练基线”，不是生产验收模型。
- AWAC 18k 已通过全 Replay KL 安全审计，但两次 dev100 选择结论相反；最终没有晋级。当前不可把 AWAC candidate 当成部署模型，也没有证据支持继续直接扩到 24k。
- 原先 formal evaluation 的主要不可重复来源已定位并修复：host-timed 25-frame primitive 会产生 24/25 个实际积分 tick。schema v3 现在把完整 primitive 原子提交给 Unity，并由 `FixedUpdate` 连续执行恰好 25 tick；episode 451/action 52 的 5-repeat 回归为 `[25,25,25,25,25]`，endpoint max delta 为 0。
- 修复后的 episode 451 完整轨迹 20-repeat 在所有行为内容上完全一致，但严格 comparator 仍在 reset 阶段报告 timing-only `F` divergence：state/depth 内容相同，sensor skew 的观测范围为 47.691～70.961 ms。按诊断门控，multi-mission 与三次完整 dev100 reproducibility 尚未执行；因此 formal evaluation 仍未取得“可用于 BC/RL policy comparison”的最终验收证据。
- `final_holdout_100.csv` 尚未用于模型选择，必须继续封存，直到出现通过重复 dev 选择的最终候选。
- 历史 SAC 没有候选模型通过晋级。当前 AWAC 已完成独立 owner 收敛；旧
  `sac_*` 生产模块和 guarded runner 不再是当前依赖，详见
  [`AWAC_ONLY_ARCHITECTURE_MIGRATION_V1.md`](AWAC_ONLY_ARCHITECTURE_MIGRATION_V1.md)。

## 2. 不可变边界与验收合同

### 2.1 Unity 边界

Unity 项目位于 `/home/xm/XM/XMflight`，运行时使用已构建程序 `../unity/XMflight.x86_64`。2026-08-09 为修复已确认的 command-to-physics-tick 协议缺陷，Unity 与 bridge 已同步升级到 schema v3；这是经过真实 RED reproduction 约束的协议修复，不是通过 Unity 改动掩盖算法问题。

当前 Unity player/bridge 身份：

- `unity/XMflight.x86_64` SHA256：`61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365`；
- `unity/XMflight_Data/Managed/Assembly-CSharp.dll` SHA256：`50cbec315d728dd0a13996551c7ec0af52398310aee1969c17a6b67f5975f6ac`；
- `devel/lib/planning/unity_bridge_node` SHA256：`d9c149b6a0f9e07d10ad9b6ebd5151a1e92350ea0c7acffda7bdb3578c91bc1d`。

后续不得无证据修改 Unity。任何必要的协议变更必须显式升级 schema、同时更新 Unity/bridge/Python validator，并由真实 Unity contract test 验收；旧 schema 不允许 silent fallback。

### 2.2 任务合同

合同实现位于 `python/planning/mission_spec.py`：

| 字段 | 固定值 |
| --- | ---: |
| task contract | `xm_3d_flight_z1_3` |
| task SHA256 | `5862af0f354c408d74cf4904dc7ecf609944553443ae1128097136be979fd30a` |
| 起终点水平距离 | 40 m |
| 飞行高度 | `[1.0, 3.0]` m |
| 成功水平半径 | 1.2 m |
| 成功高度误差 | 不超过 0.2 m |
| 单回合最大基元数 | 45 |
| 终止优先级 | 失败终止优先于 progress/success |

### 2.3 运动基元合同

配置位于 `config/motion_primitives.yaml`：

- 15 个水平选择 × 7 个垂直选择 = 105 个离散动作；
- 目标前向速度 3 m/s；
- 每个基元前向距离 1.5 m、持续 0.5 s；
- 控制周期 0.02 s；
- 坐标系为机体/ROS：x 向前、y 向左、z 向上；
- 当前 primitive SHA256：`f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d`。

### 2.4 专家与路径合同

专家实现位于 `python/planning/teacher_policy.py`，当前流程是：

```text
pose=[x,y,z,yaw]
  -> 高度 action mask
  -> 点云 voxel 碰撞检查
  -> 0.25 m 粗粒度全局路线提供局部引导
  -> depth=3、width=8、branching=4 的有界 beam search
  -> 代价排序并执行首个 primitive
```

正式数据合同：专家计划路径不超过 44 m；Unity 实际观测折线路径不超过 46 m。44～46 m 是执行误差带，仍可接受。全局 voxel/route 只用于生成专家数据，不是策略运行时输入。

### 2.5 部署输入与安全合同

策略输入合同为 `depth_goal_state_prev_action`：

- 深度图；
- 当前状态；
- 最终目标；
- 上一动作 one-hot（105 维）；
- 连续特征 22 维，总向量 127 维。

禁止的部署输入包括 `global_map`、`global_route`、`route_waypoint`、`collision_voxel_map`、`teacher_costs`、`future_expert_path`。正式运行必须使用 `safety_mask=depth`、`execution_mode=continuous`，checkpoint 与评估参数不一致时应直接拒绝，不能静默覆盖。

### 2.6 Deterministic primitive execution 合同

正式 evaluator/RL 的 `UnityForestEnv.step_primitive()` 使用 physics-clock-bound schema v3：

```text
Python/ROS 一次提交完整 execution
  -> execution_id + frame_count + 逐帧 frame_index/command_id/payload
  -> Unity 完整校验并缓冲
  -> FixedUpdate K 应用 frame 0，StepDynamics，发布 post-integration state K
  -> ...
  -> FixedUpdate K+24 应用 frame 24，发布 complete state K+24
```

`XMState` 回显 `applied_execution_id`、`applied_execution_frame_index`、`applied_command_id`、`state_id`、`sim_time_ns` 和 `execution_status`。Python 必须验证：

- frame 数严格为 N，frame index 和 state ID 连续；
- command ID 与提交序列完全一致；
- endpoint 为 frame N-1 的 post-integration state；
- N=25 时 `endpoint_state_id = first_applied_state_id + 24`；
- collision/frozen 导致未积分时显式 `execution_failed`，不得伪造 applied/complete；
- deterministic evaluator/RL 不得回退到旧的 host-timed `cmd_vel + rospy.Rate` 协议。

协议入口位于 `include/planning/protocol/xm_protocol.hpp`、`msg/PrimitiveExecution.msg`、`msg/XMState.msg`、`src/unity_bridge_main.cpp`、`src/bridge/*`、`python/planning/primitive_execution_contract.py` 和 `python/planning/unity_env.py`。`execute_primitive_stream()` 仍是显式 legacy teacher-rollout 路径，不是 evaluator/RL fallback。

## 3. 系统结构与代码入口

| 目录 | 职责 |
| --- | --- |
| `python/planning/` | 正式 Python 实现和合同；业务逻辑应放这里 |
| `python/planning/cli/` | CLI 参数解析和流程入口 |
| `scripts/*.py` | ROS/源码兼容薄包装，不应复制业务逻辑 |
| `scripts/*.sh` | managed evaluation 生命周期和诊断编排；正式 Collection/AWAC 使用 Python |
| `src/`、`include/` | Unity ZMQ 桥、C++ voxel 碰撞、深度安全和可视化 |
| `config/` | 运动基元和 Unity 端口配置 |
| `launch/` | ROS bridge、地图和可视化 launch |
| `tests/` | CPU unit、ROS、Unity、CUDA 和 performance 分组测试 |
| `data/` | 生成数据、模型和实验结果；默认不进 Git |

关键模块：

- `mission_spec.py`：任务、成功条件、高度和 reward 合同；
- `motion_primitives.py`：105 个基元的运行时读取与高度 mask；
- `collision_checker.py`、`collision_checker.cpp/.cu`：CPU/CUDA voxel 碰撞检查；
- `global_route.py`、`teacher_policy.py`：粗路线和有界 beam 专家；
- `unity_env.py`：连续执行、终止判定、奖励和深度安全；
- `primitive_execution_contract.py`：physics-clock primitive receipt 的 exact-N、ID、顺序和 endpoint validator；
- `feature_contract.py`、`policy_runtime_contract.py`：部署输入、mask、执行方式；
- `bc_model.py`、`train_soft_bc.py`：Soft BC；
- `awac_learner.py`：离散 masked AWAC；
- `awac/model.py`、`awac/learner.py`、`awac/optimization.py`：正式 AWAC
  Actor/Critic 与优化 owner；
- `awac/replay.py`、`awac/replay_audit.py`：Replay 和审计 owner；
- `awac/trainer.py`、`scripts/train_awac.py`：正式 AWAC trainer entrypoint；
- `scripts/evaluate_policy_unity.py`：Python evaluation entrypoint；需要
  managed ROS/Unity/Bridge 生命周期时由 `evaluate_policy_unity_managed.sh`
  负责外层进程管理。

旧 SAC/AWAC 混合模块和 runner 的删除、调用者核对与测试覆盖映射记录在
[`SHELL_ENTRYPOINT_CLEANUP_AND_TEST_SUITE_INTEGRITY_V1.md`](SHELL_ENTRYPOINT_CLEANUP_AND_TEST_SUITE_INTEGRITY_V1.md)。

## 4. 运行环境

已使用环境：

- Ubuntu 20.04 / Linux 5.15；
- ROS Noetic；
- Conda 环境 `/home/xm/anaconda3/envs/xm`；
- Python 3.8.20；
- PyTorch 2.4.1，构建 CUDA 11.8；
- NumPy 1.24.4、PyYAML 6.0.2、TensorBoard 2.14.0；
- CUDA toolkit 11.8；
- 硬件：RTX 4060 Ti 16 GB、64 GB RAM、24 核 CPU。

系统依赖至少包括 `libzmq3-dev`、`libmsgpack-dev` 和 ROS package.xml 中列出的组件。当前仓库没有完整的 Conda/requirements 锁文件，这是交接风险；接手前应从真实训练环境导出锁定文件，而不是根据本节手工猜依赖。

基础构建：

```bash
conda activate xm
cd /home/xm/XM/xm_ws
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
cd src/planning
```

用户环境目前会在激活 `xm` 时自动 source workspace，但自动行为不应替代上述显式构建步骤。

## 5. 当前资产清单

### 5.1 必须交付

| 资产 | 相对路径 | 大小/摘要 |
| --- | --- | --- |
| voxel 地图 | `data/map_data/forest_voxels_10cm.npz` | 地图数据目录约 287 MB |
| 运动基元 | `data/motion_primitives/` | 约 2.4 MB |
| 6 万专家数据 | `data/teach/flight_20260717_6w/` | 约 48 GB |
| BC mmap | `data/bc/flight_20260717_6w/mmap/` | BC 目录合计约 46 GB |
| 当前 BC | `data/bc/flight_20260717_6w/model/checkpoint_best.pt` | SHA256 `188e4130…c31f7` |
| dev/final 留出集 | `data/smoke/bc_flight_20260717_6w/holdout_seed55/` | 约 3 MB |
| AWAC 18k 证据 | `data/sac/flight_20260717_6w/awac_guarded_12k_seed55_trust_tail_v3/` | 约 651 MB |

`data/teach`、`data/bc`、`data/sac`、`data/smoke` 和大二进制文件均被 `.gitignore` 排除。代码仓库和数据归档必须分开传输，并在接收端执行 SHA256 校验。

### 5.2 6 万数据证据

- 审计通过 mission：100,000；mission index SHA256 `e02d56dc2cd24d361521a918c9bafda6b81a22da5c3d727e311adae1d2c2194d`。
- Unity 尝试 75,382 回合，接受 60,011，接受率 79.61%。
- 接受数据碰撞 0、高度越界 0、collector error 0。
- 接受轨迹实际最大长度 45.9988 m，满足 46 m 合同。
- 训练 transition 共 1,652,285；105 个动作全部覆盖。
- 离线标签 invalid teacher/behavior 均为 0；最大 endpoint error 0.59969 m；最大 sensor skew 78.958 ms。
- `teacher_labels.npz` SHA256 `1ec06b11669f390048f3c511103708119968d04e90e7c39ed3ae68702fc32159`。
- `depth_action_masks.npz` SHA256 `114787a6260541efa641f1243503c1d28c1f89d20ac1a13ceeda327da5e7e409`。

### 5.3 BC 训练证据

- train：48,009 回合 / 1,321,806 transitions；
- validation：12,002 回合 / 330,479 transitions；
- 最佳 epoch：53；
- validation loss：1.96131；
- teacher top-1：57.81%，top-5：88.94%；
- behavior top-1：54.28%；
- 内部训练质量门通过。

内部 validation 通过只表示监督训练过程正常，不等价于 Unity 闭环质量通过。

### 5.4 固定评估集

目录：`data/smoke/bc_flight_20260717_6w/holdout_seed55/`

| 文件 | SHA256 | 状态 |
| --- | --- | --- |
| `fixed_dev_100.csv` | `135f21f73eeae2b034eda2ddcec0dd0dcaa915e8bf18426ebf9a04508f050316` | 可反复用于开发选择 |
| `final_holdout_100.csv` | `5506d0e33c8204fd846221e8f4084abfb65c5fcc214727664cffedb4a7003377` | 封存，尚未使用 |

选择种子为 55；dev/final 与旧 30,010 条及新 60,011 条 BC 数据的 mission 并集零重合，dev 与 final 也零重合。

新旧 BC 在同一 dev100 的一次配对比较：

| 模型 | 成功 | 碰撞 | dead-end | timeout | 最终距离均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 BC | 63% | 12% | 24% | 1% | 6.372 m |
| 6 万 BC | 67% | 7% | 24% | 2% | 5.842 m |

质量门为成功率至少 60%、碰撞率至多 5%、dead-end 至多 25%、timeout 至多 5%、far 为 0。新 BC 因 7% 碰撞未通过。

## 6. AWAC 18k 结果与停止理由

实验目录：`data/sac/flight_20260717_6w/awac_guarded_12k_seed55_trust_tail_v3/`

18k candidate SHA256：`e132148a2b3a226088a1e7cd1816dbd7c145b9b058f5d6012f7b8472e3089d73`。

全 Replay 审计通过：

- 18,000/18,000 replay rows 有效；
- BC KL mean 0.01219、p95 0.03167、p99 0.04133、max 0.05759；
- KL 超过 0.10 的行数为 0；
- candidate argmax 与 BC 不同 5.31%；
- candidate argmax 位于 BC top-3 之外 0.083%。

闭环选择却不稳定：

| 评估 | BC | AWAC 18k | 结论 |
| --- | --- | --- | --- |
| primary dev100 | 60/10/25/5 | 65/9/23/3 | candidate 胜出 |
| confirmation dev100 | 74/7/17/2 | 67/6/24/3 | incumbent 胜出 |

表中顺序为成功/碰撞/dead-end/timeout，单位为百分比。两次结果不一致，最终 decision 为 `candidate_selected=false`，accepted checkpoint 仍是 BC。说明当前主要问题不是 KL 越界，而是闭环收益没有跨重复运行稳定成立。

不要用 18k primary 的单次 65% 结果声称 AWAC 优于 BC，也不要启用 final100 来替代开发集调参。

## 7. 从零复现到 BC

以下命令都从 `planning` 目录执行。优先直接调用 `python3 scripts/...`；当前源码树和 devel 空间同时存在入口时，`rosrun` 可能提示 non-unique executable。

### 7.1 生成运动基元

```bash
python3 scripts/generate_motion_primitives.py \
  --config config/motion_primitives.yaml
```

### 7.2 生成并审计 mission

```bash
RUN=flight_20260717_6w
BASE="data/teach/$RUN"
ROLL="$BASE/rollouts"
BC="data/bc/$RUN"

python3 scripts/generate_missions.py   --out-index "$BASE/mission_candidates.csv"   --num-episodes 2000000   --seed 20260717 
  --num-workers 12   --route-workers 12   --max-sampling-attempts 800000   --route-candidate-factor 1.15   --overwrite

python3 scripts/audit_teacher_missions.py   --candidate-index "$BASE/mission_candidates.csv"   --out-index "$BASE/missions.csv"   --required-passing 100000   --max-steps 45   --num-workers 12   --collision-threads 2   --collision-backend cpu   --stop-when-required   --beam-depth 3   --beam-width 8   --beam-branching 4   --beam-discount 0.95   --progress-interval 100   --checkpoint-interval 1000   --overwrite
```

审计会增量写入 progress/checkpoint；中断后应先核对同一合同，再使用 `--resume`，不能一边改参数一边续跑。

### 7.3 20 Unity 并行收集 60,000 条成功轨迹

不要预先手工启动 Unity；Python collector 会管理 20 套 ROS master、bridge、Unity 和端口。

```bash
python scripts/collect_rollouts_parallel.py \
  --mission-index "$BASE/missions.csv" \
  --out-dir "$ROLL" \
  --workers 20 \
  --target-accepted 60000 \
  --max-steps 45 \
  --beam-depth 3 \
  --beam-width 8 \
  --beam-branching 4 \
  --beam-discount 0.95
```

完成后必须读取 `collection_summary.json` 的 `quality_pass` 和 `quality_gates`，不要再用容易因字段改名失效的手写单字段断言。

### 7.4 离线软标签、深度 mask 和审计

```bash
PLANNING_COLLISION_THREADS=2 python3 scripts/label_teacher_rollouts.py   --index "$ROLL/rollout_index.csv"   --out-labels "$ROLL/teacher_labels.npz"   --collision-cache data/map_data/forest_voxels_10cm.npz   --num-workers 12   --worker-chunksize 1   --progress-interval 100   --collision-backend cpu   --collision-threads 2   --beam-depth 3   --beam-width 8   --beam-branching 4   --beam-discount 0.95

python3 scripts/generate_depth_action_masks.py   --index "$ROLL/rollout_index.csv"   --out-masks "$ROLL/depth_action_masks.npz"   --num-workers 12   --progress-interval 100   --collision-radius-m 0.40   --depth-slack-m 0.08   --path-sample-stride 4   --max-patch-radius-px 14

python3 scripts/audit_teacher_dataset.py   --index "$ROLL/rollout_index.csv"   --labels "$ROLL/teacher_labels.npz"   --out-json "$ROLL/labeled_audit.json"
```

当前 60,011 条数据的 CPU 标签阶段约耗时 3.67 小时；这是实际历史值，不是所有机器的保证。

### 7.5 mmap 与 BC 训练

```bash
python3 scripts/build_bc_mmap_dataset.py   --index "$ROLL/rollout_index.csv"   --labels "$ROLL/teacher_labels.npz"   --depth-action-masks "$ROLL/depth_action_masks.npz"   --out-dir "$BC/mmap"   --overwrite

python3 scripts/train_soft_bc.py   --index "$ROLL/rollout_index.csv"   --labels "$ROLL/teacher_labels.npz"   --depth-action-masks "$ROLL/depth_action_masks.npz"   --dataset-cache "$BC/mmap"   --out-dir "$BC/model"   --epochs 60   --batch-size 128   --lr 3e-4   --weight-decay 1e-4   --val-ratio 0.20   --seed 2026   --device cuda   --num-workers 8   --prefetch-factor 4   --cpu-threads 24   --depth-history-frames 1   --ce-weight 1.0   --kl-weight 0.3   --ce-target teacher   --loss-mask height_depth   --deployment-safety-mask depth   --deployment-execution-mode continuous   --grad-clip 5.0   --save-plots   --tensorboard-log-dir "$BC/model/tensorboard"

tensorboard --logdir "$MODEL/tensorboard" --port 6006
```

所有新运行都应使用新目录。除非已另行归档并验证 SHA，否则不要 `--overwrite` 当前 6 万数据和模型。

## 8. 闭环评估与留出集管理

正式评估由 wrapper 独立启动一套 ROS/Unity，并在退出时清理；看到一个 Unity 窗口是正常现象。

```bash
BC=data/bc/flight_20260717_6w/model/checkpoint_best.pt
DEV=data/smoke/bc_flight_20260717_6w/holdout_seed55/fixed_dev_100.csv
EVAL=data/smoke/handoff_bc_dev100

bash scripts/evaluate_policy_unity_managed.sh \
  "$BC" "$DEV" "$EVAL" 100
```

同一个输出目录不会被静默覆盖；要做重复性确认，必须使用新的输出目录。比较模型时必须使用完全相同、顺序相同的 mission index，并检查 summary 中的 checkpoint 和 mission SHA。

### 8.1 Evaluation reproducibility 当前状态

deterministic primitive 的真实 RED→GREEN 证据：

```bash
pytest -q \
  tests/test_primitive_execution_protocol_contract.py \
  tests/test_primitive_execution_record_contract.py

PLANNING_TEST_UNITY=1 pytest -q \
  tests/test_deterministic_primitive_execution_unity.py
```

结果分别为 `13 passed` 和 `5 passed`。episode 451/action 52 的 5 次 execution 均为 25 effective ticks，endpoint delta 为 0，primitive-after-0 fingerprint 只有 1 种。

随后对同一 BC checkpoint、同一 episode 451 做了 20 次隔离 ROS/Unity 的完整轨迹评估：

```bash
bash scripts/run_policy_eval_repro.sh \
  data/bc/flight_20260717_6w/model/checkpoint_best.pt \
  data/smoke/bc_flight_20260717_6w/holdout_seed55/fixed_dev_100.csv \
  /tmp/formal_eval_repro_20260809/layer1_episode451 \
  451 20
```

行为内容证据：

- 20/20 success，全部为 29 primitives；
- complete action sequence、state/depth/action-mask/logits/top-k trajectory 完全一致；
- 580 个 primitive 全部为 frame 0..24、state offset 0..24、endpoint offset 24；
- final position 完全一致，final distance 全部为 `0.3890925347805023 m`；
- 190 个 pairwise comparison 中 content mismatch 为 0。

但严格 comparator 为 RED：190/190 pairs 都存在 timing-only divergence，最早发生在 reset/step -1/category F。20 次 reset sensor skew 为 47.691～70.961 ms，跨度 23.270 ms；state/depth pixels、position、velocity、yaw 均相同。当前 reset 在 settle 后返回首个满足 callback freshness 与 80 ms sync budget 的 observation，没有绑定唯一 Unity state/depth capture boundary。

快速复核命令：

```bash
python3 scripts/diagnose_policy_eval_divergence.py \
  --root /tmp/formal_eval_repro_20260809/layer1_episode451 \
  --min-repeats 20
```

按既定门控，发现 Layer 1 divergence 后没有进入 Layer 2（20 missions × 3 repeats）和 Layer 3（完整 dev100 × 3）。因此不得把“episode 451 行为内容零噪声”扩展解读为“formal dev100 已可重复”，也不得在当前证据上比较 BC incumbent 与 RL candidate。

生成新的 dev100/final100 时采用原子双选择，并排除所有要比较的 BC 数据：

```bash
HOLDOUT=data/smoke/new_holdout_seed55

python3 scripts/build_awac_final_holdout.py \
  --source-index data/teach/flight_20260717_6w/missions.csv \
  --bc-train-index data/teach/flight_20260717/rollouts/rollout_index.csv \
  --bc-train-index data/teach/flight_20260717_6w/rollouts/rollout_index.csv \
  --out-dev-index "$HOLDOUT/fixed_dev_100.csv" \
  --dev-count 100 \
  --count 100 \
  --seed 55 \
  --out-index "$HOLDOUT/final_holdout_100.csv" \
  --out-selection "$HOLDOUT/holdout_selection.json"
```

mission 已通过专家审计后，从该 audited index 做确定性留出切分不需要再次运行专家审计；必须验证 selection JSON 中三个 intersection 都为 0。

## 9. AWAC 入口、恢复和监控

当前不建议继续现有 18k 实验。若后续有新的、证据支持的 AWAC 假设，应创建新输出目录，从当前 BC 重新开始：

```bash
BC=data/bc/flight_20260717_6w/model/checkpoint_best.pt
TRAIN=data/teach/flight_20260717_6w/awac_split_seed55/train_missions.csv
SPLIT=data/teach/flight_20260717_6w/awac_split_seed55/split_manifest.json
OUT=data/awac/flight_20260717_6w/awac_new_hypothesis

python scripts/train_awac.py \
  --bc-checkpoint "$BC" \
  --train-index "$TRAIN" \
  --split-manifest "$SPLIT" \
  --out-dir "$OUT" \
  --env-workers 20
```

同一命令和同一输出目录可从完整 checkpoint 恢复。每 6k 步执行全 Replay KL 审计；失败时只回滚 Actor/Actor optimizer，保留 Critic、Replay 和时间线。下一段采集使用最近被接受的策略，不会自动使用被 dev 拒绝的 candidate。

TensorBoard：

```bash
tensorboard --logdir "$OUT/tensorboard" --port 6006
```

训练日志包含 `rollout/episode_return_raw`、`rollout/episode_return_scaled`、逐回合 success/collision/dead_end/timeout，以及 `awac/*`、collection 和性能指标。dev 曲线位于 `tensorboard/dev`。不应只看 reward 是否单调，应同时看固定 dev 的成功、碰撞、dead-end 和两次选择一致性。

`RUN_FINAL_HOLDOUT` 默认是 0。只有候选经过两阶段 dev 选择且开发工作冻结后，才允许单次设置 `RUN_FINAL_HOLDOUT=1`；final 结果不得继续用于调参。

## 10. 测试与健康检查

CPU/unit 测试：

```bash
conda activate xm
cd /home/xm/XM/xm_ws/src/planning
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash
pytest -m unit -q
```

本次 deterministic protocol 修复后的完整 CPU/default suite 结果为 `286 passed, 14 skipped`（43.26 s）；另外显式启用真实 Unity 的 deterministic execution suite 为 `5 passed`（44.10 s）。14 个 skip 是需要显式 ROS/CUDA/Unity 前提的测试，不能误报为已覆盖。如果未 source devel 空间，`planning.msg` 不在 Python path 中，pytest 会在收集阶段失败；这不是可以忽略的测试失败，也不要通过修改 import 绕过。

按模块快速检查：

```bash
pytest -q \
  tests/test_awac_core.py \
  tests/test_offpolicy_contract.py \
  tests/test_awac_checkpoint_migration.py \
  tests/test_awac_replay_audit.py \
  tests/test_awac_training_runtime_contract.py

python scripts/train_awac.py --help
python scripts/evaluate_policy_unity.py --help
python scripts/audit_awac_replay.py --help
python scripts/select_awac_dev_checkpoint.py --help
bash -n scripts/evaluate_policy_unity_managed.sh
```

测试 marker 在 `pyproject.toml` 中定义。`unity`、`ros`、`cuda` 和 `performance` 测试有外部前提，不要把未运行误报为通过。运动基元文件被 Git 忽略，合同测试会在需要时生成临时 fixture；正式运行仍需第 7.1 节生成的 artifact。

## 11. 日志、故障定位与安全停机

- 教师采集：查看 `$ROLL/logs/collector_N.log`、`collection_progress.json`、`collection_summary.json`；
- managed eval：查看输出目录下 `runtime_logs/roscore.log`、`bridge.log`、`unity.log`；
- AWAC：查看 `$OUT/runtime/step_*/logs/`、`learner/training.jsonl`、`guarded_state.json`、`trust/audits/` 和 `decisions/`；
- 终端统一进度字段包括 `progress`、`passing`、`rate`、`estimated_stop`、`eta_h`；
- 结构化日志字段包括 timestamp、run_id、mission_id、episode、step、component、duration_ms、memory_mb、gpu_memory_mb、status、error_type。

managed 脚本使用独立端口并负责清理。异常中止后如果端口被占用，先确认残留进程的归属，不要直接大范围 `pkill`。训练脚本在启动前会预检 worker 端口，任一冲突都会 fail-closed。

## 12. 已知风险与后续优先级

### P0：交接前完成

1. **恢复可追溯的 Git 历史。** 本次审计环境中的 `/home/xm/XM/xm_ws/src/.git` 为空，无法获得 commit、branch 或 dirty 状态。`package.xml` 的 `0.10.2` 不是 Git 证据。交接方应恢复原仓库或建立受控快照，提交代码、本文档和环境锁文件，并打 annotated tag；禁止虚构 commit SHA。
2. **分开归档并校验约 95 GB 核心数据。** 至少传输第 5.1 节资产；接收端重新计算 SHA256，并保存归档清单。
3. **保护 final100。** 只向接手人说明路径和 SHA，不运行它，不把它复制成新的 dev 集。

### P1：下一项值得做的研发工作

不要立刻扩大 BC 数据、延长 AWAC 或比较新的 RL candidate。第一优先级是完成 formal evaluation reproducibility：

1. 继续诊断 reset timing-only F divergence，定义 state/depth capture boundary，而不是用 arbitrary sleep、放宽 comparator tolerance 或忽略 timing 字段；
2. episode 451 的 20-repeat 严格 GREEN 后，执行至少 20 个固定 missions × 3 repeats，并逐 episode 比较 outcome/action/trajectory；
3. Layer 2 GREEN 后，再对同一 fixed_dev_100 做 3 次完整隔离评估和 episode-by-episode diff；
4. 只有 Layer 3 给出可接受的 noise floor，才能恢复 BC incumbent 与 RL candidate 的 guarded comparison。

evaluation reproducibility 完成后，再在同一 dev100 上对 BC 碰撞进行逐 mission 诊断，区分：

- 深度 mask 漏检（危险动作仍被判为有效）；
- 策略在有效动作中选错；
- 深度/状态时间偏差；
- 连续基元衔接导致的动力学偏差。

只有统计出主要类别和占比后，才决定修改 mask、训练目标、观测同步或在线算法。每项修改都应绑定同一 dev mission 的前后对照和 checkpoint/config SHA。

### P2：若再次尝试 AWAC

- 从 6 万 BC 新目录开始；
- 仍按 6k/12k/18k 分段；
- 保持 full-depth action domain、完整 Replay KL 审计和 0.10 hard budget；
- primary 与 confirmation 必须一致；
- 不能用单次 success 上升抵消 collision/dead-end 退化；
- 没有稳定 dev 改善前不得开启 final100，也不得长跑到 60k/500k。

## 13. 接手验收清单

- [ ] 能在 `xm` 环境中完成 Release catkin 构建；
- [ ] Unity schema v3 source、player、bridge 和 Python validator 成套存在，SHA 与第 2.1 节一致；
- [ ] motion primitive 和 task contract SHA 与本文一致；
- [ ] 60,011 条 rollout、labels、depth masks、mmap 均可读取；
- [ ] 当前 BC checkpoint SHA 为 `188e4130…c31f7`；
- [ ] dev100/final100 SHA 正确且三方 mission intersection 为 0；
- [ ] 能在 dev100 上完成一次 managed eval，summary 身份校验通过；
- [ ] 理解 episode 451 的行为内容已可重复，但 reset timing comparator 仍为 RED，Layer 2/3 尚未执行；
- [ ] 能运行 unit/合同测试，并记录实际 pass/skip/fail；
- [ ] 理解 AWAC 18k 未晋级，accepted checkpoint 仍是 BC；
- [ ] 已恢复 Git 可追溯性并完成独立数据归档校验；
- [ ] final100 仍未使用。
