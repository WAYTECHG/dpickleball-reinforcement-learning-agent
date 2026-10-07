# Wall_sim_V82_realft(v3 混合部署)—— 本分支真机微调版本(精简包)

本分支(realft 谱系)的完整工作版本。训练版本与 v2 同源(产品档 `realft_01600004`);
**本包的部署件 `deploy/` 是 v3 混合最优**:左右两侧各装"赢下该侧真机门"的检查点。
精简包:已去掉 57 个中间训练 `.zip`,保留代码、关键档、**全部可部署 `policy.pt`**、对手池、真机门脚本与战绩、部署件。

## 本包部署件 = v3 混合(deploy/)
| 槽位 | 模型 | 真机门(vs router,serve 212) |
|---|---|---|
| 左 teamX | `realft_01600004`(1.6M PFSP 微调) | **4/4,平均 21:9.8** |
| 右 teamY | `realft_00300000` 镜像(300k 基线镜像) | **3/3,平均 21:13.3** |

合计 **7 场全胜零败**,逐槽取各侧真机最强(优于全 1600k 的 v2、全 realft 的 v1)。teamX/teamY 各自独立加载 model_left/right.pt,故可混搭。

## 目录说明
| 路径 | 内容 |
|---|---|
| `realft_env.py` | 真机对战环境(learner 左,对手池每局轮换;**PFSP 加权**:学习度 wr(1-wr)、软地板永不淘汰、router 保底) |
| `realft_train.py` | PPO 微调(`--pfsp`、`--opponent-pool-dir`、`--continue-timesteps`) |
| `run_overnight_watchdog.py` | 过夜看门狗(keep-awake + 卡死自愈 + 续训;写 PID/心跳/done) |
| `_pfsp_offline_test.py` | PFSP 采样分布离线验证 |
| `checkpoints/realft_01600004.zip` | 产品档(SB3,可续训) |
| `checkpoints/r2/realft_00300000.zip` | 300k 基线(realft 起点,= v3 右拍来源) |
| `checkpoints/*_policy.pt` | 每 2.5 万步导出的 TorchScript 策略(可部署 / 可 gate),旋转恒 0,13 维角度盲 |
| `opp_pool/` | PFSP 对手池 5 个(全右原生) |
| `logs/` | 训练 ckpt_log + 早期真机对打记录 |
| `gate/` | 真机门:`realunity_gate.py` / `batch_gate.py` / `multi_match*.py` / `run_gate_pfsp.bat`;`results/` = 本次 gate 战绩 |
| `tools/` | `distill_v54_student.py`(realft_train 依赖)、`export_mirror_right.py`(造右镜像) |
| `deploy/` | **可直接交赛的 v3 混合部署件**(左 1600k + 右 300k 镜像 + 说明) |

## 真机战绩(都 vs router,真机门,serve 212=部署条件,旋转 0)
| 档 | 左侧 | 右侧(镜像) |
|---|---|---|
| realft 300k(基线) | 3/4 @ 20.8:16.2 | **3/3 @ 21:13.3** ← v3 右拍 |
| 1600k | **4/4 @ 21:9.8** ← v3 左拍 | 2/3 @ 19.7:15.7 |
- v3 = 取每侧更强者 → 左 1600k + 右 300k 镜像 = **7/7**。
- C-2.9M(组长 sim-SFL 碰运气档):sim 虐 V54 21:4,真机 PK 输给 realft → sim 胜≠真机胜的活样本。

## 怎么用
- **直接交赛/试跑**:用 `deploy/` 整个文件夹(4 文件放进 Competition.py 旁边即可,CPU 跑,依赖 torch/numpy/opencv;teamX/Y 已内嵌特征提取器,**部署无需外部依赖**)。
- **续训**:`python run_overnight_watchdog.py`(从 `checkpoints/` 最新档续,PFSP 默认开)。
- **真机门选档**:`gate/batch_gate.py` 把 `checkpoints/*_policy.pt` 逐个 vs router 打、排名。

## 外部依赖(精简包未内附,需谱系共享文件;组长那边应已有)
> 仅**续训/gate** 需要;**部署 deploy/ 不需要**(已自包含)。
- OpenCV 提取器:`Wall_sim_V67_FT\envs\opencv_state_extractor.py`(realft_env 与 gate 脚本 import)
- 蒸馏 student:`Wall_sim_V81_distill\checkpoints\student_net_state.pt`(仅从零重训需要;续训不需要)
- router 基准:`Wall_sim_V67_FT\pretrained\v54_v67_originals\best_2_policy.pt`(gate 对手)
- 竞赛构建:`dp.exe`(脚本里是绝对路径,按你环境改)

## 铁律 / 注意点
1. **只信真机门**:sim 胜率与真机脱钩,选档必须真 dp.exe 对打。
2. **旋转恒 0**(从训练起禁):sim 钳拍角、Unity 不钳,依赖旋转的模型真机会乱转。
3. **右侧 = 角度盲镜像**(`tools/export_mirror_right.py`,翻 idx 0/2/4/8/14 + 水平动作 (3-h)%3 + 旋转0,验零误差)。
4. **左右可混搭**:teamX/teamY 独立加载,v3 即左 1600k + 右 300k 镜像的最优组合。
5. **真机微调需活动显示会话**;长训用 WMI 脱离 + 看门狗,真重启仍会停(每 2.5 万步存档可续)。
