# Wall_sim_V82_realft — 真机微调（训练分布 = 部署分布）

## 这是什么
在真实 dp.exe 里直接微调 **V81 蒸馏学生**（13 维角度盲、旋转硬 0、已 Unity 验证 21:18 胜路由器）。
这是组长 config.py 写明但从未执行的"After pretraining, fine-tune in real Unity"一步。

## 结构性保证
- **观测**：官方 agent0 共享图 → OpenCVStateExtractor(left) → 16 维 → 13 维学生口径
  （删 sin/cos/success，time_own_side 恒 0）——和比赛部署逐位一致，没有任何测不准的特征。
- **动作**：MultiDiscrete([3,3])，旋转不在动作空间里——Unity 旋转崩坏模式被构造性移除。
- **奖励**：真实 ±1（左得分−右得分），无塑形奖励可骗。
- **对手**：右侧冻结 V67 路由器（比赛级锚），走同样的图像路径。
- **生命周期**：高赛点连续打（serve 213），每个得分作为软回合边界；
  比赛结束进程退出时自动重启（worker_id 自动递增防端口占用）。

## 防漂移设计
- 起点 = 蒸馏学生权重灌入 SB3 policy（灌入后 500 次随机输入逐位核对，必须 0 失配才开训）。
- `--critic-warmup 8000`：前 8k 步冻结 actor 只训 critic（新 critic 的垃圾 advantage 不许碰已验证的 actor）。
- lr 5e-5、clip 0.1、ent_coef 0（不鼓励无谓探索）、gamma 0.999（一分 300-600 步的信用分配）。
- 每 `--save-freq` 步导出 SB3 zip + 部署格式 .pt，打印该窗口的真实得分胜率。

## 跑法
```bash
conda activate dpickleball
cd D:\pickleball\Wall_sim_V82_realft
set OMP_NUM_THREADS=1
python realft_train.py --total-steps 50000 --save-freq 10000
```
注意：此构建真 headless 会挂（communicator 超时），默认带图形窗口跑；训练时会弹出游戏窗口属正常。

## 验收（每个 checkpoint）
1. 看训练日志里逐窗口 point_winrate 趋势（vs 冻结路由器，>0.5 即超过锚）。
2. `tools/frontal_anchor_diag.py --candidate-model checkpoints\realft_XXXX_policy.pt`（正面球不许掉）。
3. Unity 对打：`competition_like_fight.py --left-model checkpoints\realft_XXXX_policy.pt --right-model <V67路由器>`，肉眼行为 + 比分。
4. 任何一关不过 → 回退上一个 checkpoint；学生本体（realft_00000000_policy.pt）永远是保底。
