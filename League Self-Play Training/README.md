# League Self-Play Training Core Full Members

This folder contains the cleaned **League Self-Play Training** stage for the dPickleBall agent.

The purpose of this stage is to continue training the policy in the real Unity dPickleBall environment using PPO, a league opponent pool, and self-play checkpoint recycling. This folder does **not** include tournament or ranking code. It only keeps the core training workflow:

1. **Fast training:** 800k steps at `time_scale=2`.
2. **Normal fine-tuning:** 200k steps at `time_scale=1`.

## Opponent pools

| Folder | Meaning |
|---|---|
| `league_pool_filtered/` | Default balanced runtime opponent pool used by the training scripts. It now contains **40 models**: weak, medium, member-contributed, and strong opponents. |
| `league_pool_all_submitted/` | Full submitted model archive with **211 models**, including Zhiling, Haoren, Huayang, Jason, Wilbert, and WW/V-series legacy models. This is kept for reproducibility and report evidence. |

The default training scripts use `league_pool_filtered/` because it is more balanced and less repetitive than the full archive. The full archive is still included so the report can state that the league opponent pool was prepared from all member-contributed models.

## Main commands

Run from this folder:

```powershell
conda activate dpickleball
cd "C:\Users\User\dpickleball-ml-agents\League Self-Play Training Core Full Members"
```

### Phase 1: fast 800k league training

```powershell
python run_fast_training.py
```

This launches the 6 learner starts and trains them using:

```text
steps = 800000
chunk_steps = 20000
time_scale = 2
max_parallel = 6
trainer_pool = league_pool_filtered
```

### Phase 2: normal 200k fine-tuning

After fast training finishes, run:

```powershell
python run_normal_finetune.py
```

This resumes from the fast 800k checkpoints and fine-tunes using:

```text
steps = 200000
chunk_steps = 40000
time_scale = 1
max_parallel = 6
trainer_pool = league_pool_filtered
```

## Optional: train with the full submitted pool

The balanced 40-model runtime pool is recommended. If you need to train with the full archive pool for experimentation, run:

```powershell
python run_fast_training.py --trainer-pool league_pool_all_submitted
python run_normal_finetune.py --trainer-pool league_pool_all_submitted
```

## Balanced 40-model runtime pool design

The default runtime pool intentionally mixes different opponent strengths and styles:

| Group | Purpose |
|---|---|
| V3 / V31 / V33 / base policies | Older or weaker opponents for easier early learning and diversity. |
| Zhiling, Haoren, Huayang, Jason, Wilbert models | Member-contributed policies at different training stages. |
| Wilbert blocker models | Defensive/blocking opponent behaviour. |
| V54 / V61 / V67 / V82 / V84 policies | Stronger opponents for robustness and harder training pressure. |

This gives the learner a curriculum-like range of opponents instead of only very strong policies.

## Main files

| File / folder | Purpose |
|---|---|
| `run_fast_training.py` | Python launcher for the 800k fast league training phase. |
| `run_normal_finetune.py` | Python launcher for the 200k normal-speed fine-tuning phase. |
| `run_6_learners_parallel.py` | Runs the 6 learners in parallel. |
| `train_real_league_ppo.py` | Main PPO training script for one learner. |
| `real_dpickleball_env.py` | Real Unity environment wrapper and reward handling. |
| `opencv_state_extractor.py` | Converts visual observation into 16-feature state input. |
| `league_selector.py` | Selects opponents from the league pool and self-play checkpoints. |
| `export_sb3_to_torchscript.py` | Exports SB3 PPO checkpoints to TorchScript `.pt` policies. |
| `learner_starts/` | Initial PPO checkpoints for the 6 learners. |
| `league_pool_filtered/` | Balanced 40-model runtime opponent pool. |
| `league_pool_all_submitted/` | Full 211-model submitted archive. |
| `manifests/` | CSV/JSON summaries of both pools. |

## Report wording

You can describe this stage like this:

> The real-environment training stage used a league self-play PPO setup. The opponent pool was prepared from all group member contributions, including Zhiling, Haoren, Huayang, Jason, Wilbert, and earlier WW/V-series policies. A balanced 40-model runtime pool was used for training, containing weak, medium, defensive, and strong opponents. During training, new self-play checkpoints were also added back into the pool, allowing the learner to face both fixed historical policies and its own previous versions.

## Notes

- The default pool is `league_pool_filtered/`.
- The full contributed pool is preserved in `league_pool_all_submitted/`.
- Use `outputs/` for training results and exported `.pt` files.
