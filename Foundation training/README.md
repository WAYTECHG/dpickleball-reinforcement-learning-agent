# Foundation training

This folder contains the cleaned wall warm-up code for the dPickleBall agent.

The purpose of this stage is to train the agent in a lightweight Python wall simulator before moving to real Unity or league training. The agent learns paddle movement, ball chasing, hitting, wall returns, recovery, and stronger returns through a 6-level curriculum.

## Workflow

Training and visualisation are separated. Run training first. Visualisation is optional and can be opened in another terminal.

### Terminal 1: training

```powershell
conda activate dpickleball
cd "C:\Users\User\dpickleball-ml-agents\Foundation training"
python run_train.py
```

This launches:

```powershell
python train_python_wall_ppo.py --num-envs 8 --obs-source state --fresh
```

To continue from `checkpoints/latest.zip`:

```powershell
python run_train.py --continue-training
```

### Terminal 2: optional visual check

```powershell
conda activate dpickleball
cd "C:\Users\User\dpickleball-ml-agents\Foundation training"
python run_visual_check.py --mode env --level 1 --steps 2000
```

To watch the latest saved policy:

```powershell
python run_visual_check.py --mode latest-policy --level 1 --steps 2000
```

## Main files

| File | Purpose |
|---|---|
| `run_train.py` | Simple launcher for foundation PPO training. |
| `run_visual_check.py` | Optional visual checker. |
| `train_python_wall_ppo.py` | Main PPO wall training loop. |
| `envs/python_wall_env.py` | Wall simulator environment and reward logic. |
| `envs/court_renderer.py` | OpenCV renderer for the wall simulator. |
| `envs/opencv_state_extractor.py` | Converts the rendered frame into compact features. |
| `curriculum/wall_levels.py` | Defines the 6 wall curriculum levels. |
| `curriculum/curriculum_manager.py` | Controls level advancement. |
| `curriculum/checkpoint_manager.py` | Saves latest, best checkpoints, and exported `.pt` policies. |
| `tools/debug_render_env.py` | Visual check for the standalone environment. |
| `tools/watch_latest_policy.py` | Visualises the latest saved PPO checkpoint. |

## Six-level curriculum

| Level | Name | Purpose |
|---|---|---|
| 1 | `slow_90deg` | Basic slow-ball return foundation. |
| 2 | `fast_90deg` | Faster reaction training. |
| 3 | `very_slow_150deg` | Wide-angle chase training. |
| 4 | `medium_fast_150deg` | Medium-fast wide-angle returns. |
| 5 | `mixed_speed_150deg_final` | Mixed-speed foundation training. |
| 6 | `level6_smash_finetune_150deg` | Smash and fast-return fine-tuning. |

## Important checkpoints

| File | Meaning |
|---|---|
| `checkpoints/latest.zip` | Latest SB3 PPO checkpoint for continuing training. |
| `checkpoints/best_1.zip` | Best saved PPO checkpoint. |
| `checkpoints/best_2.zip` | Second-best saved PPO checkpoint. |
| `checkpoints/best_3.zip` | Third-best saved PPO checkpoint. |
| `checkpoints/*_policy.pt` | TorchScript policy export for deployment or testing. |

## Notes

- No `.cmd` files are included.
- Code comments were simplified and old version notes were removed.
- Training and visual checking are separate Python workflows.
- Press `q` in the visual window to close the viewer.
