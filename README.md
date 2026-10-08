# 🏓 dPickleBall — Deep Reinforcement Learning Agent

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Unity](https://img.shields.io/badge/Unity_ML--Agents-222222?style=flat-square&logo=unity&logoColor=white)
![PPO](https://img.shields.io/badge/Reinforcement_Learning-PPO-2563EB?style=flat-square)
![Competition](https://img.shields.io/badge/Competition-2nd_of_13-F59E0B?style=flat-square)

A **PPO-based multi-agent reinforcement learning system** developed to train autonomous agents in the Unity dPickleBall environment using curriculum learning, reward shaping, and adaptive league self-play.

**Course:** AIT306 — Deep Reinforcement Learning  
**Institution:** Xiamen University Malaysia  
**Project:** Group 11 | April 2026 Semester  
**Achievement:** 🥈 2nd Place among 13 competing teams

[Competition Recording](https://www.youtube.com/live/CitFn9FK12s) · [Original Environment](https://github.com/dPickleball/dPickleBallEnv)

---

## Project Overview

This project explores reinforcement learning strategies for improving autonomous gameplay in a simulated pickleball environment. The agent learns three-dimensional discrete action control involving vertical movement, horizontal movement, and paddle rotation.

A major challenge in multi-agent reinforcement learning is training agents that initially lack fundamental gameplay skills. Direct self-play between untrained policies can produce ineffective interactions and limited learning signals.

To address this, the project introduces a **two-stage training framework**, combining foundational skill development with competitive policy optimization.

## Training Pipeline

```text
              dPickleBall Environment
                         │
                         ▼
                Foundation Training
               Six-Level Wall Curriculum
                         │
                         ▼
                 PPO + Reward Shaping
                         │
                         ▼
                  Trained Policies
                         │
                         ▼
                 League Self-Play
                Dynamic Opponent Pool
                         │
                         ▼
             PSRO / SFL-Inspired Selection
                         │
                         ▼
                Round-Robin Evaluation
                         │
                         ▼
                 Final Agent: RealFT
```

### 1. Foundation Wall Training

A six-level curriculum progressively develops essential gameplay behaviors in a simplified wall-based environment.

| Level | Ball Speed Range | Rebound Angle | Focus |
|---|---|---|---|
| 1 | 0.85–1.35 | ±45° | Basic tracking and contact |
| 2 | 2.35–3.05 | ±45° | Faster reactions |
| 3 | 0.35–0.80 | ±75° | Wide-angle interception |
| 4 | 1.85–2.35 | ±75° | Diverse medium-fast trajectories |
| 5 | 0.35–3.70 | ±75° | Mixed-speed adaptation |
| 6 | 0.80–3.70 | ±75° | Stronger controlled returns |

**Reward shaping** encourages effective movement, paddle alignment, successful returns, and recovery positioning. Penalties discourage passive behavior, poor contact, and ineffective positioning.

Different reward configurations were explored to produce agents with distinct playing behaviors and strengths.

### 2. League Self-Play Training

Following foundation training, the agents transition to competitive self-play within the Unity environment.

**Proximal Policy Optimization (PPO)** is used to improve action selection through repeated interactions and clipped policy updates.

The agent operates with a multidiscrete action space:

`MultiDiscrete([3, 3, 3])`

| Action | Available Controls |
|---|---|
| Vertical movement | None, Up, Down |
| Horizontal movement | None, Right, Left |
| Paddle rotation | None, Counter-clockwise, Clockwise |

An opponent pool containing independently trained agents, historical checkpoints, and previous learner policies supports diverse training experiences.

To avoid relying on a single opponent, the system incorporates:

- **PSRO-inspired selection:** Uses a population of policies for adaptive opponent sampling.
- **SFL-inspired frontier sampling:** Prioritizes opponents near the learner's current skill level.
- **Softmax-based sampling:** Balances opponent difficulty and selection diversity.

This design aims to improve policy robustness and adaptability against different gameplay strategies.

## Model Evaluation

Candidate agents were compared through internal round-robin tournaments, followed by a Top-4 confirmation stage.

| Rank | Agent | Series W–L | Game W–L | Point Difference |
|---|---|---|---|---|
| 1 | **RealFT** | **2–1** | **5–2** | **+20** |
| 2 | Haoren Original Lifted | 2–1 | 3–3 | −17 |
| 3 | Wilbert Blocker Converted | 1–2 | 4–4 | +1 |
| 4 | V67 Best 3 | 1–2 | 2–5 | −4 |

**RealFT** achieved the strongest overall internal tournament result and was selected as the final competition agent.

The final agent was then evaluated against agents developed by other participating groups.

**[Watch the competition on YouTube](https://www.youtube.com/live/CitFn9FK12s)**

## My Contributions

My responsibilities focused on independent reinforcement learning experimentation and collaborative agent evaluation:

- Trained and optimized an autonomous PPO-based agent.
- Experimented with reward and penalty configurations to improve gameplay behavior.
- Evaluated trained policies through iterative testing and performance comparison.
- Contributed candidate policies for team-wide evaluation.
- Participated in selecting the strongest agent for the final competition.

The training and final selection process was conducted collaboratively as part of the group project.

## Technology Stack

| Category | Technologies / Methods |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Simulation | Unity, Unity ML-Agents |
| RL Algorithm | Proximal Policy Optimization (PPO) |
| Training | Curriculum Learning, Reward Shaping, League Self-Play |
| Opponent Selection | PSRO-Style, SFL-Inspired Sampling |
| Evaluation | Round-Robin Tournament |

## Repository Structure

```text
dpickleball-reinforcement-learning-agent/
│
├── Competition Script/
│   └── Competition execution and evaluation
│
├── Foundation training/
│   └── Curriculum-based wall training
│
├── League Self-Play Training/
│   └── PPO training and opponent selection
│
└── README.md
```

## Installation and Execution

### Prerequisites

- Python 3.10.12
- Conda or Miniconda
- Compatible Unity ML-Agents dependencies
- Windows dPickleBall competition build
- Required trained agent checkpoints

### 1. Set Up the Environment

Create and activate a Python environment:

```bash
conda create -n dpickleball python=3.10.12 pip
conda activate dpickleball
```

Install the ML-Agents implementation and other dependencies following the [official dPickleBall environment instructions](https://github.com/dPickleball/dPickleBallEnv).

### 2. Clone the Repository

```bash
git clone https://github.com/WAYTECHG/dpickleball-reinforcement-learning-agent.git
cd dpickleball-reinforcement-learning-agent
```

### 3. Prepare the Unity Game

Obtain the compatible Windows competition build from the original dPickleBall project.

Locate the game executable:

```text
Competition/
└── Windows/
    └── dp.exe
```

Navigate to the directory containing `competition_like_fight.py` and ensure its required model checkpoints are available.

### 4. Run the Competition

Execute the following command, replacing the placeholder with the actual path to the Unity executable:

```powershell
python competition_like_fight.py --env-path "C:\path\to\Competition\Windows\dp.exe" --graphics
```

| Argument | Description |
|---|---|
| `competition_like_fight.py` | Competition simulation script |
| `--env-path` | Location of the Unity executable |
| `--graphics` | Enables graphical rendering |

**Note:** This command launches the competition simulation using existing trained policies. It does not initiate a new training session. A compatible environment build, dependencies, and agent checkpoints are required.

## References and Acknowledgements

Developed collaboratively for **AIT306 — Deep Reinforcement Learning** at Xiamen University Malaysia.

The project extends the original dPickleBall simulation through reinforcement learning experimentation and agent training.

**Source Repositories**
- [Original dPickleBall Environment](https://github.com/dPickleball/dPickleBallEnv)
