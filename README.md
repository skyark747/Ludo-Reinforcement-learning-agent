
### A reinforcement learning agent trained to play Ludo using an Actor–Critic architecture.

## Setting the environment

```
conda create LudoEnv python=3.12
conda activate LudoEnv
pip install requirements.txt
```
---
**Training**
- To train the model open ludo_train.ipynb and run all cells.

**Testing**
- To test the model run policy_crow_shadow.py. Change the path to .pth file in class policy_crow_shadow().

---

## Architecture

The agent consists of two separate neural networks:

**Actor Network**
- Input: state–action feature vector (8 features)
- Architecture: 256 → 256 → action logits
- Output: probability distribution over valid token–dice actions

**Critic Network**
- Input: critic state vector (11 features)
- Architecture: 256 → 256 → 1
- Output: state value estimate V(s)

---

## Feature Representation

**Actor features (state–action):**
- Is the token in danger
- Can capture an opponent token
- Is on a safe square
- Distance from home
- Is on home stretch
- Can enter the board
- Token progress

**Critic features (global state):**
- Red tokens at home, active, goal, safe
- Yellow tokens at home, active, goal, safe
- Whether a six was rolled
- Number of sixes rolled
- Sum of dice rolls

---

## Reward Shaping

| Event | Reward |
|---|---|
| Forward movement of a token | +1 |
| Moving a token out of the base | +10 |
| Landing on a safe square | +5 |
| Capturing an opponent's token | +30 |
| Own token being captured | -30 |
| Reaching the goal | +50 |
| Winning the game | +100 |
| Losing the game | -100 |

---

## Training

The agent was trained for **40,000 episodes** against a heuristic opponent :

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Initial exploration rate (ε) | 0.9 |
| ε decay | max(ε × 0.99995, 0.1) |
| Initial learning rate | 0.01 |
| LR schedule | Cosine annealing |
| Validation frequency | Every 1,000 episodes (vs random opponent) |

---

## Results

The Actor–Critic agent's win percentage was evaluated every 5,000 episodes. After the early exploration phase, win rate rose sharply and stabilized between **70–75%**, indicating policy convergence.

---

## Ablation Studies

Multiple RL algorithms were benchmarked using the same environment and reward structure:

| Policy | Win Rate vs Random (%) |
|---|---|
| SARSA | 51.0 |
| Q-learning | 53.0 |
| Dyna-Q | 55.0 |
| Linear TD Approximation | 57.0 |
| **Actor–Critic** | **77.0** |

The Actor–Critic agent significantly outperformed all alternatives, achieving a **77% win rate** against a random policy.
