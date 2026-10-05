# CMPE 260 Project 1: basic DQN on Pong, straight from the textbook repo

Source: https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On, `Chapter06/`.
The algorithm is the repo's basic DQN (replay buffer + target network) with no specialised
variants (no Double/Dueling/Noisy/n-step/Rainbow). Every change is listed in `CHANGES_vs_repo.diff`.

## Files
| File | Assignment step | What it is |
|---|---|---|
| `lib/wrappers.py`, `lib/dqn_model.py` | all | Repo files. Wrappers ported to Gymnasium; model **unchanged**. |
| `02_dqn_pong.py` | 1: baseline | Repo script, Gymnasium API fixes only. |
| `03_dqn_play.py` | eval | Repo script: plays one game with a saved model. |
| `04_dqn_pong_boltzmann.py` | 2 | Copy of `02`; epsilon-greedy replaced by Boltzmann (softmax) action selection. |
| `05_dqn_pong_maxboltzmann.py` | 2 | Copy of `02`; the random action in epsilon-greedy is drawn from softmax(Q/τ). |
| `06_dqn_pong_per.py` | 3 | Copy of `02` + PER buffer and weighted loss from the repo's `Chapter07/05_dqn_prio_replay.py`. `--explore` picks the step-2 method. |
| `compare.py` | report | Reads `runs/` TensorBoard logs → results table and reward-vs-time / reward-vs-frames plots. |
| `CHANGES_vs_repo.diff` | report | Exact diff of every file against the repo (or, for 04–06, against the ported 02). |

## Changes needed for current Gymnasium/PyTorch/NumPy (baseline)
- `gym` → `gymnasium` and `ale_py` env registration.
- `reset()` returns `(obs, info)`; `step()` returns `(obs, r, terminated, truncated, info)`, with `done = terminated or truncated`.
- `np.array(..., copy=False)` → `np.array(...)`, which NumPy 2 requires.
- `torch.ByteTensor` mask → `torch.BoolTensor`. Current PyTorch rejects uint8 masks.
- The "Solved" message also prints elapsed minutes.

## Run (Colab, GPU runtime)
```bash
pip install -r requirements.txt
python 02_dqn_pong.py --cuda --reward 20                              # step 1 (ran past 19)
python 04_dqn_pong_boltzmann.py --cuda --reward 19                    # step 2
python 05_dqn_pong_maxboltzmann.py --cuda --reward 19                 # step 2
python 06_dqn_pong_per.py --cuda --reward 19 --explore maxboltz       # step 3 (step-2 winner)
tensorboard --logdir runs
python compare.py --logdir runs --target 19
python 03_dqn_play.py -m PongNoFrameskip-v4-best.dat --no-visualize
```
`--reward 19` matches the assignment; the repo's default is 19.5.
Tuning knobs for step 2: `--tau-start`, `--tau-final`, `--tau-frames`.

## Measuring convergence (assignment requirement)
In TensorBoard, set the horizontal axis to **Relative** and read when `reward_100` crosses 19.
`compare.py` computes the same number from the logs, plus frames to 19, which doesn't depend on hardware.
Record the GPU with `nvidia-smi`.

## Results (Colab Tesla T4, one run per method)
See `results_table.md`, `compare_frames.png` and `compare_time.png`.
