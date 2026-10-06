# CMPE 260 Project 1: Teaching DQN to Play Pong

Team: Jaya Vyas, Prachi Gupta, Keerthana Muralidharan, Yashaswini Dinesh

Basic DQN from the textbook repo (Lapan, *Deep Reinforcement Learning Hands-On*, Chapter 6),
ported to Gymnasium 1.x, plus two exploration methods (step 2) and Prioritized Experience Replay (step 3).
All runs: Google Colab, NVIDIA Tesla T4, one run per configuration.

## Results (frames and minutes to a mean reward of 19 over the last 100 games)

| Method | Buffer | Minutes | Frames | Fewer frames vs baseline |
|---|---|---|---|---|
| ε-greedy (baseline) | 10k | 138.8 | 862k | – |
| Boltzmann | 10k | 86.5 | 550k | 36% |
| Max-Boltzmann | 10k | 75.8 | 492k | 43% |
| Max-Boltzmann + PER | 10k | 101.4 | 514k | 40% |
| Max-Boltzmann | 50k | 129.6 | 638k | 26% |
| Max-Boltzmann + PER | 50k | 91.7 | 448k | 48% |

PER vs no PER at the same buffer: no gain at 10k, 30% fewer frames and 29% less time at 50k.
Minutes compare fairly only within one buffer size (the 50k buffer runs about 20% slower per frame).

## How to present / rerun

Open `notebooks/CMPE260_Project1_Final.ipynb` in Colab with this folder on Drive at
`MyDrive/CMPE260_Project1` (or change `DRIVE_FOLDER` in the first cell). Run the cells top to bottom.
Training is off by default (`RUN_TRAINING = False`); results load from the saved logs and tables.
The last cell zips everything for download.

## Folder layout

| Path | Contents |
|---|---|
| `02_dqn_pong.py` | Step 1 baseline: textbook DQN with ε-greedy |
| `03_dqn_play.py` | Plays a greedy test game with a saved model |
| `04_dqn_pong_boltzmann.py`, `05_dqn_pong_maxboltzmann.py` | Step 2 exploration methods |
| `06_dqn_pong_per.py` | Step 3 PER (`--explore egreedy/boltzmann/maxboltz`) |
| `compare.py` | Results tables and reward plots from TensorBoard logs |
| `lib/` | Atari wrappers (Gymnasium port) and the DQN network (unchanged) |
| `CHANGES_vs_repo.diff` | Every change relative to the textbook repo |
| `notebooks/` | The collated notebook; `archive/` keeps the original 50k follow-up notebook |
| `results/` | Results tables, plots and milestone data for both buffer sizes |
| `runs/` | TensorBoard event files (50k runs included; 10k runs copied in by the notebook) |
| `logs/`, `models/` | Console logs and best models of the 50k runs |

Scripts default to the textbook 10k buffer; the notebook switches 05 and 06 to 50k for the follow-up.
