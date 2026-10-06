# CMPE 260 Project 1: submission contents

- Code: `02`–`06` training scripts, `03_dqn_play.py`, `compare.py`, `lib/`, `requirements.txt`.
  Scripts default to the textbook 10k replay buffer; the notebook sets 50k for the follow-up runs.
- `CHANGES_vs_repo.diff`: every change relative to the textbook repo's Chapter 6 code.
- `notebooks/CMPE260_Project1_Final.ipynb`: all experiments, tables, TensorBoard and plots in one notebook.
  `notebooks/archive/followup_50k_original.ipynb` is the notebook the 50k runs were trained in.
- `results/buffer_10k/`, `results/buffer_50k/`: `compare.py` tables and reward plots per buffer size.
- `results/milestones.csv`: frames and minutes to mean reward 0, 15 and 19 for every run.
- `results/buffer_10k/per_on_baseline_partial.csv`: the ε-greedy + PER check, stopped at 786k frames by a disconnect.
- `runs/buffer_50k/`: TensorBoard event files of the 50k runs. The 10k runs' event files stay in the original
  Drive folder and are copied into `runs/buffer_10k/` by the notebook.
- `logs/`, `models/`: console logs and best saved models of the 50k runs.

One run per configuration on a Colab Tesla T4.
