| Method              | Runs reaching target   |   Minutes to 19 |   Frames to 19 (k) |   Best mean-100 reward | Speed-up vs baseline   | Fewer frames vs baseline   |
|:--------------------|:-----------------------|----------------:|-------------------:|-----------------------:|:-----------------------|:---------------------------|
| epsilon-greedy      | 1/1                    |           138.8 |                862 |                  19.36 | +0.0%                  | +0.0%                      |
| Boltzmann           | 1/1                    |            86.5 |                550 |                  19.04 | +37.7%                 | +36.1%                     |
| Max-Boltzmann       | 1/1                    |            75.8 |                492 |                  19.01 | +45.4%                 | +42.9%                     |
| Max-Boltzmann + PER | 1/1                    |           101.4 |                514 |                  19.05 | +26.9%                 | +40.3%                     |
