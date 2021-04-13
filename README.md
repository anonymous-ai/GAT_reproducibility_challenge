# Implementation for Graph Attention Networks: Oxford's ATML Reproducibility Challenge

All codes for the network layers are placed in models/layers.py

Code for running experiments should be in each corresponding folder.

## Results
1. Tasks
    - Node Classification: Cora, CiteSeer, PubMed, PPI 
    - Graph Classification: Coauthor CS (CCS)
    - Link Prediction: TSP
    - Graph Regression: QM9
2. Evaluation Metrics
    - Cora, CiteSeer, PubMed, CCS: Accuracy
    - PPI, TSP: F1 score
    - QM9: MSE
3. Recorded values are (ours / reported) where reported values are originated from
    - CiteSeer: [Velickovic et al., 2017](https://arxiv.org/abs/1710.10903)
    - Coauthor CS: [Shchur et al., 2018](https://arxiv.org/abs/1811.05868)


|              | CiteSeer | CCS | PubMed | Cora | QM9 [1] | PPI | TSP |
| :--------: | :------------------: | :-----------------: | --------------------- | --------------- | -------------------- | --------------- | --------------- |
|     GCN      | 69.2±0.9 / 70.9±0.5  |    91.3±0.5 / -     | 78.6±0.4 / -           | 80.9±0.3        | 0.1161±0.0011        | 0.852±0.012 | 0.5485 |
| ConstGAT  |     70.5±0.6 / -     |    89.6±0.6 / -     | 78.7±0.4 / -           | 80.7±0.6        | 0.1260±0.0010        | 0.805±0.015 | 0.5359 |
|     GAT      | 70.3±1.1 / 72.5±0.7  | 89.0±0.7 / 90.5±0.6 | 78.6±0.4 / -      | 80.9±0.8        | 0.1235±0.0015        | 0.951±0.004 | 0.6137 |
| DotGAT |     70.6±0.7 / -     |    89.6±0.6 / -     | 78.8±0.4/ -           | 81.0±0.6        | 0.1132±0.0021        | 0.980±0.004 | 0.7265 |

[1] QM9 dataset, predicting the first task (Dipole moment), using 1000 graph as training, 500 as validation and 1000 as testing.
