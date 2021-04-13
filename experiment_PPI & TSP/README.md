# PPI

1. train & test with GAT:

python run.py --residual --bsz 2 --epoch 5000 --patience 100 --lr 5e-3 --wd 0 --dp 0 --mode 'train' --name 'ppi'

python run.py --load-model 'Log/ppi.dict' --residual --bsz 2 --mode 'test' --name 'ppi'

2. train & test with DotGAT:

python run.py --dataset 'PPI' --att 'transformer' --residual --bsz 2 --epoch 5000 --patience 100 --lr 5e-4 --wd 0 --dp 0 --mode 'train' --name 'ppi_trans'

python run.py --dataset 'PPI' --load-model 'Log/ppi_trans.dict' --att 'transformer' --residual --bsz 2 --mode 'test' --name 'ppi_trans'

3. train & test with ConstGAT:

python run.py --dataset 'PPI' --att 'const' --residual --bsz 2 --epoch 5000 --patience 100 --lr 5e-3 --wd 0 --dp 0 --mode 'train' --name 'ppi_const'

python run.py --dataset 'PPI' --load-model 'Log/ppi_const.dict' --att 'const' --residual --bsz 2 --mode 'test' --name 'ppi_const'

4. train & test with GCN:

python run.py --dataset 'PPI' --att 'gcn' --residual --bsz 2 --epoch 5000 --patience 100 --lr 5e-3 --wd 0 --dp 0 --mode 'train' --name 'ppi_gcn'

python run.py --dataset 'PPI' --load-model 'Log/ppi_gcn.dict' --att 'gcn' --residual --bsz 2 --mode 'test' --name 'ppi_gcn'


# TSP

1. train & test with GAT:

python run.py --dataset 'TSP' --att 'gat' --residual --nblocks 3 --bsz 16 --epoch 200 --patience 10 --lr 5e-4 --wd 0 --dp 0 --mode 'train' --name 'tsp_gat_3b'

python run.py --dataset 'TSP' --load-model 'Log/tsp_gat_3b.dict' --att 'gat' --residual --nblocks 3 --bsz 16 --mode 'test' --name 'tsp_gat_3b'

2. train & test with DotGAT:

python run.py --dataset 'TSP' --att 'transformer' --residual --nblocks 3 --bsz 16 --epoch 200 --patience 10 --lr 5e-4 --wd 0 --dp 0 --mode 'train' --name 'tsp_dot_3b_lr4'

python run.py --dataset 'TSP' --load-model 'Log/tsp_dot_3b_lr4.dict' --att 'transformer' --residual --nblocks 3 --bsz 16 --mode 'test' --name 'tsp_dot_3b_lr4'

3. train & test with ConstGAT:

python run.py --dataset 'TSP' --att 'const' --residual --nblocks 3 --bsz 16 --epoch 200 --patience 10 --lr 5e-4 --wd 0 --dp 0 --mode 'train' --name 'tsp_const_3b'

python run.py --dataset 'TSP' --load-model 'Log/tsp_const_3b.dict' --att 'const' --residual --nblocks 3 --bsz 16 --mode 'test' --name 'tsp_const_3b'

4. train & test with GCN:

python run.py --dataset 'TSP' --att 'gcn' --residual --nblocks 3 --bsz 16 --epoch 200 --patience 10 --lr 5e-4 --wd 0 --dp 0 --mode 'train' --name 'tsp_gcn_3b'

python run.py --dataset 'TSP' --load-model 'Log/tsp_gcn_3b.dict' --att 'gcn' --residual --nblocks 3 --bsz 16 --mode 'test' --name 'tsp_gcn_3b'





