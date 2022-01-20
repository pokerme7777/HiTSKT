# HiTSKT: A Hierarchical Transformer Model for Session-awared Knowledge Tracing

## Setup

The requiring environments is as bellow:

- Python 3.6+
- PyTorch 1.9.0
- Scikit-learn 0.24.2
- Numpy 1.19.5
- Pandas 1.1.5

## Run HiTSKT

We list the command to run the HiTSKT on different datasets. Listed hyperparameters are the optimal parameters for the respective datasets.

Running the HiTSKT model on the ASSISTment 2012 dataset:

```
python main.py --dataset='2012' --epoch_num=60 --batch_size=64 --session_size=8 --action_size=48 --embedding_size=128 --learning_rate=5e-5 --d_inner=1024 --n_layers=1 --n_head=2 --d_k=64 --d_v=64 --dropout=0.1 
```

Running the HiTSKT model on the ASSISTment 2017 dataset:

```
python main.py --dataset='2017' --epoch_num=100 --batch_size=64 --session_size=16 --action_size=64 --embedding_size=256 --learning_rate=5e-5 --d_inner=2048 --n_layers=1 --n_head=4 --d_k=64 --d_v=64 --dropout=0.1 
```

Running the HiTSKT model on the Junyi dataset:

```
python main.py --dataset='Junyi' --epoch_num=50 --batch_size=64 --session_size=16 --action_size=32 --embedding_size=128 --learning_rate=5e-5 --d_inner=1024 --n_layers=1 --n_head=2 --d_k=64 --d_v=64 --dropout=0.1 
```

Running the HiTSKT model on the EdNet dataset:

```
python main.py --dataset='ednet' --epoch_num=40 --batch_size=64 --session_size=16 --action_size=32 --embedding_size=128 --learning_rate=8e-5 --d_inner=1024 --n_layers=1 --n_head=2 --d_k=64 --d_v=64 --dropout=0.1 
```
