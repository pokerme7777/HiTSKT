# HiTSKT: A Hierarchical Transformer Model for Session-awared Knowledge Tracing
## Keywords
-  Hierarchical Transformer
-  Session-awared Knowledge Tracing
-  Knowledge Tracing
- 
🔥This is the code for the paper HiTSKT: A Hierarchical Transformer Model for Session-awared Knowledge Tracing, accepted by Knowledge-Based Systems \[[Papge]([https://hydra-vl4ai.github.io](https://www.sciencedirect.com/science/article/pii/S0950705123010481))\].🔥

## Setup

The requiring environments is as bellow:

- Python 3.6+
- PyTorch 1.9.0
- Scikit-learn 0.24.2
- Numpy 1.19.5
- Pandas 1.1.5
- Dask 2021.3.0

## Data and Data Preprocessing

We list the command to run the HiTSKT on different datasets. Listed hyperparameters are the optimal parameters for the respective datasets. The preprocessed data of ASSISTment 2017 and Junyi datasets are provided in the ``dataset`` directory. Due to the file size limitation of GitHub, we are not able to provide the preprocessed data of the EdNet dataset at this stage. Please download the ``train.csv`` from [this kaggle page](https://www.kaggle.com/c/riiid-test-answer-prediction/data), rename it to "ednet.csv".

Then, create a new directory ``Dataset`` and put ``ednet.csv`` into this directory.

```
mkdir Dataset
```

To preprocess the ``ednet.csv`` file, run the preprocessing script.

```
python preprocessing.py --dataset=ednet 
```

## Training and Testing HiTSKT


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
