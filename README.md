# HiTSKT: A Hierarchical Transformer Model for Session-awared Knowledge Tracing

## Setup
The requiring environments is as bellow:
- Python 3.6+
- PyTorch 1.9.0
- Scikit-learn 0.24.2
- Numpy 1.19.5
- Pandas 1.1.5

## Run HiTSKT
Here are the example for running HiTSKT on ASSISTments2017 dataset:
```
python main.py
```

| Args          |  Default      |              Help              | 
| ------------- | ------------- |         -------------
| epoch_num     |    100        |    number of iterations        |
| batch_size    |    64         |      number of batch           |
| session_size  |    16         |      number of sessions        |
| action_size   |    64         |  number of interactions in each session  |
|embedding_size |    256        |      embedding dimensions      |
| learning_rate |    5e-5       |      learning rate             |
| d_inner       |    2048       |      FFN hidden dimension      |
| n_layers      |    1          |      number of layers          |
| n_head        |    4          |   number of head for multihead attention           |
| d_k           |    64         |      k query dimensions        |
| d_v           |    64         |      v query dimensions        |
| dropout       |    0.1        |      dropout                   |
| dataset       |    2017       |      dataset name              |

