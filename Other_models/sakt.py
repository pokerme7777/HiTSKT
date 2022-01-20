# Code reused from https://github.com/theophilee/learner-performance-prediction
# import library
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import random
import math
import copy
from random import shuffle
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import os
import os.path
import glob



def split_dataset(df):

    train_idx = []
    val_idx = []
    test_idx = []
    for st_id in df.studentId.unique():
        df_student = df[df.studentId == st_id]
        stu_session_len = max(df_student.session_no.unique())
        test_len = stu_session_len//5
        test_range = list(range(stu_session_len-test_len+1,stu_session_len+1))
        val_range = list(range(stu_session_len-2*test_len+1,stu_session_len-test_len+1))
        train_range = list(range(1,stu_session_len-2*test_len+1))

        stu_train = list(df_student[df_student['session_no'].isin(train_range)].index)
        stu_val = list(df_student[df_student['session_no'].isin(val_range)].index)
        stu_test = list(df_student[df_student['session_no'].isin(test_range)].index)

        train_idx.extend(stu_train)
        val_idx.extend(stu_val)
        test_idx.extend(stu_test)
    
        

    def sort_drop (df_):
        df_ = df_.sort_values(by=['studentId', 'startTime'])
        df_student = df_.reset_index(drop=True)
        return df_

    df_train = sort_drop(df.iloc[train_idx])
    df_val = sort_drop(df.iloc[val_idx])
    df_test = sort_drop(df.iloc[test_idx])

    return df_train, df_val, df_test

def get_data(df, max_length):
    """Extract sequences from dataframe.
    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        max_length (int): maximum length of a sequence chunk
    """
    item_ids = []
    skill_ids = []
    labels = []
    data = []
    for i in df.studentId.unique():
        # dataframe for one student
        df_student = df[df.studentId == i]

        skill_ids = df_student.skill.tolist()   
        item_ids = df_student.problemId.tolist()
        labels = df_student.correct.tolist()
        start_indx=0
        if len(skill_ids) >= max_length:
            n_splits = len(skill_ids) // max_length +1
            
            for split in range(n_splits):
                if split != n_splits-1:
                    skill_ids1 = skill_ids[start_indx: start_indx+max_length]
                    item_ids1 = item_ids[start_indx: start_indx+max_length]
                    labels1 = labels[start_indx: start_indx+max_length]
                    start_indx +=max_length

                    item_inputs = [i+1 for i in item_ids1][:-1]
                    item_inputs.insert(0, 0)
                    skill_inputs = [i+1 for i in skill_ids1][:-1]
                    skill_inputs.insert(0, 0)
                    label_inputs = [i+1 for i in labels1][:-1]
                    label_inputs.insert(0, 2)

                    data.append([item_inputs,skill_inputs,label_inputs,item_ids1,skill_ids1,labels1])

                else:
                    skill_ids = skill_ids[start_indx:]
                    item_ids = item_ids[start_indx:]
                    labels = labels[start_indx:]

        
        skill_ids.extend([0] * (max_length - len(skill_ids) ))
        item_ids.extend([0] * (max_length - len(item_ids) ))
        labels.extend([2] * (max_length - len(labels) ))

        item_inputs = [i+1 for i in item_ids][:-1]
        item_inputs.insert(0, 0)
        skill_inputs = [i+1 for i in skill_ids][:-1]
        skill_inputs.insert(0, 0)
        label_inputs = [i+1 for i in labels][:-1]
        label_inputs.insert(0, 2)

        # (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels)

        data.append([item_inputs,skill_inputs,label_inputs,item_ids,skill_ids,labels])
    return data


def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


def attention(query, key, value, mask=None, dropout=None):
    """Compute scaled dot product attention.
    """
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    prob_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn


def relative_attention(query, key, value, pos_key_embeds, pos_value_embeds, mask=None, dropout=None):
    """Compute scaled dot product attention with relative position embeddings.
    (https://arxiv.org/pdf/1803.02155.pdf)
    """
    assert pos_key_embeds.num_embeddings == pos_value_embeds.num_embeddings

    scores = torch.matmul(query, key.transpose(-2, -1))

    idxs = torch.arange(scores.size(-1))
    if query.is_cuda:
        idxs = idxs.cuda()
    idxs = idxs.view(-1, 1) - idxs.view(1, -1)
    idxs = torch.clamp(idxs, 0, pos_key_embeds.num_embeddings - 1)

    pos_key = pos_key_embeds(idxs).transpose(-2, -1)
    pos_scores = torch.matmul(query.unsqueeze(-2), pos_key)
    scores = scores.unsqueeze(-2) + pos_scores
    scores = scores / math.sqrt(query.size(-1))

    pos_value = pos_value_embeds(idxs)
    value = value.unsqueeze(-3) + pos_value

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(-2), -1e9)
    prob_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)

    output = torch.matmul(prob_attn, value).unsqueeze(-2)
    prob_attn = prob_attn.unsqueeze(-2)
    return output, prob_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.linear_layers = clone(nn.Linear(total_size, total_size), 3)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, query, key, value, encode_pos, pos_key_embeds, pos_value_embeds, mask=None):
        batch_size, seq_length = query.shape[:2]

        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Project inputs
        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                                for l, x in zip(self.linear_layers, (query, key, value))]

        # Apply attention
        if encode_pos:
            out, self.prob_attn = relative_attention(
                query, key, value, pos_key_embeds, pos_value_embeds, mask, self.dropout)
        else:
            out, self.prob_attn = attention(query, key, value, mask, self.dropout)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        return out


class SAKT(nn.Module):
    def __init__(self, num_items, num_skills, embed_size, num_attn_layers, num_heads,
                    encode_pos, max_pos, drop_prob):
        """Self-attentive knowledge tracing.
        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            encode_pos (bool): if True, use relative position embeddings
            max_pos (int): number of position embeddings to use
            drop_prob (float): dropout probability
        """
        super(SAKT, self).__init__()
        self.embed_size = embed_size
        self.encode_pos = encode_pos

        self.item_embeds = nn.Embedding(num_items + 1, embed_size // 2, padding_idx=0)
        self.skill_embeds = nn.Embedding(num_skills + 1, embed_size // 2, padding_idx=0)

        self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)

        self.lin_in = nn.Linear(2 * embed_size, embed_size)
        self.attn_layers = clone(MultiHeadedAttention(embed_size, num_heads, drop_prob), num_attn_layers)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lin_out = nn.Linear(embed_size, 1)
        
    def get_inputs(self, item_inputs, skill_inputs, label_inputs):
        item_inputs = self.item_embeds(item_inputs)
        skill_inputs = self.skill_embeds(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()

        inputs = torch.cat([item_inputs, skill_inputs, item_inputs, skill_inputs], dim=-1)
        inputs[..., :self.embed_size] *= label_inputs
        inputs[..., self.embed_size:] *= 1 - label_inputs
        return inputs

    def get_query(self, item_ids, skill_ids):
        item_ids = self.item_embeds(item_ids)
        skill_ids = self.skill_embeds(skill_ids)
        query = torch.cat([item_ids, skill_ids], dim=-1)
        return query

    def forward(self, item_inputs, skill_inputs, label_inputs, item_ids, skill_ids):
        inputs = self.get_inputs(item_inputs, skill_inputs, label_inputs)
        inputs = F.relu(self.lin_in(inputs))

        query = self.get_query(item_ids, skill_ids)

        mask = future_mask(inputs.size(-2))
        if inputs.is_cuda:
            mask = mask.cuda()

        outputs = self.dropout(self.attn_layers[0](query, inputs, inputs, self.encode_pos,
                                                    self.pos_key_embeds, self.pos_value_embeds, mask))
        for l in self.attn_layers[1:]:
            residual = l(query, outputs, outputs, self.encode_pos, self.pos_key_embeds,
                            self.pos_value_embeds, mask)
            outputs = self.dropout(outputs + F.relu(residual))

        return self.lin_out(outputs)

def sakt_train(train_data, model, optimizer, batch_size, grad_clip):
    """Train SAKT model.
    Arguments:
        train_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        batch_size (int)
        grad_clip (float): max norm of the gradients
    """
    criterion = nn.BCEWithLogitsLoss()
    step = 0
    Plabel_list = torch.Tensor([])
    Ground_true_list = torch.Tensor([])
    loss_total = 0

    CEL = nn.BCELoss(reduction='sum')
    train_batches = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    model.train()

    # (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels)
    # Training
    for input in train_batches:
        input = input.cuda()

        preds = model(input[:,0], input[:,1], input[:,2], input[:,3], input[:,4])
        preds = preds.squeeze(2)

        label_mask = (input[:,5] != 2)
        z = torch.sigmoid(preds[label_mask])
        label_seq = input[:,5][label_mask]

        if z.size() == torch.Size([]):
            z = torch.stack([z])

        one_batch_loss = CEL(z, label_seq.float())

        z = z.to('cpu')
        label_seq = label_seq.to('cpu')

        Plabel_list = torch.cat((Plabel_list, z))
        Ground_true_list = torch.cat((Ground_true_list, label_seq))


        one_batch_loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        loss_total += one_batch_loss.to('cpu')
        del input

    train_auc =  metrics.roc_auc_score(Ground_true_list.detach().numpy().astype(int),
                                            Plabel_list.detach().numpy())

    return float(loss_total), train_auc

def sakt_test(test_data, model, optimizer, batch_size, grad_clip):
    """Train SAKT model.
    Arguments:
        test_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        batch_size (int)
        grad_clip (float): max norm of the gradients
    """
    criterion = nn.BCEWithLogitsLoss()
    step = 0
    Plabel_list = torch.Tensor([])
    Ground_true_list = torch.Tensor([])
    loss_total = 0

    CEL = nn.BCELoss(reduction='sum')
    test_batches = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model.eval()

    # Test
    for input in test_batches:
        input = input.cuda()
        with torch.no_grad():
            preds = model(input[:,0], input[:,1], input[:,2], input[:,3], input[:,4])
            preds = preds.squeeze(2)

        label_mask = (input[:,5] != 2) 
        z = torch.sigmoid(preds[label_mask])
        label_seq = input[:,5][label_mask]

        if z.size() == torch.Size([]):
            z = torch.stack([z])

        one_batch_loss = CEL(z, label_seq.float())

        z = z.to('cpu')
        label_seq = label_seq.to('cpu')

        Plabel_list = torch.cat((Plabel_list, z))
        Ground_true_list = torch.cat((Ground_true_list, label_seq))   

        loss_total += one_batch_loss
        del input
    test_auc =  metrics.roc_auc_score(Ground_true_list.detach().numpy().astype(int),
                                            Plabel_list.detach().numpy())

    return float(loss_total), test_auc


def main():
    '''
    The main function of the training script
    '''
    df = pd.read_csv('./dataset/2017.csv', low_memory=False)

    df_train, df_val, df_test = split_dataset(df)
    print('Data Loading')

    num_student = max(df.problemId) +1 
    num_skills = max(df.skill) +1
    d_model = 200
    n_layers= 1
    n_head=5
    max_pos=10
    drop_prob=0.2
    grad_clip=10

    batch_size =10
    lr = 1e-5
    max_length = 100

    epoch_num = 300

    print('Loading array')
    train_data = get_data(df_train, max_length=max_length)
    val_data = get_data(df_val, max_length=max_length)
    test_data = get_data(df_test, max_length=max_length)

    train_data = torch.LongTensor(train_data)
    val_data = torch.LongTensor(val_data)
    test_data = torch.LongTensor(test_data)

    print('array done')

    sakt_model = SAKT(num_items=num_student, num_skills=num_skills, embed_size=d_model, num_attn_layers=n_layers, num_heads=n_head,
                    encode_pos=True, max_pos=max_pos, drop_prob=drop_prob).to(device)

    optimizer =  torch.optim.Adam(sakt_model.parameters(), lr=lr)

    print('begin to train')

    # train hy_model 100 times
    train_loss_list = []
    train_AUC_list = []
    val_AUC_list = []
    val_loss_list=[]
    best_valid_auc = 0

    # # seed
    # seed_no = 123
    # np.random.seed(seed_no)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.manual_seed(seed_no)
    # np.random.seed(seed_no)
    # random.seed(seed_no)
    # torch.cuda.manual_seed(seed_no)
    # torch.cuda.manual_seed_all(seed_no)

    # model path
    model_path = 'sakt_model'

    for epoch in range(epoch_num):
        loss, train_AUC = sakt_train(train_data=train_data, model=sakt_model, optimizer=optimizer, batch_size=batch_size, grad_clip=grad_clip)

        print('train_one_epoch: ', loss, 'train_AUC:', train_AUC)
        train_loss_list.append(loss)
        train_AUC_list.append(train_AUC)

        # scheduler.step()
        print('------------------------------')
        print('------------------------------')

        val_loss, val_AUC = sakt_test(test_data=val_data, model=sakt_model, optimizer=optimizer, batch_size=batch_size, grad_clip=grad_clip)


        print('------------------------------')
        print('Epoch: ',epoch ,' val AUC:', val_AUC)
        val_AUC_list.append(val_AUC)
        val_loss_list.append(val_loss)
        print('------------------------------------------')

        if val_AUC > best_valid_auc:
            path = os.path.join(model_path, 'val') + '_*'

            for i in glob.glob(path):
                os.remove(i)

            best_valid_auc = val_AUC
            best_epoch = epoch +1

            torch.save({'epoch': epoch,
                        'model_state_dict': sakt_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        },
                        os.path.join(model_path,  'val')+'_' + str(best_epoch)
                        )

        if (val_loss - min(val_loss_list)) > 8000:
            break

    print('------------------------------')
    print('train_loss_list', train_loss_list)
    print('train_AUC_list', train_AUC_list)
    print('VAL AUC List:', val_AUC_list)
    print('val loss List:', val_loss_list)
    print('max_val_auc:',max(val_AUC_list))
    print('------------------------------')
    print('Begin to test.........')

    checkpoint = torch.load(os.path.join(model_path,  'val') + '_'+str(best_epoch))
    sakt_model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_AUC =  sakt_test(test_data=test_data, model=sakt_model, optimizer=optimizer, batch_size=batch_size, grad_clip=grad_clip)


    print('Test_AUC', test_AUC)
    print('Best_epoch', best_epoch)

    # path = os.path.join(model_path,  'val') + '_*'
    # for i in glob.glob(path):
    #     os.remove(i)

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	main()
