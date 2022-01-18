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
import os
import os.path
import glob

def select_student(df,ses_min_no):
	student_use = []
	for st_id in df.studentId.unique():
		df_student = df[df.studentId == st_id]
		if max(df_student.session_no.to_list()) >=ses_min_no:
			student_use.append(st_id)

	df = df[df['studentId'].isin(student_use)]
	df = df.sort_values(by=['studentId', 'startTime'])
	df = df.reset_index(drop=True)

	return df

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

def akt_load_data(df1, n_question, seqlen):
	q_data = []
	qa_data = []
	p_data = []
	# target_data = []

	df = df1.copy()
	df['Xindex'] = df.skill + df.correct * n_question
	for i in df.studentId.unique():

		# dataframe for one student
		df_student = df[df.studentId == i]
		Q = df_student.skill.tolist()
		P = df_student.problemId.tolist()
		A = df_student.correct.tolist()
		Xindex = df_student.Xindex.tolist()

		# start split the data
		n_split = 1
		# print('len(Q):',len(Q))
		if len(Q) > seqlen:
			n_split = math.floor(len(Q) / seqlen)
			if len(Q) % seqlen:
				n_split = n_split + 1
		# print('n_split:',n_split)
		for k in range(n_split):

			if k == n_split - 1:
				endINdex = len(A)
			else:
				endINdex = (k+1) * seqlen

			q_data.append(Q[k * seqlen: endINdex])
			qa_data.append(Xindex[k * seqlen: endINdex])
			p_data.append(P[k * seqlen: endINdex])

	# convert data into ndarrays for better speed during training
	q_dataArray = np.zeros((len(q_data), seqlen))
	for j in range(len(q_data)):
		dat = q_data[j]
		q_dataArray[j, :len(dat)] = dat

	qa_dataArray = np.zeros((len(qa_data), seqlen))
	for j in range(len(qa_data)):
		dat = qa_data[j]
		qa_dataArray[j, :len(dat)] = dat

	p_dataArray = np.zeros((len(p_data), seqlen))
	for j in range(len(p_data)):
		dat = p_data[j]
		p_dataArray[j, :len(dat)] = dat

	return q_dataArray, qa_dataArray, p_dataArray

class AKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks,
                 kq_same, dropout, model_type, final_fc_dim=512, n_heads=8, d_ff=2048,  l2=1e-5, separate_qa=False):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            n_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
        """
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = model_type
        self.separate_qa = separate_qa
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question+1, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def forward(self, q_data, qa_data, target, pid_data=None):
        # Batch First
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            # BS, seqlen, d_model #f_(ct,rt)
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_data = (qa_data-q_data)//self.n_question  # rt
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(qa_data)+q_embed_data

        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct
            pid_embed_data = self.difficult_param(pid_data)  # uq
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct
            qa_embed_diff_data = self.qa_embed_diff(
                qa_data)  # f_(ct,rt) or #h_rt
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct)
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        d_output = self.model(q_embed_data, qa_embed_data)  # 211x512

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q)
        labels = target.reshape(-1)
        m = nn.Sigmoid()
        preds = (output.reshape(-1))  # logit
        mask = labels > -0.9
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        output = loss(masked_preds, masked_labels)
        return output.sum()+c_reg_loss, m(preds), mask.sum()


class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'akt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False)
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)
        Output:
            query: Input gets changed over the layer and returned.
        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
        (1.0 - target) * np.log(np.maximum(1e-10, 1.0-pred))
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target, all_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def akt_train(model,  optimizer,  q_data, qa_data, pid_data,  batch_size, n_question):
	model.train()

	N = int(math.ceil(len(q_data) / batch_size))
	q_data = q_data.T  # Shape: (200,3633)
	qa_data = qa_data.T  # Shape: (200,3633)
	# Shuffle the data
	shuffled_ind = np.arange(q_data.shape[1])
	np.random.shuffle(shuffled_ind)
	q_data = q_data[:, shuffled_ind]
	qa_data = qa_data[:, shuffled_ind]


	pid_data = pid_data.T
	pid_data = pid_data[:, shuffled_ind]

	pred_list = []
	target_list = []

	element_count = 0
	true_el = 0

	for idx in range(N):
		optimizer.zero_grad()

		q_one_seq = q_data[:, idx*batch_size:(idx+1)*batch_size]
		pid_one_seq = pid_data[:, idx * batch_size:(idx+1) * batch_size]

		qa_one_seq = qa_data[:, idx *batch_size:(idx+1) * batch_size]


		input_q = np.transpose(q_one_seq[:, :])  # Shape (bs, seqlen)
		input_qa = np.transpose(qa_one_seq[:, :])  # Shape (bs, seqlen)
		target = np.transpose(qa_one_seq[:, :])

		# Shape (seqlen, batch_size)
		input_pid = np.transpose(pid_one_seq[:, :])

		target = (target - 1) / n_question
		target_1 = np.floor(target)
		el = np.sum(target_1 >= -.9)
		element_count += el

		input_q = torch.from_numpy(input_q).long().to(device)
		input_qa = torch.from_numpy(input_qa).long().to(device)
		target = torch.from_numpy(target_1).float().to(device)

		input_pid = torch.from_numpy(input_pid).long().to(device)

		loss, pred, true_ct = model(input_q, input_qa, target, input_pid)

		pred = pred.detach().cpu().numpy()  # (seqlen * batch_size, 1)
		loss.backward()
		true_el += true_ct.cpu().numpy()


		optimizer.step()

		target = target_1.reshape((-1,))

		nopadding_index = np.flatnonzero(target >= -0.9)
		nopadding_index = nopadding_index.tolist()
		pred_nopadding = pred[nopadding_index]
		target_nopadding = target[nopadding_index]

		pred_list.append(pred_nopadding)
		target_list.append(target_nopadding)

	all_pred = np.concatenate(pred_list, axis=0)
	all_target = np.concatenate(target_list, axis=0)

	loss = binaryEntropy(all_target, all_pred)
	auc = compute_auc(all_target, all_pred)
	# accuracy = compute_accuracy(all_target, all_pred)

	return loss,  auc

def akt_test(model, optimizer, q_data, qa_data, pid_data, batch_size, n_question):

	model.eval()
	N = int(math.ceil(float(len(q_data)) / float(batch_size)))
	q_data = q_data.T  # Shape: (200,3633)
	qa_data = qa_data.T  # Shape: (200,3633)

	pid_data = pid_data.T
	seq_num = q_data.shape[1]
	pred_list = []
	target_list = []

	count = 0
	true_el = 0
	element_count = 0
	for idx in range(N):

		q_one_seq = q_data[:, idx*batch_size:(idx+1)*batch_size]

		pid_one_seq = pid_data[:, idx * batch_size:(idx+1) * batch_size]
		input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
		qa_one_seq = qa_data[:, idx * batch_size:(idx+1) *batch_size]
		input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)

		# Shape (seqlen, batch_size)
		input_q = np.transpose(q_one_seq[:, :])
		# Shape (seqlen, batch_size)
		input_qa = np.transpose(qa_one_seq[:, :])
		target = np.transpose(qa_one_seq[:, :])
		input_pid = np.transpose(pid_one_seq[:, :])
		
		target = (target - 1) / n_question
		target_1 = np.floor(target)
		#target = np.random.randint(0,2, size = (target.shape[0],target.shape[1]))

		input_q = torch.from_numpy(input_q).long().to(device)
		input_qa = torch.from_numpy(input_qa).long().to(device)
		target = torch.from_numpy(target_1).float().to(device)

		input_pid = torch.from_numpy(input_pid).long().to(device)

		with torch.no_grad():
			loss, pred, ct = model(input_q, input_qa, target, input_pid)

		pred = pred.cpu().numpy()  # (seqlen * batch_size, 1)
		true_el += ct.cpu().numpy()

		if (idx + 1) * batch_size > seq_num:
			real_batch_size = seq_num - idx * batch_size
			count += real_batch_size
		else:
			count += batch_size


		target = target_1.reshape((-1,))
		nopadding_index = np.flatnonzero(target >= -0.9)
		nopadding_index = nopadding_index.tolist()
		pred_nopadding = pred[nopadding_index]
		target_nopadding = target[nopadding_index]

		element_count += pred_nopadding.shape[0]
		# print avg_loss
		pred_list.append(pred_nopadding)
		target_list.append(target_nopadding)

	assert count == seq_num, "Seq not matching"

	all_pred = np.concatenate(pred_list, axis=0)
	all_target = np.concatenate(target_list, axis=0)
	loss = binaryEntropy(all_target, all_pred)
	auc = compute_auc(all_target, all_pred)
	# accuracy = compute_accuracy(all_target, all_pred)

	return loss, auc


def main():
    '''
    The main function of the training script
    '''
    df = pd.read_csv('./dataset/2017.csv', low_memory=False)

    df_train, df_val, df_test = split_dataset(df)
    print('Data Loading')

    n_question= max(df.skill) + 2  #question is skill
    n_pid = max(df.problemId) + 2  #problem is problemId
    n_layer =  1
    d_model = 256
    dropout = 0.05
    kq_same = 1
    model_type = 'akt'
    l2 = 1e-5
    lr = 5e-5
    batch_size = 24
    seqlen = 200

    epoch_num = 200
    seed_no = 123

    # model path
    model_path = './aktmodel1'

    # seed
    # np.random.seed(seed_no)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.manual_seed(seed_no)
    # np.random.seed(seed_no)
    # random.seed(seed_no)
    # torch.cuda.manual_seed(seed_no)
    # torch.cuda.manual_seed_all(seed_no)

    train_q_data, train_qa_data, train_pid = akt_load_data(df1=df_train, n_question=n_question, seqlen=seqlen)
    val_q_data, val_qa_data, val_pid = akt_load_data(df1=df_val, n_question=n_question, seqlen=seqlen)
    test_q_data, test_qa_data, test_pid = akt_load_data(df1=df_test, n_question=n_question, seqlen=seqlen)

    print('Arrary finished')
    model_akt = AKT(n_question=n_question, n_pid=n_pid, n_blocks=n_layer, d_model=d_model,
                dropout=dropout, kq_same=kq_same, model_type=model_type, l2=l2,
                final_fc_dim=512, n_heads=8, d_ff=1024).to(device)

    optimizer = torch.optim.Adam(model_akt.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    print('Begin to train model')
    # train hy_model 100 times
    train_loss_list = []
    train_AUC_list = []
    val_AUC_list = []
    val_loss_list=[]
    best_valid_auc = 0
    for epoch in range(epoch_num):
        loss, train_AUC = akt_train(model=model_akt,  optimizer=optimizer,  q_data=train_q_data, qa_data=train_qa_data, pid_data=train_pid,  batch_size=batch_size, n_question=n_question)

        print('train_one_epoch: ', loss, 'train_AUC:', train_AUC)
        train_loss_list.append(loss)
        train_AUC_list.append(train_AUC)

        # scheduler.step()
        print('------------------------------')
        print('------------------------------')

        val_loss, val_AUC = akt_test(model=model_akt,  optimizer=optimizer,  q_data=val_q_data, qa_data=val_qa_data, pid_data=val_pid,  batch_size=batch_size, n_question=n_question)


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
                        'model_state_dict': model_akt.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        },
                        os.path.join(model_path,  'val')+'_' + str(best_epoch)
                        )

        if (val_loss - min(val_loss_list)) > 0.1:
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
    model_akt.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_AUC =  akt_test(model=model_akt,  optimizer=optimizer,  q_data=test_q_data, qa_data=test_qa_data, pid_data=test_pid,  batch_size=batch_size, n_question=n_question)


    print('Test_AUC', test_AUC)
    print('Best_epoch', best_epoch)

    path = os.path.join(model_path,  'val') + '_*'
    for i in glob.glob(path):
        os.remove(i)


if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	main()