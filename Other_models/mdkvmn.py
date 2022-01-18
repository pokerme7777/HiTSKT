# import library
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
import random
import math
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



def dkvmn_load_data(df1, n_question, seqlen):
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

    return q_dataArray, qa_dataArray


class DKVMN_sub(nn.Module):
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, init_memory_key):
        super(DKVMN_sub, self).__init__()
        """
        :param memory_size:             scalar
        :param memory_key_state_dim:    scalar
        :param memory_value_state_dim:  scalar
        :param init_memory_key:         Shape (memory_size, memory_value_state_dim)
        :param init_memory_value:       Shape (batch_size, memory_size, memory_value_state_dim)
        """
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim

        self.key_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                       memory_state_dim=self.memory_key_state_dim,
                                       is_write=False)

        self.value_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                         memory_state_dim=self.memory_value_state_dim,
                                         is_write=True)

        self.memory_key = init_memory_key

        # self.memory_value = self.init_memory_value
        self.memory_value = None

    def init_value_memory(self, memory_value):
        self.memory_value = memory_value

    def attention(self, control_input):
        correlation_weight = self.key_head.addressing(control_input=control_input, memory=self.memory_key)
        return correlation_weight

    def read(self, read_weight):
        read_content = self.value_head.read(memory=self.memory_value, read_weight=read_weight)

        return read_content

    def write(self, write_weight, control_input, if_write_memory):
        memory_value = self.value_head.write(control_input=control_input,
                                             memory=self.memory_value,
                                             write_weight=write_weight)

        self.memory_value = nn.Parameter(memory_value.data)

        return self.memory_value

class DKVMNHeadGroup(nn.Module):
    def __init__(self, memory_size, memory_state_dim, is_write):
        super(DKVMNHeadGroup, self).__init__()
        """"
        Parameters
            memory_size:        scalar
            memory_state_dim:   scalar
            is_write:           boolean
        """
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write
        if self.is_write:
            self.erase = torch.nn.Linear(self.memory_state_dim, self.memory_state_dim, bias=True)
            self.add = torch.nn.Linear(self.memory_state_dim, self.memory_state_dim, bias=True)
            nn.init.kaiming_normal_(self.erase.weight)
            nn.init.kaiming_normal_(self.add.weight)
            nn.init.constant_(self.erase.bias, 0)
            nn.init.constant_(self.add.bias, 0)


    def addressing(self, control_input, memory):
        """
        Parameters
            control_input:          Shape (batch_size, control_state_dim)
            memory:                 Shape (memory_size, memory_state_dim)
        Returns
            correlation_weight:     Shape (batch_size, memory_size)
        """
        similarity_score = torch.matmul(control_input, torch.t(memory))
        correlation_weight = torch.nn.functional.softmax(similarity_score, dim=1) # Shape: (batch_size, memory_size)
        return correlation_weight

    def read(self, memory, control_input=None, read_weight=None):
        """
        Parameters
            control_input:  Shape (batch_size, control_state_dim)
            memory:         Shape (batch_size, memory_size, memory_state_dim)
            read_weight:    Shape (batch_size, memory_size)
        Returns
            read_content:   Shape (batch_size,  memory_state_dim)
        """
        if read_weight is None:
            read_weight = self.addressing(control_input=control_input, memory=memory)
        read_weight = read_weight.view(-1, 1)
        memory = memory.view(-1, self.memory_state_dim)
        rc = torch.mul(read_weight, memory)
        read_content = rc.view(-1, self.memory_size, self.memory_state_dim)
        read_content = torch.sum(read_content, dim=1)
        return read_content

    def write(self, control_input, memory, write_weight=None):
        """
        Parameters
            control_input:      Shape (batch_size, control_state_dim)
            write_weight:       Shape (batch_size, memory_size)
            memory:             Shape (batch_size, memory_size, memory_state_dim)
        Returns
            new_memory:         Shape (batch_size, memory_size, memory_state_dim)
        """
        assert self.is_write
        if write_weight is None:
            write_weight = self.addressing(control_input=control_input, memory=memory)
        erase_signal = torch.sigmoid(self.erase(control_input))
        add_signal = torch.tanh(self.add(control_input))
        erase_reshape = erase_signal.view(-1, 1, self.memory_state_dim)
        add_reshape = add_signal.view(-1, 1, self.memory_state_dim)
        write_weight_reshape = write_weight.view(-1, self.memory_size, 1)
        erase_mult = torch.mul(erase_reshape, write_weight_reshape)
        add_mul = torch.mul(add_reshape, write_weight_reshape)
        new_memory = memory * (1 - erase_mult) + add_mul
        return new_memory




class DKVMN(nn.Module):

    def __init__(self, n_question, batch_size, q_embed_dim, qa_embed_dim,
                 memory_size, memory_key_state_dim, memory_value_state_dim, final_fc_dim):
        super(DKVMN, self).__init__()
        self.n_question = n_question
        self.batch_size = batch_size
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        self.final_fc_dim = final_fc_dim

        self.input_embed_linear = nn.Linear(self.q_embed_dim, self.final_fc_dim, bias=True)
        self.read_embed_linear = nn.Linear(self.memory_value_state_dim + self.final_fc_dim, self.final_fc_dim, bias=True)
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)
        self.init_memory_key = nn.Parameter(torch.randn(self.memory_size, self.memory_key_state_dim))
        nn.init.kaiming_normal_(self.init_memory_key)
        self.init_memory_value = nn.Parameter(torch.randn(self.memory_size, self.memory_value_state_dim))
        nn.init.kaiming_normal_(self.init_memory_value)

        self.mem = DKVMN_sub(memory_size=self.memory_size,
                   memory_key_state_dim=self.memory_key_state_dim,
                   memory_value_state_dim=self.memory_value_state_dim, init_memory_key=self.init_memory_key)

        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        self.q_embed = nn.Embedding(self.n_question + 1, self.q_embed_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * self.n_question + 1, self.qa_embed_dim, padding_idx=0)

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.predict_linear.bias, 0)


    def init_embeddings(self):

        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)

    def forward(self, q_data, qa_data, target):

        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)

        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        slice_q_data = torch.chunk(q_data, seqlen, 1)
        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)

        value_read_content_l = []
        input_embed_l = []
        predict_logs = []
        for i in range(seqlen):
            ## Attention
            q = slice_q_embed_data[i].squeeze(1)
            correlation_weight = self.mem.attention(q)
            if_memory_write = slice_q_data[i].squeeze(1).ge(1)
            if_memory_write = torch.FloatTensor(if_memory_write.data.tolist()).to(device)


            ## Read Process
            read_content = self.mem.read(correlation_weight)
            value_read_content_l.append(read_content)
            input_embed_l.append(q)
            ## Write Process
            qa = slice_qa_embed_data[i].squeeze(1)
            new_memory_value = self.mem.write(correlation_weight, qa, if_memory_write)


        all_read_value_content = torch.cat([value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        input_embed_content = torch.cat([input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1)


        predict_input = torch.cat([all_read_value_content, input_embed_content], 2)
        read_content_embed = torch.tanh(self.read_embed_linear(predict_input.view(batch_size*seqlen, -1)))

        pred = self.predict_linear(read_content_embed)
        target_1d = target                   # [batch_size * seq_len, 1]
        mask = target_1d.ge(0)               # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)           # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        return loss, torch.sigmoid(filtered_pred), filtered_target

def to_scalar(var):
    return var.view(-1).data.tolist()[0]

def dkvmn_train(model, optimizer, q_data, qa_data, batch_size, n_question, maxgradnorm):
    N = int(math.floor(len(q_data) / batch_size))

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.train()

    for idx in range(N):
        q_one_seq = q_data[idx * batch_size:(idx + 1) * batch_size, :]
        qa_batch_seq = qa_data[idx * batch_size:(idx + 1) * batch_size, :]
        target = qa_data[idx * batch_size:(idx + 1) * batch_size, :]

        target = (target - 1) / n_question
        target = np.floor(target)
        input_q = torch.LongTensor(q_one_seq).to(device)
        input_qa = torch.LongTensor(qa_batch_seq).to(device)
        target = torch.FloatTensor(target).to(device)
        target_to_1d = torch.chunk(target, batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(batch_size)], 1)
        target_1d = target_1d.permute(1, 0)

        model.zero_grad()
        loss, filtered_pred, filtered_target = model.forward(input_q, input_qa, target_1d)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), maxgradnorm)
        optimizer.step()
        epoch_loss += to_scalar(loss)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())

        pred_list.append(right_pred)
        target_list.append(right_target)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)


    return epoch_loss/N, auc



def dkvmn_test(model, optimizer, q_data, qa_data, batch_size, n_question):
    N = int(math.floor(len(q_data) / batch_size))

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.eval()


    for idx in range(N):

        q_one_seq = q_data[idx * batch_size:(idx + 1) * batch_size, :]
        qa_batch_seq = qa_data[idx * batch_size:(idx + 1) * batch_size, :]
        target = qa_data[idx * batch_size:(idx + 1) * batch_size, :]

        target = (target - 1) / n_question
        target = np.floor(target)

        input_q = torch.LongTensor(q_one_seq).to(device)
        input_qa = torch.LongTensor(qa_batch_seq).to(device)
        target = torch.FloatTensor(target).to(device)

        target_to_1d = torch.chunk(target, batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(batch_size)], 1)
        target_1d = target_1d.permute(1, 0)

        loss, filtered_pred, filtered_target = model.forward(input_q, input_qa, target_1d)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)
        epoch_loss += to_scalar(loss)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)


    return epoch_loss/N, auc


def main():
    '''
    The main function of the training script
    '''
    df = pd.read_csv('./dataset/2017.csv', low_memory=False)
    df_train, df_val, df_test = split_dataset(df)
    print('Data Loading')


    n_question = max(df.problemId)+1
    batch_size = 32
    q_embed_dim= 50
    qa_embed_dim= 100
    memory_size = 20
    memory_key_state_dim =q_embed_dim
    memory_value_state_dim=qa_embed_dim
    final_fc_dim=50
    maxgradnorm=50
    lr=1e-3
    epoch_num = 100
    seqlen=200

    print('Loading array')
    train_q_data, train_qa_data = dkvmn_load_data(df1=df_train, n_question=n_question, seqlen=seqlen)
    val_q_data, val_qa_data = dkvmn_load_data(df1=df_val, n_question=n_question, seqlen=seqlen)
    test_q_data, test_qa_data = dkvmn_load_data(df1=df_test, n_question=n_question, seqlen=seqlen)

    print('array done')

    dkvmn_model = DKVMN(n_question=n_question,
                batch_size=batch_size,
                q_embed_dim=q_embed_dim,
                qa_embed_dim=qa_embed_dim,
                memory_size=memory_size,
                memory_key_state_dim=memory_key_state_dim,
                memory_value_state_dim=memory_value_state_dim,
                final_fc_dim=final_fc_dim).to(device)

    optimizer =  torch.optim.Adam(dkvmn_model.parameters(), lr=lr)

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
    model_path = 'dkvmn_model'

    for epoch in range(epoch_num):
        loss, train_AUC = dkvmn_train(model=dkvmn_model, optimizer=optimizer, q_data=train_q_data, qa_data=train_qa_data,
                              batch_size=batch_size, n_question=n_question, maxgradnorm=maxgradnorm)

        print('train_one_epoch: ', loss, 'train_AUC:', train_AUC)
        train_loss_list.append(loss)
        train_AUC_list.append(train_AUC)

        # scheduler.step()
        print('------------------------------')
        print('------------------------------')

        val_loss, val_AUC = dkvmn_test(model=dkvmn_model, optimizer=optimizer, q_data=val_q_data, qa_data=val_qa_data,
                              batch_size=batch_size, n_question=n_question)


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
                        'model_state_dict': dkvmn_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        },
                        os.path.join(model_path,  'val')+'_' + str(best_epoch)
                        )

        if (val_loss - min(val_loss_list)) > 1000:
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
    dkvmn_model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_AUC =  dkvmn_test(model=dkvmn_model, optimizer=optimizer, q_data=test_q_data, qa_data=test_qa_data,
                              batch_size=batch_size, n_question=n_question)


    print('Test_AUC', test_AUC)
    print('Best_epoch', best_epoch)

    path = os.path.join(model_path,  'val') + '_*'
    for i in glob.glob(path):
        os.remove(i)

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	main()