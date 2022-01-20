# This is referenced from https://github.com/xiaopengguo/ATKT
# import library
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
import random
import math
from random import shuffle
from torch.utils.data import DataLoader
import os
import os.path
import glob
from torch.autograd import Variable, grad


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
    problem_array = []
    skill_array = []
    time_array = []
    correct_array = []
    data = []
    
    for i in df.studentId.unique():
        # dataframe for one student
        df_student = df[df.studentId == i]

        skill_array = df_student.skill.tolist()   
        problem_array = df_student.problemId.tolist()
        correct_array = df_student.correct.tolist()
        time_array = df_student.startTime.tolist()
        start_indx=0
        if len(problem_array) > max_length:
            n_splits = len(problem_array) // max_length +1
            for split in range(n_splits):
                if split != n_splits-1:
                    k_out = skill_array[start_indx: start_indx+max_length]
                    q_out = problem_array[start_indx: start_indx+max_length]
                    a_out = correct_array[start_indx: start_indx+max_length]
                    t_out = time_array[start_indx: start_indx+max_length]
                    start_indx +=max_length
                else:
                    pad = [0]*(max_length-len(skill_array[start_indx:]))
                    k_out = skill_array[start_indx:]
                    k_out.extend(pad)
                    q_out = problem_array[start_indx:]
                    q_out.extend(pad)
                    a_out = correct_array[start_indx:]
                    a_out.extend([2]*(max_length-len(correct_array[start_indx:])))
                    t_out = time_array[start_indx:]
                    t_out.extend(pad)
                
                data.append([k_out,q_out,a_out,t_out])

        
        else:
            pad = [0]*(max_length-len(skill_array[start_indx:]))
            k_out = skill_array[start_indx:]
            k_out.extend(pad)
            q_out = problem_array[start_indx:]
            q_out.extend(pad)
            a_out = correct_array[start_indx:]
            a_out.extend([2]*(max_length-len(correct_array[start_indx:])))
            t_out = time_array[start_indx:]
            t_out.extend(pad)
            data.append([k_out,q_out,a_out,t_out])

    return data


class ATKT(nn.Module):
    def __init__(self, skill_dim, answer_dim, hidden_dim, output_dim):
        super(ATKT, self).__init__()
        self.skill_dim=skill_dim
        self.answer_dim=answer_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(self.skill_dim+self.answer_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.sig = nn.Sigmoid()
        
        self.skill_emb = nn.Embedding(self.output_dim+1, self.skill_dim)
        self.skill_emb.weight.data[-1]= 0
        
        self.answer_emb = nn.Embedding(2+1, self.answer_dim)
        self.answer_emb.weight.data[-1]= 0
        
        self.attention_dim = 80
        self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)
        
    def _get_next_pred(self, res, skill):
        
        one_hot = torch.eye(self.output_dim, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)
        
        pred = (res * one_hot_skill).sum(dim=-1)
        return pred
    
    def attention_module(self, lstm_output):
        
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)
        
        alphas=nn.Softmax(dim=1)(att_w)
        
        attn_ouput=alphas*lstm_output
        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1=attn_output_cum-attn_ouput

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        
        return final_output


    def forward(self, skill, answer, perturbation=None):
        
        skill_embedding=self.skill_emb(skill)
        answer_embedding=self.answer_emb(answer)
        
        skill_answer=torch.cat((skill_embedding,answer_embedding), 2)
        answer_skill=torch.cat((answer_embedding,skill_embedding), 2)
        
        answer=answer.unsqueeze(2).expand_as(skill_answer)
        
        skill_answer_embedding=torch.where(answer==1, skill_answer, answer_skill)

        skill_answer_embedding1=skill_answer_embedding
        
        if  perturbation is not None:
            skill_answer_embedding+=perturbation
        
        out,_ = self.rnn(skill_answer_embedding)
        out=self.attention_module(out)
        res = self.sig(self.fc(out))

        res = res[:, :-1, :]
        pred_res = self._get_next_pred(res, skill)
        
        return pred_res, skill_answer_embedding1


class KTLoss(nn.Module):

    def __init__(self):
        super(KTLoss, self).__init__()

    def forward(self, pred_answers, real_answers):

        real_answers = real_answers[:, 1:]
        answer_mask = torch.ne(real_answers, 2)
        
        y_pred = pred_answers[answer_mask].float()
        y_true = real_answers[answer_mask].float()
        
        loss=nn.BCELoss()(y_pred, y_true)
        return loss, y_pred, y_true



def _l2_normalize_adv(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
        d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)

def ATKT_train(train_data, model, optimizer, batch_size, epsilon, beta):
    """Train DKT model.
    Arguments:
        train_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        batch_size (int)
    """
    Plabel_list = torch.Tensor([])
    Ground_true_list = torch.Tensor([])

    kt_loss = KTLoss()

    loss_total = 0

    train_batches = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    model.train()

    # Training
    for input in train_batches:
        input = input.to(device)

        skill = input[:,0]
        answer = input[:,2]

        pred_res, features = model(skill, answer)

        loss, y_pred, y_true = kt_loss(pred_res, answer)

        features_grad = grad(loss, features, retain_graph=True)
        p_adv = torch.FloatTensor(epsilon * _l2_normalize_adv(features_grad[0].data))
        p_adv = Variable(p_adv).to(device)
        pred_res, features = model(skill, answer, p_adv)
        adv_loss, _ , _ = kt_loss(pred_res, answer)

        total_loss=loss+ beta*adv_loss
        total_loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        loss_total += total_loss.to('cpu')
        del input

        y_pred = y_pred.to('cpu')
        y_true = y_true.to('cpu')


        Plabel_list = torch.cat((Plabel_list, y_pred))
        Ground_true_list = torch.cat((Ground_true_list, y_true))    

    train_auc =  metrics.roc_auc_score(Ground_true_list.detach().numpy().astype(int),
                                            Plabel_list.detach().numpy())

    return float(loss_total), train_auc


def ATKT_test(test_data, model, optimizer, batch_size):
    """Train DKT model.
    Arguments:
        test_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        batch_size (int)
    """
    Plabel_list = torch.Tensor([])
    Ground_true_list = torch.Tensor([])
    loss_total = 0

    kt_loss = KTLoss()


    test_batches = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model.eval()

    # Test
    for input in test_batches:
        input = input.to(device)

        skill = input[:,0]
        answer = input[:,2]

        with torch.no_grad():
            pred_res, features = model(skill, answer)
            loss, y_pred, y_true = kt_loss(pred_res, answer)

        y_pred = y_pred.to('cpu')
        y_true = y_true.to('cpu')


        Plabel_list = torch.cat((Plabel_list, y_pred))
        Ground_true_list = torch.cat((Ground_true_list, y_true))        

        loss_total += loss.to('cpu')
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

    n_skills= max(df.skill) + 2  #question is skill
    n_problems = max(df.problemId) + 2  #problem is problemId

    max_length=200

    seed_no=123
    epoch_num = 40
    lr = 1e-3
    batch_size = 24
    epsilon=12
    beta=1
    skill_emb_dim=256
    answer_emb_dim=96
    hidden_emb_dim=80

    # model path
    model_path = './atkt_model'

    # seed
    # np.random.seed(seed_no)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.manual_seed(seed_no)
    # np.random.seed(seed_no)
    # random.seed(seed_no)
    # torch.cuda.manual_seed(seed_no)
    # torch.cuda.manual_seed_all(seed_no)

    train_data = get_data(df=df_train, max_length=max_length)
    val_data = get_data(df=df_val, max_length=max_length)
    test_data = get_data(df=df_test, max_length=max_length)

    train_data = torch.LongTensor(train_data)
    val_data = torch.LongTensor(val_data)
    test_data = torch.LongTensor(test_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Arrary finished')
    atkt_model = ATKT(skill_dim=skill_emb_dim, answer_dim=answer_emb_dim, hidden_dim=hidden_emb_dim, output_dim=n_skills).to(device)

    optimizer = torch.optim.Adam(atkt_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    print('Begin to train model')
    # train hy_model 100 times
    train_loss_list = []
    train_AUC_list = []
    val_AUC_list = []
    val_loss_list=[]
    best_valid_auc = 0
    for epoch in range(epoch_num):
        loss, train_AUC = ATKT_train(train_data, atkt_model, optimizer, batch_size, epsilon, beta)

        print('train_one_epoch: ', loss, 'train_AUC:', train_AUC)
        train_loss_list.append(loss)
        train_AUC_list.append(train_AUC)

        # scheduler.step()
        print('------------------------------')
        print('------------------------------')

        val_loss, val_AUC = ATKT_test(test_data=val_data, model=atkt_model, optimizer=optimizer, batch_size=batch_size)


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
                        'model_state_dict': atkt_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        },
                        os.path.join(model_path,  'val')+'_' + str(best_epoch)
                        )

        if (val_loss - min(val_loss_list)) > 100:
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
    atkt_model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_AUC =  ATKT_test(test_data=test_data, model=atkt_model, optimizer=optimizer, batch_size=batch_size)


    print('Test_AUC', test_AUC)
    print('Best_epoch', best_epoch)

    # path = os.path.join(model_path,  'val') + '_*'
    # for i in glob.glob(path):
    #     os.remove(i)


if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	main()
