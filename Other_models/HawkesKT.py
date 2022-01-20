# This is referenced from https://github.com/THUwangcy/HawkesKT

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
                
                data.append([k_out,q_out,a_out,t_out])

        

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



class HawkesKT(nn.Module):

    def __init__(self,n_problems, n_skills, emb_size, time_log):
        super().__init__()
        self.problem_num = int(n_problems)
        self.skill_num = int(n_skills)
        self.emb_size = emb_size
        self.time_log = time_log
        self.problem_base = torch.nn.Embedding(self.problem_num, 1)
        self.skill_base = torch.nn.Embedding(self.skill_num, 1)

        self.alpha_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.alpha_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
        self.beta_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.beta_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)

    @staticmethod
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        elif type(m) in [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


    def forward(self, input):
        skills = input[:,0]      # [batch_size, seq_len]
        problems = input[:,1]  # [batch_size, seq_len]
        times = input[:,3]       # [batch_size, seq_len]
        labels = input[:,2]      # [batch_size, seq_len]

        mask_labels = labels * (labels < 2).long()
        inters = skills + mask_labels * self.skill_num
        alpha_src_emb = self.alpha_inter_embeddings(inters)  # [bs, seq_len, emb]
        alpha_target_emb = self.alpha_skill_embeddings(skills)
        alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
        beta_src_emb = self.beta_inter_embeddings(inters)  # [bs, seq_len, emb]
        beta_target_emb = self.beta_skill_embeddings(skills)
        betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
        betas = torch.clamp(betas + 1, min=0, max=10)


        delta_t = (times[:, :, None] - times[:, None, :]).abs().double()
        delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

        cross_effects = alphas * torch.exp(-betas * delta_t)


        seq_len = skills.shape[1]
        valid_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
        mask = (torch.from_numpy(valid_mask) == 0)
        mask = mask.to(device)
        sum_t = cross_effects.masked_fill(mask, 0).sum(-2)

        problem_bias = self.problem_base(problems).squeeze(dim=-1)
        skill_bias = self.skill_base(skills).squeeze(dim=-1)

        prediction = (problem_bias + skill_bias + sum_t).sigmoid()

        return prediction




def hkkt_train(train_data, model, optimizer, batch_size):
    """Train SAKT model.
    Arguments:
        train_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        batch_size (int)
    """
    Plabel_list = torch.Tensor([])
    Ground_true_list = torch.Tensor([])
    loss_total = 0

    CEL = nn.BCELoss(reduction='sum')

    train_batches = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    model.train()

    # (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels)
    # Training
    for input in train_batches:
        input = input.to(device)

        preds = model(input)

        label_mask = (input[:,2,1:] != 2)
        z = preds[:,1:][label_mask]
        label_seq = input[:,2,1:][label_mask]

        if z.size() == torch.Size([]):
            z = torch.stack([z])

        one_batch_loss = CEL(z, label_seq.double())

        z = z.to('cpu')
        label_seq = label_seq.to('cpu')

        Plabel_list = torch.cat((Plabel_list, z))
        Ground_true_list = torch.cat((Ground_true_list, label_seq))

        one_batch_loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        loss_total += one_batch_loss.to('cpu')
        del input

    train_auc =  metrics.roc_auc_score(Ground_true_list.detach().numpy().astype(int),
                                            Plabel_list.detach().numpy())

    return float(loss_total), train_auc


def hkkt_test(test_data, model, optimizer, batch_size):
    """Train SAKT model.
    Arguments:
        test_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        batch_size (int)
    """
    Plabel_list = torch.Tensor([])
    Ground_true_list = torch.Tensor([])
    loss_total = 0

    CEL = nn.BCELoss(reduction='sum')
    test_batches = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model.eval()

    # Test
    for input in test_batches:
        input = input.to(device)
        with torch.no_grad():
            preds = model(input)

        label_mask = (input[:,2,1:] != 2) 
        z = preds[:,1:][label_mask]
        label_seq = input[:,2,1:][label_mask]

        if z.size() == torch.Size([]):
            z = torch.stack([z])

        one_batch_loss = CEL(z, label_seq.double())

        z = z.to('cpu')
        label_seq = label_seq.to('cpu')

        Plabel_list = torch.cat((Plabel_list, z))
        Ground_true_list = torch.cat((Ground_true_list, label_seq))        

        loss_total += one_batch_loss.to('cpu')
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

    n_skills= max(df.skill)+1 #question is skill
    n_problems = max(df.problemId)+1  #problem is problemId
    emb_size=64 
    time_log=5.0
    lr=1e-3
    batch_size= 100
    max_length=200

    epoch_num = 200
    seed_no = 123

    # model path
    model_path = './hkkt_model1'

    # # seed
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

    print('Arrary finished')
    hkkt_model = HawkesKT(n_problems=n_problems, n_skills=n_skills, emb_size=emb_size, time_log=time_log).to(device)
    hkkt_model = hkkt_model.double()
    hkkt_model.apply(hkkt_model.init_weights)
    optimizer = torch.optim.Adam(hkkt_model.parameters(), lr=lr)

    print('Begin to train model')
    # train hy_model 100 times
    train_loss_list = []
    train_AUC_list = []
    val_AUC_list = []
    val_loss_list=[]
    best_valid_auc = 0

    # checkpoint = torch.load(os.path.join(model_path,  'val_8'))
    # hkkt_model.load_state_dict(checkpoint['model_state_dict'])

    for epoch in range(8, epoch_num):
        loss, train_AUC = hkkt_train(train_data=train_data, model=hkkt_model, optimizer=optimizer, batch_size=batch_size)

        print('train_one_epoch: ', loss, 'train_AUC:', train_AUC)
        train_loss_list.append(loss)
        train_AUC_list.append(train_AUC)

        # scheduler.step()
        print('------------------------------')
        print('------------------------------')

        val_loss, val_AUC = hkkt_test(test_data=val_data, model=hkkt_model, optimizer=optimizer, batch_size=batch_size)


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
                        'model_state_dict': hkkt_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        },
                        os.path.join(model_path,  'val')+'_' + str(best_epoch)
                        )

        # if (val_loss - min(val_loss_list)) > 5000:
        #     break

    print('------------------------------')
    print('train_loss_list', train_loss_list)
    print('train_AUC_list', train_AUC_list)
    print('VAL AUC List:', val_AUC_list)
    print('val loss List:', val_loss_list)
    print('max_val_auc:',max(val_AUC_list))
    print('------------------------------')
    print('Begin to test.........')

    # checkpoint = torch.load(os.path.join(model_path,  'val') + '_'+str(best_epoch))
    # hkkt_model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_AUC =  hkkt_test(test_data=test_data, model=hkkt_model, optimizer=optimizer, batch_size=batch_size)


    print('Test_AUC', test_AUC)
    print('Best_epoch', best_epoch)

    path = os.path.join(model_path,  'val') + '_*'
    for i in glob.glob(path):
        os.remove(i)


if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	main()
