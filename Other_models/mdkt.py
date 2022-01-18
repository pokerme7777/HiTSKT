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


class DKT(nn.Module):

    def __init__(self, n_skills, emb_size, hidden_size, num_layer, dropout):
        super().__init__()
        self.skill_num = n_skills
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.dropout = dropout

        self.skill_embeddings = torch.nn.Embedding(self.skill_num * 3, self.emb_size)
        self.rnn = torch.nn.LSTM(
            input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True,
            num_layers=self.num_layer
        )
        self.out = torch.nn.Linear(self.hidden_size, self.skill_num)

        self.loss_function = torch.nn.BCELoss()

    def forward(self, input_array ):
        seq_sorted = input_array[:,0]      # [batch_size, seq_len]
        labels_sorted = input_array[:,2]      # [batch_size, seq_len]
        lengths = torch.from_numpy(np.array(list(map(lambda lst: len(lst), seq_sorted))))      # [batch_size]

        embed_history_i = self.skill_embeddings(seq_sorted + labels_sorted * self.skill_num)
        embed_history_i_packed = torch.nn.utils.rnn.pack_padded_sequence(embed_history_i, lengths-1, batch_first=True)
        output, hidden = self.rnn(embed_history_i_packed, None)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        pred_vector = self.out(output)
        target_item = seq_sorted[:, 1:]
        prediction_sorted = torch.gather(pred_vector, dim=-1, index=target_item.unsqueeze(dim=-1)).squeeze(dim=-1)
        label = labels_sorted[:, 1:]

        prediction = torch.sigmoid(prediction_sorted)
        
        return prediction

def dkt_train(train_data, model, optimizer, batch_size):
    """Train DKT model.
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

        preds = model( input_array=input)

        label_mask = (input[:,2,1:] != 2)
        z = preds[label_mask]
        label_seq = input[:,2,1:][label_mask]

        if z.size() == torch.Size([]):
            z = torch.stack([z])

        one_batch_loss = CEL(z, label_seq.float())

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


def dkt_test(test_data, model, optimizer, batch_size):
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

    CEL = nn.BCELoss(reduction='sum')
    test_batches = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model.eval()

    # Test
    for input in test_batches:
        input = input.to(device)
        with torch.no_grad():
            preds = model( input_array=input)

        label_mask = (input[:,2,1:] != 2) 
        z = preds[label_mask]
        label_seq = input[:,2,1:][label_mask]

        if z.size() == torch.Size([]):
            z = torch.stack([z])

        one_batch_loss = CEL(z, label_seq.float())

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

    n_skills= max(df.skill) + 2  #question is skill
    n_problems = max(df.problemId) + 2  #problem is problemId

    emb_size=64
    hidden_size=64
    num_layer=1
    dropout=0.1
    max_length=200

    seed_no=123
    epoch_num = 200
    lr = 2e-4
    batch_size = 20

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    print('Arrary finished')
    dkt_model = DKT(n_skills=n_skills, emb_size=emb_size, hidden_size=hidden_size, num_layer=num_layer, dropout=dropout).to(device)

    optimizer = torch.optim.Adam(dkt_model.parameters(), lr=lr)

    # train hy_model 100 times
    train_loss_list = []
    train_AUC_list = []
    val_AUC_list = []
    val_loss_list=[]
    best_valid_auc = 0

    # model path
    model_path = 'dkt_model'

    for epoch in range(epoch_num):
        loss, train_AUC = dkt_train(train_data=train_data, model=dkt_model, optimizer=optimizer, batch_size=batch_size)

        print('train_one_epoch: ', loss, 'train_AUC:', train_AUC)
        train_loss_list.append(loss)
        train_AUC_list.append(train_AUC)

        # scheduler.step()
        print('------------------------------')
        print('------------------------------')

        val_loss, val_AUC = dkt_test(test_data=val_data, model=dkt_model, optimizer=optimizer, batch_size=batch_size)


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
                        'model_state_dict': dkt_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        },
                        os.path.join(model_path,  'val')+'_' + str(best_epoch)
                        )

        # if (val_loss - min(val_loss_list)) > 20000:
        #     break

    print('------------------------------')
    print('train_loss_list', train_loss_list)
    print('train_AUC_list', train_AUC_list)
    print('VAL AUC List:', val_AUC_list)
    print('val loss List:', val_loss_list)
    print('max_val_auc:',max(val_AUC_list))
    print('------------------------------')
    print('Begin to test.........')

    checkpoint = torch.load(os.path.join(model_path,  'val') + '_'+str(best_epoch))
    dkt_model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_AUC =  dkt_test(test_data=test_data, model=dkt_model, optimizer=optimizer, batch_size=batch_size)


    print('Test_AUC', test_AUC)
    print('Best_epoch', best_epoch)

    path = os.path.join(model_path,  'val') + '_*'
    for i in glob.glob(path):
        os.remove(i)


if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	main()
