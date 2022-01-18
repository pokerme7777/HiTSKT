# import library
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
import random
import math
import os
import os.path
import glob
import argparse
from loading_array import read_data_into_array
from model_layers import myTransformer
from train_test import train, test


def main():
	'''
	The main function of the training script
	'''

	# Parse Arguments
	parser = argparse.ArgumentParser(description='Script for KT experiment')


	parser.add_argument('--epoch_num', type=int, default=100,
						help='number of iterations')
	parser.add_argument('--batch_size', type=int, default=64,
						help='number of batch')
	parser.add_argument('--session_size', type=int, default=16,
						help='number of sessions')
	parser.add_argument('--action_size', type=int, default=64,
						help='number of actions in each session')
	parser.add_argument('--embedding_size', type=int, default=256,
						help='embedding dimensions')
	parser.add_argument('--learning_rate', type=int, default=5e-5,
						help='learning rate')
	parser.add_argument('--d_inner', type=int, default=2048,
						help='FFN dimension')
	parser.add_argument('--n_layers', type=int, default=1,
						help='number of layers')
	parser.add_argument('--n_head', type=int, default=4,
						help='number of layers')
	parser.add_argument('--d_k', type=int, default=64,
						help='k query dimensions')
	parser.add_argument('--d_v', type=int, default=64,
						help='v query dimensions')
	parser.add_argument('--dropout', type=int, default=0.1,
						help='dropout')


	params = parser.parse_args()
	dataset_name = params.dataset

	df = pd.read_csv(dataset_name, low_memory=False)

	# Create Variable
	batch_size = params.batch_size
	session_size =params.session_size
	embedding_size = params.embedding_size
	action_size = params.action_size
	padding_correct_value = 2
	padding_node = 0

	num_problem = max(df.problemId.unique().tolist())+3 # (padding, EOS, Session EOS)
	num_skill = max(df.skill.unique().tolist())+3 # (padding, EOS, Session EOS)
	num_qno = max(df.question_no.unique().tolist())+3 # (padding, EOS, Session EOS)
	num_stuno = max(df.studentId.unique().tolist())+3 # (padding, EOS, Session EOS)

	#-- EOS value
	EOS_quesiton = num_problem-2
	EOS_skill = num_skill-2
	EOS_C_value = padding_correct_value+1
	EOS_q_no = num_qno-2
	EOS_stu_no = num_stuno-2

	session_EOS_question = num_problem -1
	session_EOS_skill = num_skill -1
	session_EOS_c_value = EOS_C_value + 1
	SEOS_q_no = num_qno-1
	SEOS_stu_no = num_stuno-1


	BOS_c_Value = session_EOS_c_value +1

	n_type_correctness = 6 #-- (0, 1, padding, EOS, Session EOS, BOS)

	#-- model parameter
	learning_rate = params.learning_rate
	d_inner=params.d_inner

	n_layers=params.n_layers
	n_head=params.n_head
	d_k=params.d_k
	d_v=params.d_v
	dropout=params.dropout

	# model path
	model_path = './model2'

	epoch_num = params.epoch_num
	seed_no = 123

	# seed
	# np.random.seed(seed_no)
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False
	# torch.manual_seed(seed_no)
	# np.random.seed(seed_no)
	# random.seed(seed_no)
	# torch.cuda.manual_seed(seed_no)
	# torch.cuda.manual_seed_all(seed_no)


	print("File loaded")

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Get training array
	training_array,val_array,test_array = read_data_into_array(df=df, padding_node=padding_node,
											padding_correct_value=padding_correct_value, action_size=action_size, session_size=session_size,
											EOS_quesiton=EOS_quesiton, EOS_skill=EOS_skill, EOS_C_value=EOS_C_value, BOS_c_Value=BOS_c_Value,
											session_EOS_question=session_EOS_question, session_EOS_skill=session_EOS_skill, session_EOS_c_value=session_EOS_c_value,
											EOS_q_no=EOS_q_no, EOS_stu_no=EOS_stu_no, SEOS_q_no=SEOS_q_no, SEOS_stu_no=SEOS_stu_no)


	print("loaded array")


	print("Begin to train model")
	training_array = torch.LongTensor(training_array)
	val_array = torch.LongTensor(val_array)
	test_array = torch.LongTensor(test_array)

	print('train size:', training_array.size())
	print('val size:', val_array.size())
	print('test size:', test_array.size())
	# get my model
	my_model = myTransformer(embedding_size=embedding_size, d_model=embedding_size, d_inner=d_inner,
								n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout, action_position=action_size,
								session_position=session_size, n_type_correctness=n_type_correctness, n_type_problem=num_problem,
							n_type_skill=num_skill,session_head=n_head, session_layer=n_layers, n_type_qno=num_qno)

	my_model = my_model.to(device)

	optimizer1 = torch.optim.Adam(my_model.parameters(), betas=(0.9, 0.999), lr=learning_rate, eps=1e-8)

	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=60, gamma=0.6, last_epoch=-1, verbose=False)

	# train hy_model 100 times
	train_loss_list = []
	train_AUC_list = []
	val_AUC_list = []
	val_loss_list=[]
	best_valid_auc = 0
	for epoch in range(epoch_num):
		loss, Plabel, Ground_true = train(my_model=my_model, optimizer=optimizer1, training_array=training_array,
											device=device, batch_size=batch_size, session_size=session_size,
											padding_correct_value=padding_correct_value, EOS_C_value=EOS_C_value,
											padding_node=padding_node, EOS_skill=EOS_skill)

		one_AUC = metrics.roc_auc_score(Ground_true.detach().numpy().astype(int), Plabel.detach().numpy())

		print('train_one_epoch: ', loss, 'train_AUG:', one_AUC)
		train_loss_list.append(loss)
		train_AUC_list.append(one_AUC)

		# scheduler.step()
		print('------------------------------')
		print('------------------------------')

		val_loss, val_Plabel, val_Ground_true = test(my_model=my_model, optimizer=optimizer1, test_array=val_array,
												device=device, batch_size=batch_size, session_size=session_size,
												padding_correct_value=padding_correct_value, EOS_C_value=EOS_C_value,
												padding_node=padding_node, EOS_skill=EOS_skill)

		val_AUC = metrics.roc_auc_score(val_Ground_true.detach().numpy().astype(int),
											val_Plabel.detach().numpy())
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
						'model_state_dict': my_model.state_dict(),
						'optimizer_state_dict': optimizer1.state_dict(),
						'loss': loss,
						},
						os.path.join(model_path,  'val')+'_' + str(best_epoch)
						)

		if (val_loss - min(val_loss_list)) > 10000:
			break

	print('------------------------------')
	print('train_loss_list', train_loss_list)
	print('train_AUC_list', train_AUC_list)
	print('VAL AUC List:', val_AUC_list)
	print('val loss List:', val_loss_list)
	print('max_val_auc:',max(val_AUC_list))
	print('------------------------------')
	print('Begin to test.........')

	checkpoint = torch.load(os.path.join(model_path,  'val') + '_' +str(best_epoch))
	my_model.load_state_dict(checkpoint['model_state_dict'])

	test_loss, test_Plabel, test_Ground_true = test(my_model=my_model, optimizer=optimizer1, test_array=test_array,
											device=device, batch_size=batch_size, session_size=session_size,
											padding_correct_value=padding_correct_value, EOS_C_value=EOS_C_value,
											padding_node=padding_node, EOS_skill=EOS_skill)

	test_AUC = metrics.roc_auc_score(test_Ground_true.detach().numpy().astype(int),
										test_Plabel.detach().numpy())

	print('Test_AUC', test_AUC)
	print('Best_epoch', best_epoch)

	# path = os.path.join(model_path,  'val') + '_*'
	# for i in glob.glob(path):
	# 	os.remove(i)

if __name__ == '__main__':
	main()
