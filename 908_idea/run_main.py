# import library
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
import random
from my_pre_train import read_data_into_array
from my_model import myTransformer



def train(my_model, optimizer, training_array, device, batch_size, session_size):
	''' Training model for h_transformer
	'''

	dataset = DataLoader(training_array, batch_size=batch_size, shuffle=True)

	Plabel_list = torch.Tensor([])
	Ground_true_list = torch.Tensor([])

	one_epoch_loss = 0
	batch_no = 0

	for input_array in dataset:
		input_array = input_array.to(device)
		my_model.train()
		optimizer.zero_grad()

		model_predict_list = my_model(input_array=input_array, session_size=session_size)

		# Cross Entropy
		CEL = nn.BCELoss(reduction='sum')

		one_batch_loss = 0

		for i in range(len(input_array)):

			label_seq = torch.stack([lab for lab in input_array[i, session_size, 2] if lab != 2])
			one_len = len(label_seq)

			z = model_predict_list[i][:one_len]

			if z.size() == torch.Size([]):
				z = torch.stack([z])

			one_loss = CEL(z, label_seq.float())
			one_batch_loss += one_loss

			z = z.to('cpu')
			label_seq = label_seq.to('cpu')

			Plabel_list = torch.cat((Plabel_list, z))
			Ground_true_list = torch.cat((Ground_true_list, label_seq))

		one_batch_loss.backward()
		optimizer.step()

		one_epoch_loss += one_batch_loss.to('cpu')

		# print(batch_no, '_batch_loss: ', one_batch_loss)
		batch_no += 1
		input_array = input_array.to('cpu')

	return float(one_epoch_loss), Plabel_list, Ground_true_list


def validation(my_model, optimizer, val_array, device, batch_size, session_size):
	''' val for Training model
	'''

	dataset = DataLoader(val_array, batch_size=batch_size, shuffle=True)

	one_epoch_loss = 0

	for input_array in dataset:
		input_array = input_array.to(device)
		my_model.eval()
		with torch.no_grad():
			model_predict_list = my_model(input_array=input_array, session_size=session_size)

		# Cross Entropy
		CEL = nn.BCELoss(reduction='sum')

		one_batch_loss = 0

		for i in range(len(input_array)):

			label_seq = torch.stack([lab for lab in input_array[i, session_size, 2] if lab != 2])
			one_len = len(label_seq)

			z = model_predict_list[i][:one_len]

			if z.size() == torch.Size([]):
				z = torch.stack([z])

			one_loss = CEL(z, label_seq.float())
			one_batch_loss += one_loss

		one_epoch_loss += one_batch_loss.to('cpu')
		input_array = input_array.to('cpu')

	return float(one_epoch_loss)

def test(my_model, optimizer, test_array, device, batch_size, session_size):
	''' Test model for h_transformer
	'''

	dataset = DataLoader(test_array, batch_size=batch_size, shuffle=True)

	Plabel_list = torch.Tensor([])
	Ground_true_list = torch.Tensor([])

	for input_array in dataset:
		input_array = input_array.to(device)

		my_model.eval()
		with torch.no_grad():
			model_predict_list = my_model(input_array=input_array, session_size=session_size)


		for i in range(len(input_array)):

			label_seq = torch.stack([lab for lab in input_array[i, session_size, 2] if lab != 2])
			one_len = len(label_seq)

			z = model_predict_list[i][:one_len]

			if z.size() == torch.Size([]):
				z = torch.stack([z])

			z = z.to('cpu')
			label_seq = label_seq.to('cpu')

			Plabel_list = torch.cat((Plabel_list, z))
			Ground_true_list = torch.cat((Ground_true_list, label_seq))

		input_array = input_array.to('cpu')


	return Plabel_list, Ground_true_list


def main():
	'''
	The main function of the training script
	'''
	df = pd.read_csv('./2017_clean_full.csv', low_memory=False)
	df_train = pd.read_csv('./2017train1.csv', low_memory=False)
	df_val = pd.read_csv('./2017val1.csv', low_memory=False)
	df_test = pd.read_csv('./2017test1.csv', low_memory=False)

	# Create Variable
	batch_size = 32
	session_size = 16
	embedding_size = 512
	action_size = 64
	padding_correct_value = 2
	padding_node = 0

	num_problem = len(df.problemId.unique().tolist())+1
	num_skill = len(df.skill.unique().tolist())+1

	epoch_num = 40
	seed_no = 123
	learning_rate = 5e-6


	# seed
	np.random.seed(seed_no)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.manual_seed(seed_no)
	np.random.seed(seed_no)
	random.seed(seed_no)
	torch.cuda.manual_seed(seed_no)
	torch.cuda.manual_seed_all(seed_no)


	print("File loaded")

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Get training array
	training_array = read_data_into_array(df=df_train, padding_node=padding_node,
											padding_correct_value=padding_correct_value, action_size=action_size, session_size=session_size)


	test_array = read_data_into_array(df=df_test, padding_node=padding_node,
											padding_correct_value=padding_correct_value, action_size=action_size, session_size=session_size)

	val_array = read_data_into_array(df=df_val, padding_node=padding_node,
											padding_correct_value=padding_correct_value, action_size=action_size, session_size=session_size)

	print("loaded array")

	print("Begin to train model")
	training_array = torch.LongTensor(training_array)

	test_array = torch.LongTensor(test_array)

	val_array = torch.LongTensor(val_array)


	# get my model
	my_model = myTransformer(embedding_size=embedding_size, d_model=embedding_size, d_inner=2048,
								n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, action_position=action_size,
								session_position=session_size, n_type_correctness=3, n_type_problem=num_problem,
							n_type_skill=num_skill)

	my_model = my_model.to(device)

	optimizer1 = torch.optim.Adam(my_model.parameters(), betas=(0.9, 0.999), lr=learning_rate, eps=1e-8)

	# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.99, last_epoch=-1)

	# train hy_model 100 times
	train_loss_list = []
	train_AUC_list = []
	val_loss_list = [0]
	test_AUC_list = []
	for epoch in range(epoch_num):
		loss, Plabel, Ground_true = train(my_model=my_model, optimizer=optimizer1, training_array=training_array,
											device=device, batch_size=batch_size, session_size=session_size)

		one_AUC = metrics.roc_auc_score(Ground_true.detach().numpy().astype(int), Plabel.detach().numpy())

		print('train_one_epoch: ', loss, 'train_AUG:', one_AUC)
		train_loss_list.append(loss)
		train_AUC_list.append(one_AUC)

		# scheduler.step()
		print('train_loss_list', train_loss_list)
		print('train_AUC_list', train_AUC_list)
		print('------------------------------')
		print(epoch, ' Prediction1:', Plabel[:20])
		print(epoch, ' Label1:', Ground_true[:20])
		print('------------------------------')

		val_loss = validation(my_model=my_model, optimizer=optimizer1, val_array=val_array,
											device=device, batch_size=batch_size, session_size=session_size)

		if (epoch % 10 == 0) or (epoch == epoch_num-1):
			# Testing
			test_Plabel, test_Ground_true = test(my_model=my_model, optimizer=optimizer1, test_array=test_array,
													device=device, batch_size=batch_size, session_size=session_size)

			test_AUC = metrics.roc_auc_score(test_Ground_true.detach().numpy().astype(int),
												test_Plabel.detach().numpy())
			print('------------------------------')
			print('Epoch: ',epoch ,' Test AUC:', test_AUC)
			test_AUC_list.append(test_AUC)
			print('------------------------------------------')
			print('Epoch: ',epoch,' Test_Prediction:', test_Plabel[:10])
			print('------------------------------------------')
			print('Epoch: ',epoch,' Test_Label:', test_Ground_true[:10])

		if abs(val_loss - val_loss_list[-1])<0.0001:
			val_loss_list.append(val_loss)
			break


	# # Testing
	# test_Plabel, test_Ground_true = test(my_model=my_model, optimizer=optimizer1, test_array=test_array,
	# 								  device=device, batch_size=batch_size, session_size=session_size)
	#
	#
	# test_AUC = metrics.roc_auc_score(test_Ground_true.detach().numpy().astype(int), test_Plabel.detach().numpy())
	print('------------------------------')
	print('Test AUC Lit:', test_AUC_list)


if __name__ == '__main__':
	main()