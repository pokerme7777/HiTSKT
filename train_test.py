# import library
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import metrics
import random
import math


def train(my_model, optimizer, training_array, device, batch_size, session_size, padding_correct_value, EOS_C_value, padding_node, EOS_skill):
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

		model_predict_list = my_model(input_array=input_array, session_size=session_size, padding_correct_value=padding_correct_value, padding_node=padding_node, EOS_skill=EOS_skill)

		# Cross Entropy
		CEL = nn.BCELoss(reduction='sum')


		label_mask = (input_array[:,session_size,2] != padding_correct_value) & ( input_array[:,session_size,2] != EOS_C_value)
		z = model_predict_list[label_mask]
		label_seq = input_array[:,session_size,2][label_mask]

		if z.size() == torch.Size([]):
			z = torch.stack([z])

		one_batch_loss = CEL(z, label_seq.float())

		z = z.to('cpu')
		label_seq = label_seq.to('cpu')

		Plabel_list = torch.cat((Plabel_list, z))
		Ground_true_list = torch.cat((Ground_true_list, label_seq))

		one_batch_loss.backward()
		optimizer.step()

		one_epoch_loss += one_batch_loss.to('cpu')
		input_array = input_array.to('cpu')
		batch_no += 1

	return float(one_epoch_loss), Plabel_list, Ground_true_list



def test(my_model, optimizer, test_array, device, batch_size, session_size, padding_correct_value, EOS_C_value, padding_node, EOS_skill):
	''' Test model for h_transformer
	'''

	dataset = DataLoader(test_array, batch_size=batch_size, shuffle=True)

	Plabel_list = torch.Tensor([])
	Ground_true_list = torch.Tensor([])
	one_epoch_loss = 0

	for input_array in dataset:
		input_array = input_array.to(device)


		my_model.eval()
		with torch.no_grad():
			model_predict_list = my_model(input_array=input_array, session_size=session_size, padding_correct_value=padding_correct_value, padding_node=padding_node, EOS_skill=EOS_skill)

		# Cross Entropy
		CEL = nn.BCELoss(reduction='sum')


		label_mask = (input_array[:,session_size,2] != padding_correct_value) &( input_array[:,session_size,2] != EOS_C_value)
		z = model_predict_list[label_mask]
		label_seq = input_array[:,session_size,2][label_mask]

		if z.size() == torch.Size([]):
			z = torch.stack([z])

		one_batch_loss = CEL(z, label_seq.float())

		z = z.to('cpu')
		label_seq = label_seq.to('cpu')

		Plabel_list = torch.cat((Plabel_list, z))
		Ground_true_list = torch.cat((Ground_true_list, label_seq))

		
		one_epoch_loss += one_batch_loss.to('cpu')
		input_array = input_array.to('cpu')

	return float(one_epoch_loss), Plabel_list, Ground_true_list
