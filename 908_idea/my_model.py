# import library
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import random
from my_module import get_subsequent_mask, get_pad_mask, ScaledDotProductAttention, MultiHeadAttention, PositionwiseFeedForward, EncoderLayer, PositionalEncoding, DecoderLayer


class ActionEncoder(nn.Module):
	''' A encoder model with self attention mechanism.
	input_dim: batch_size*3*session_length
	out_dim: batch_size*3*session_length
	'''

	def __init__(
			self, n_type_correctness, n_type_problem, n_type_skill, embedding_size, n_layers, n_head, d_k, d_v,
			d_model, d_inner, dropout=0.1, n_position=200):

		super().__init__()
		self.problem_emb = nn.Embedding(n_type_problem, embedding_size)
		self.skill_emb = nn.Embedding(n_type_skill, embedding_size)
		self.correctness_emb = nn.Embedding(n_type_correctness, embedding_size)
		self.position_enc = PositionalEncoding(embedding_size, n_position=n_position)
		self.dropout = nn.Dropout(p=dropout)
		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
		self.d_model = d_model

	def forward(self, input_array, pad_mask=None, return_attns=False):

		enc_slf_attn_list = []

		# -- Forward

		# element wise addition on correctness embedding and problemID embedding
		problemId_embedding = self.problem_emb(input_array[:,0])
		skill_embedding = self.skill_emb(input_array[:,1])
		correct_embedding = self.correctness_emb(input_array[:,2])

		enc_output = correct_embedding + problemId_embedding + skill_embedding

		position_encoding = self.position_enc(enc_output)

		# print(self.position_enc)
		enc_output = self.dropout(position_encoding)
		enc_output = self.layer_norm(enc_output)

		for enc_layer in self.layer_stack:
			enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=pad_mask)
			enc_slf_attn_list += [enc_slf_attn] if return_attns else []

		if return_attns:
			return enc_output, enc_slf_attn_list
		return enc_output


class SessionEncoder(nn.Module):
	''' A encoder model with self attention mechanism.
	input_dim: batch,num_session,dmodel
	output_dim:batch,num_session,dmodel
	'''

	def __init__(
			self, embedding_size, n_layers, n_head, d_k, d_v,
			d_model, d_inner, dropout=0.1, n_position=200):

		super().__init__()
		self.position_enc = PositionalEncoding(embedding_size, n_position=n_position)
		self.dropout = nn.Dropout(p=dropout)
		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
		self.d_model = d_model

	def forward(self, input_array, pad_mask=None, return_attns=False):

		enc_slf_attn_list = []

		# -- Forward

		enc_output = self.dropout(self.position_enc(input_array))
		enc_output = self.layer_norm(enc_output)

		for enc_layer in self.layer_stack:
			enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=pad_mask)
			enc_slf_attn_list += [enc_slf_attn] if return_attns else []

		if return_attns:
			return enc_output, enc_slf_attn_list
		return enc_output


class h_decoder(nn.Module):
	''' A decoder model with self attention mechanism. '''

	def __init__(
			self, embedding_size, n_layers, n_head, d_k, d_v,
			d_model, d_inner, n_type_problem, n_type_skill,n_type_correctness, n_position=200, dropout=0.1):

		super().__init__()
		self.problem_emb = nn.Embedding(n_type_problem, embedding_size)
		self.skill_emb = nn.Embedding(n_type_skill, embedding_size)
		self.pro_pad_emb = nn.Embedding(n_type_problem, embedding_size)
		self.sk_pad_emb = nn.Embedding(n_type_skill, embedding_size)
		self.cor_pad_emb = nn.Embedding(n_type_correctness, embedding_size)

		self.position_enc = PositionalEncoding(embedding_size, n_position=n_position)
		self.dropout = nn.Dropout(p=dropout)
		self.layer_stack = nn.ModuleList([
			DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
		self.d_model = d_model

	def forward(self, trg_problem_seq, trg_skill_seq, trg_mask,
				pro_pad, skill_pad, correct_pad, enc_output, return_attns=False):

		dec_slf_attn_list = []

		# -- target seq embedding
		problemId_embedding = self.problem_emb(trg_problem_seq)
		skill_embedding = self.skill_emb(trg_skill_seq)

		# -- information in this target session(embedding)
		pro_pad_embedding = self.pro_pad_emb(pro_pad)
		skill_pad_embedding = self.sk_pad_emb(skill_pad)
		correct_pad_embedding = self.cor_pad_emb(correct_pad)


		# -- Forward
		dec_output = problemId_embedding + skill_embedding + correct_pad_embedding + enc_output

		dec_output = self.dropout(self.position_enc(dec_output))
		dec_output = self.layer_norm(dec_output)

		# enc_output = pro_pad_embedding + skill_pad_embedding + correct_pad_embedding + enc_output
		# enc_output =correct_pad_embedding + enc_output

		# enc_output = self.dropout(self.position_enc(enc_output))
		# enc_output = self.layer_norm(enc_output)

		for dec_layer in self.layer_stack:
			# dec_output, dec_slf_attn = dec_layer(dec_input=dec_output, enc_output=enc_output, slf_attn_mask=trg_mask)
			dec_output, dec_slf_attn = dec_layer(dec_input=dec_output, enc_output=dec_output, slf_attn_mask=trg_mask)

			dec_slf_attn_list += [dec_slf_attn] if return_attns else []

		if return_attns:
			return dec_output, dec_slf_attn_list
		return dec_output


class myTransformer(nn.Module):
	''' A sequence to sequence model with attention mechanism. '''

	def __init__(
			self, embedding_size=512, d_model=512, d_inner=2048,
			n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, action_position=64, session_position=16,
			n_type_correctness=3, n_type_problem=1500, n_type_skill=200):
		super().__init__()

		self.embedding_size = embedding_size
		self.d_model = d_model

		self.pooling_layer1 = nn.AdaptiveAvgPool2d((1, embedding_size))
		self.pooling_layer2 = nn.AdaptiveAvgPool2d((1, embedding_size))
		self.fc_layer1 = nn.Linear(embedding_size, 1)

		self.actionEncoder = ActionEncoder(
			n_type_correctness=n_type_correctness, n_type_problem=n_type_problem,
			n_type_skill=n_type_skill, embedding_size=embedding_size,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			d_model=d_model, d_inner=d_inner, dropout=dropout,
			n_position=action_position)

		self.sessionEncoder = SessionEncoder(
			embedding_size=embedding_size,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			d_model=d_model, d_inner=d_inner, dropout=dropout,
			n_position=session_position)

		self.my_decoder = h_decoder(
			embedding_size=embedding_size,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			d_model=d_model, d_inner=d_inner, dropout=dropout,
			n_position=action_position,
			n_type_problem=n_type_problem, n_type_skill=n_type_skill,
			n_type_correctness=n_type_correctness)

	def forward(self, input_array, session_size):
		session_input_array = []

		# action-level encoder
		for i in range(session_size):
			pad_mask = get_pad_mask(input_array[:, i, 2], 2)

			action_enc_out = self.actionEncoder(input_array=input_array[:, i], pad_mask=pad_mask)
			# action_enc_out = self.actionEncoder(input_array=input_array[:, i])
			session_input_array.append(self.pooling_layer1(action_enc_out))

		# session-level encoder
		session_input_array = torch.cat((session_input_array), 1)

		#-- Session padding mask
		session_pad_mask = get_pad_mask(input_array[:,:-1,2,0], 2)

		session_enc_out = self.sessionEncoder(input_array=session_input_array, pad_mask=session_pad_mask)

		all_session_enc_out = self.pooling_layer2(session_enc_out)

		target_problem_seq = input_array[:, session_size, 0]
		target_skill_seq = input_array[:, session_size, 1]

		# Get target mask
		trg_mask = get_subsequent_mask(target_problem_seq)

		# Decoder
		dec_out = self.my_decoder(trg_problem_seq=target_problem_seq, trg_skill_seq=target_skill_seq,
								  pro_pad=input_array[:, session_size, 3], skill_pad=input_array[:, session_size, 4],
								  correct_pad=input_array[:, session_size, 5], enc_output=all_session_enc_out,
								  trg_mask=trg_mask)
		dec_out = self.fc_layer1(dec_out)
		dec_out = nn.Sigmoid()(dec_out.squeeze(2))

		return dec_out
