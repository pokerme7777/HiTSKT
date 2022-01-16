# import library
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from mask_scr import get_pad_mask, get_subsequent_mask
from sublayers import EncoderLayer, PositionalEncoding, DecoderLayer


class ActionEncoder(nn.Module):
	''' A encoder model with self attention mechanism.
	input_dim: batch_size*3*session_length
	out_dim: batch_size*3*session_length
	'''

	def __init__(
			self, n_type_correctness, n_type_problem, n_type_skill, embedding_size, n_layers, n_head, d_k, d_v,
			d_model, d_inner, n_type_qno, n_type_kno, dropout=0.1, n_position=200):

		super().__init__()
		self.problem_emb = nn.Embedding(n_type_problem, embedding_size)
		self.skill_emb = nn.Embedding(n_type_skill, embedding_size)
		self.correctness_emb = nn.Embedding(n_type_correctness, embedding_size)
		self.qno_emb = nn.Embedding(n_type_qno, embedding_size)
		self.position_enc = PositionalEncoding(embedding_size, n_position=n_position, encoder_type='Action')
		self.dropout = nn.Dropout(p=dropout)
		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, encoder_type='Action')
			for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
		self.d_model = d_model

	def forward(self, input_array, pad_mask=None, return_attns=False):

		enc_slf_attn_list = []

		# -- Forward

		# element wise addition on correctness embedding and problemID embedding
		problemId_embedding = self.problem_emb(input_array[:,:,0])
		skill_embedding = self.skill_emb(input_array[:,:,1])
		correct_embedding = self.correctness_emb(input_array[:,:,2])
		qno_embedding = self.qno_emb(input_array[:,:,3])

		enc_output = correct_embedding + problemId_embedding + skill_embedding + qno_embedding

		position_encoding = self.position_enc(enc_output)

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
		self.position_enc = PositionalEncoding(embedding_size, n_position=n_position, encoder_type='Session')
		self.dropout = nn.Dropout(p=dropout)
		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, encoder_type='Session')
			for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
		self.d_model = d_model

	def forward(self, input_array, pad_mask=None, return_attns=False):

		enc_slf_attn_list = []

		# -- Forward
		enc_output = self.dropout(self.position_enc(input_array))
		enc_output = self.layer_norm(enc_output)

		for enc_layer in self.layer_stack:
			enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=pad_mask, Time_affect=True)
			enc_slf_attn_list += [enc_slf_attn] if return_attns else []

		if return_attns:
			return enc_output, enc_slf_attn_list
		return enc_output

class CorrectPadEncoder(nn.Module):
	''' A encoder model with self attention mechanism.
	input_dim: batch,num_session,dmodel
	output_dim:batch,num_session,dmodel
	'''

	def __init__(
			self, embedding_size, n_layers, n_head, d_k, d_v,
			d_model, d_inner, n_type_correctness, dropout=0.1, n_position=1000):

		super().__init__()
		self.position_enc = PositionalEncoding(embedding_size, n_position=n_position, encoder_type='PC')
		self.cor_pad_emb = nn.Embedding(n_type_correctness, embedding_size)

		self.dropout = nn.Dropout(p=dropout)
		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, encoder_type='PC')
			for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
		self.d_model = d_model

	def forward(self, input_array, correct_pad, pad_mask=None, return_attns=False):

		enc_slf_attn_list = []

		# -- Forward
		correct_pad_embedding = self.cor_pad_emb(correct_pad)

		enc_output = correct_pad_embedding + input_array

		enc_output = self.dropout(self.position_enc(enc_output))
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
			d_model, d_inner, n_type_problem, n_type_skill, n_type_qno, n_type_kno, n_position=200, dropout=0.1):

		super().__init__()
		self.problem_emb = nn.Embedding(n_type_problem, embedding_size)
		self.skill_emb = nn.Embedding(n_type_skill, embedding_size)
		self.qno_emb = nn.Embedding(n_type_qno, embedding_size)
		self.position_enc = PositionalEncoding(embedding_size, n_position=n_position, encoder_type='Dec')
		self.dropout = nn.Dropout(p=dropout)
		self.layer_stack = nn.ModuleList([
			DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, encoder_type='Dec')
			for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
		self.d_model = d_model

	def forward(self, trg_problem_seq, trg_skill_seq, trg_mask, pad_mask,
				enc_output, target_qno, target_kno, return_attns=False):

		dec_slf_attn_list = []

		# -- target seq embedding
		problemId_embedding = self.problem_emb(trg_problem_seq)
		skill_embedding = self.skill_emb(trg_skill_seq)
		qno_embedding = self.qno_emb(target_qno)

		# -- Forward
		dec_output = problemId_embedding + skill_embedding + qno_embedding

		dec_output = self.dropout(self.position_enc(dec_output))
		dec_output = self.layer_norm(dec_output)


		for dec_layer in self.layer_stack:
			dec_output, dec_slf_attn = dec_layer(dec_input=dec_output, enc_output=enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=pad_mask)

			dec_slf_attn_list += [dec_slf_attn] if return_attns else []

		if return_attns:
			return dec_output, dec_slf_attn_list
		return dec_output


class myTransformer(nn.Module):
	''' A sequence to sequence model with attention mechanism. '''

	def __init__(
			self, embedding_size=512, d_model=512, d_inner=2048,
			n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, action_position=64, session_position=16,
			n_type_correctness=6, n_type_problem=1500, n_type_skill=200,
			session_head=8, session_layer=6, n_type_qno=2000, n_type_kno=2000):
		super().__init__()

		self.embedding_size = embedding_size
		self.d_model = d_model
		self.fc_layer1 = nn.Linear(embedding_size, 1)
		self.cor_pad_emb = nn.Embedding(n_type_correctness, embedding_size)

		self.actionEncoder = ActionEncoder(
			n_type_correctness=n_type_correctness, n_type_problem=n_type_problem,
			n_type_skill=n_type_skill, embedding_size=embedding_size,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			d_model=d_model, d_inner=d_inner, dropout=dropout,
			n_position=action_position, n_type_qno=n_type_qno, n_type_kno=n_type_kno)

		self.sessionEncoder = SessionEncoder(
			embedding_size=embedding_size,
			n_layers=session_layer, n_head=session_head, d_k=d_k, d_v=d_v,
			d_model=d_model, d_inner=d_inner, dropout=dropout,
			n_position=session_position)

		self.PCEncoder = CorrectPadEncoder(
			embedding_size=embedding_size,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			d_model=d_model, d_inner=d_inner, dropout=dropout,
			n_type_correctness=n_type_correctness, n_position=action_position)

		self.my_decoder = h_decoder(
			embedding_size=embedding_size,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			d_model=d_model, d_inner=d_inner, dropout=dropout,
			n_position=action_position,
			n_type_problem=n_type_problem, n_type_skill=n_type_skill,
			n_type_qno=n_type_qno, n_type_kno=n_type_kno)

	def forward(self, input_array, session_size, padding_correct_value, padding_node, EOS_skill):
		session_input_array = []

		pad_mask = get_pad_mask(input_array[:, :session_size, 2], padding_correct_value)

		action_enc_out = self.actionEncoder(input_array=input_array[:, :session_size], pad_mask=pad_mask)

		session_input_array = action_enc_out[:,:,-1,:]

		#-- Session padding mask
		session_pad_mask = get_subsequent_mask(input_array[:,:-1,2,0]) & get_pad_mask(input_array[:,:-1,2,0], padding_correct_value)

		session_enc_out = self.sessionEncoder(input_array=session_input_array, pad_mask=session_pad_mask)

		all_session_enc_out = session_enc_out[:,-1].unsqueeze(1)

		#-- PC encoder
		correct_pad = input_array[:, session_size, 5]
		correct_pad_embedding = self.cor_pad_emb(correct_pad)
		PC_mask = get_subsequent_mask(correct_pad)
		PC_enc_out = self.PCEncoder(input_array=all_session_enc_out, correct_pad=correct_pad , pad_mask=PC_mask)
		# PC_enc_out = all_session_enc_out + correct_pad_embedding

		#-- Decoder
		target_problem_seq = input_array[:, session_size, 0]
		target_skill_seq = input_array[:, session_size, 1]
		target_qno = input_array[:, session_size, 3]
		target_kno = input_array[:, session_size, 4]

		# Get target mask
		trg_mask = get_subsequent_mask(target_problem_seq) & get_pad_mask(target_skill_seq, padding_node) & get_pad_mask(target_skill_seq, EOS_skill)

		dec_out = self.my_decoder(trg_problem_seq=target_problem_seq, trg_skill_seq=target_skill_seq,
									enc_output=PC_enc_out, trg_mask=trg_mask, pad_mask=trg_mask, target_qno=target_qno, target_kno=target_kno)
		dec_out = self.fc_layer1(dec_out)
		dec_out = nn.Sigmoid()(dec_out.squeeze(2))

		return dec_out
