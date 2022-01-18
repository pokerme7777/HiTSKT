# import library
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math
from attention_modules import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
	''' Compose with two layers '''

	def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, encoder_type='Action'):
		super(EncoderLayer, self).__init__()
		self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, encoder_type=encoder_type)
		self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

	def forward(self, enc_input, slf_attn_mask=None, Time_affect=False):
		enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask, Time_affect=Time_affect)

		enc_output = self.pos_ffn(enc_output)
		return enc_output, enc_slf_attn



class PositionalEncoding(nn.Module):
	def __init__(self, d_hid, n_position=200, encoder_type='Action'):
		super(PositionalEncoding, self).__init__()

		# Not a parameter
		self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
		self.encoder_type = encoder_type

	def _get_sinusoid_encoding_table(self, n_position, d_hid):
		''' Sinusoid position encoding table '''

		# TODO: make it with torch instead of numpy

		def get_position_angle_vec(position):
			return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

		sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])

		sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
		sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

		return torch.FloatTensor(sinusoid_table).unsqueeze(0)

	def forward(self, x):
		if self.encoder_type == 'Action':
			return x + self.pos_table[:,None,:x.size(2)].clone().detach()
		else:
			return x + self.pos_table[:, :x.size(1)].clone().detach()



class DecoderLayer(nn.Module):
	''' Compose with three layers '''

	def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, encoder_type='Action'):
		super(DecoderLayer, self).__init__()
		self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout ,encoder_type=encoder_type)
		self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

	def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):

		dec_output, dec_enc_attn = self.enc_attn(dec_input, enc_output, enc_output, mask=dec_enc_attn_mask)

		dec_output = self.pos_ffn(dec_output)
		return dec_output, dec_enc_attn
