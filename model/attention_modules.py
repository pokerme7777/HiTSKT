# import library
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
	''' Scaled Dot-Product Attention '''

	def __init__(self, temperature, attn_dropout=0.1):
		super().__init__()
		self.temperature = temperature
		self.dropout = nn.Dropout(attn_dropout)

	def forward(self, q, k, v, mask=None, encoder_type='Action'):
		if encoder_type !='Action':
			attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # bs*head*seq_l*seq_l
		else:
			attn = torch.matmul(q / self.temperature, k.transpose(3, 4))  # bs*head*seq_l*seq_l
		

		if mask is not None:
			attn = attn.masked_fill(mask == 0, -1e32)

		attn = self.dropout(F.softmax(attn, dim=-1))
		output = torch.matmul(attn, v)

		return output, attn

#-- DIY
def DIY_attention(q, k, v, d_k, mask, dropout=0.1):
	"""
	This is called by Multi-head atention object to find the values.
	"""
	scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # BS, head, seqlen, seqlen  #- dot product

	bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

	device = q.device

	#-- ABline
	x1 = (mask.squeeze(1)+0).expand(-1,seqlen,seqlen).to(device)
	x1 = torch.cumsum(x1, dim=-1)
	x2 = x1.transpose(1, 2).contiguous()
	AB_line = torch.abs(x1-x2)[:, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
	AB_line_affect = (1 / (AB_line +1 )).detach()

	scores = scores * AB_line_affect

	scores.masked_fill_(mask == 0, -1e32)
	scores = F.softmax(scores, dim=-1)  # BS,head,seqlen,seqlen
	scores = dropout(scores)
	output = torch.matmul(scores, v)
	return output, scores


class MultiHeadAttention(nn.Module):
	''' Multi-Head Attention module '''

	def __init__(self, n_head, d_model, d_k, d_v, dropout, encoder_type):
		super().__init__()

		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v
		self.encoder_type = encoder_type

		self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
		self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
		self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
		self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

		self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

		self.dropout = nn.Dropout(dropout)
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

	def forward(self, q, k, v, mask=None, Time_affect=False):
		d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
		if self.encoder_type != 'Action':
			sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
		else:
			sz_b, ses_L, len_q, len_k, len_v = q.size(0), q.size(1), q.size(2), k.size(2), v.size(2)

		residual = q

		# Pass through the pre-attention projection: b x lq x (n*dv)
		# Separate different heads: b x lq x n x dv
		if self.encoder_type != 'Action':
			q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
			k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
			v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
		else:
			q = self.w_qs(q).view(sz_b, ses_L, len_q, n_head, d_k)
			k = self.w_ks(k).view(sz_b, ses_L, len_k, n_head, d_k)
			v = self.w_vs(v).view(sz_b, ses_L, len_v, n_head, d_v)

		# Transpose for attention dot product: b x n x lq x dv
		if self.encoder_type != 'Action':
			q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
		else:
			q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)

		if mask is not None and self.encoder_type != 'Action':
			mask = mask.unsqueeze(1)  # For head axis broadcasting.
		elif mask is not None and self.encoder_type == 'Action':
			mask = mask.unsqueeze(2)  # For head axis broadcasting.

		if Time_affect==False:
			q, attn = self.attention(q, k, v, mask=mask, encoder_type=self.encoder_type)
		else:
			q, attn= DIY_attention(q, k, v, d_k=self.d_k, mask=mask, dropout=self.dropout)

		# Transpose to move the head dimension back: b x lq x n x dv
		# Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
		if self.encoder_type != 'Action':
			q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
		else:
			q = q.transpose(2, 3).contiguous().view(sz_b, ses_L, len_q, -1)
		q = self.dropout(self.fc(q))
		q += residual

		q = self.layer_norm(q)

		return q, attn


class PositionwiseFeedForward(nn.Module):
	''' A two-feed-forward-layer module '''

	def __init__(self, d_in, d_hid, dropout=0.1):
		super().__init__()
		self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
		self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
		self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		residual = x

		x = self.w_2(F.relu(self.w_1(x)))
		x = self.dropout(x)
		x += residual

		x = self.layer_norm(x)

		return x
