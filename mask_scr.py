# import library
import pandas as pd
import numpy as np
import torch
import random
import math


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
	''' For masking out the subsequent info. '''
	len_s = seq.size(1)
	subsequent_mask = (1 - torch.triu(
		torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
	return subsequent_mask
