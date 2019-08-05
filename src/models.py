import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import settings
from settings import *
from utils import *

class BertBase(nn.Module):
  def __init__(self, bert, classes):
    super().__init__()
    self.bert = bert
    d_model = bert.embeddings.word_embeddings.weight.size(1)
    self.dense = nn.Linear(d_model, classes)

  def forward(self, inp, inp_segment, inp_mask):
    # encoder
    _, cls_output = self.bert(inp, inp_segment, inp_mask,
                              output_all_encoded_layers=False)

    cls_output = self.dense(cls_output)

    return cls_output

  def init_parameters(self):
    for p in self.dense.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

