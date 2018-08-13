import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from constants import *
from util import *
import numpy as np

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_units=1024, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_units = num_units
        self.num_layers = num_layers

        self.encoder = nn.Embedding(VOCAB_SIZE, num_units)

        # RNN
        self.rnn = nn.LSTM(num_units, num_units, num_layers, batch_first=True, dropout=dropout)

        self.decoder = nn.Linear(self.num_units, VOCAB_SIZE)
        self.decoder.weight = self.encoder.weight

    def forward(self, x,  memory=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        x = self.encoder(x)
        x, memory = self.rnn(x, memory)
        x = self.decoder(x)
        return x, memory

    def generate(self, x, memory, temperature=1):
        """ Returns the probability of outputs """
        x, memory = self.forward(x, memory)
        seq_len = x.size(1)
        x = F.softmax(x / temperature, dim=2)
        return x, memory
