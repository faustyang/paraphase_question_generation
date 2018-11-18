import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

class Encoder(nn.Module):
    def __init__(self,input_dim,embedding_dim,hidden_dim,n_layers=2,dropout_rate=0.5,use_cuda=False):
        super(Encoder,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.use_cuda = use_cuda
        self.embedding = nn.Embedding(input_dim,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,num_layers=n_layers,dropout=dropout_rate)

    def forward(self,input,hidden):
        embeds = self.embedding(input).view(len(input),1,-1)
        output,hidden = self.lstm(embeds,hidden)
        return output,hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers,1,self.hidden_dim))
        if self.use_cuda: hidden = hidden.cuda()
        return hidden

encoder=Encoder(10,20,10)