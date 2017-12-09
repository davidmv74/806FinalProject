import torch
import torch.nn as nn
from torch.autograd import Variable
import parser



input_size = 450000

rnn = nn.LSTM(input_size, hidden_size=240, num_layers=1)
input_lstm = Variable()
