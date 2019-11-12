import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class TextLSTM(nn.Module):

    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weight_npy_dir):
        super(TextLSTM, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(np.load(weight_npy_dir)).double())
        # self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        # self.word_embeddings.weight = nn.Parameter(weights)
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sfm = nn.Softmax()

    def forward(self, x, batch_size):
        vec = self.word_embeddings(x)
        vec = vec.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size)).double().cuda()
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size)).double().cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(vec, (h_0, c_0))
        res = self.fc(final_hidden_state[-1])
        res = self.sfm(res)
        return res
