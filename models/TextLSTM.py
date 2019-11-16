import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data as tdata

from configs import BASE_DIR
from helper import parse_config


class TextLSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        # load config
        self.cfg = parse_config(config)
        self.output_size = self.cfg.OUTPUT_SIZE
        self.hidden_size = self.cfg.HIDDEN_SIZE
        self.vocab_size = self.cfg.VOCAB_SIZE

        self.embedding_length = self.cfg.EMBEDDING_LENGTH
        self.word_embeddings = nn.Embedding.from_pretrained(
            torch.from_numpy(np.load(BASE_DIR / self.cfg.EMBEDDING_DIR)).float()
        )
        self.lstm = nn.LSTM(self.cfg.EMBEDDING_LENGTH, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.sfm = nn.Softmax()

    def forward(self, x):
        bs = x.shape[0]
        vec = self.word_embeddings(x)
        vec = vec.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(1, bs, self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(1, bs, self.hidden_size)).cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(vec, (h_0, c_0))
        res = self.fc(final_hidden_state[-1])
        res = self.sfm(res)
        return res

    def batch_predict(self, data_loader: tdata.DataLoader):
        pass
