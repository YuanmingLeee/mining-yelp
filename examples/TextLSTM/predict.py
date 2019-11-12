import torch
import torch.nn as nn
from torch import optim
from os.path import abspath, dirname, join
from model import TextLSTM
import dataloader
import time
from configs import *


# initialize dataloader
train_dataloader = dataloader.create_dataloader(
    x_dir = config['test_x'], 
    y_dir = config['test_y'], 
    batch_size = config['batch_size']
)
print("---- dataloaders are initialized")


# initialize model
model = TextLSTM(
    batch_size = config['batch_size'], 
    output_size = config['output_size'], 
    hidden_size = config['hidden_size'],
    vocab_size = config['vocab_size'],
    embedding_length = config['embedding_length'],
    weight_npy_dir = config['embedding_dir']
    )
weight_dir = join(common_dir, "weights", "epoch_{}.pth".format( config['load_epoch']))
model.load_state_dict(torch.load(weight_dir))
model.double()
model.cuda()
print("---- TextLSTM model is initialized with CUDA")

# initialize counts
total_count = 0
correct = 0

#predict for the testing data
for batch_x, batch_y in train_dataloader:
    batch_size = batch_x.shape[0]
    output = model(batch_x, batch_size)
    total_count += batch_size
    correct += (output == batch_y).float().sum()

print("Accuracy: {}".format((correct / total_count)))

