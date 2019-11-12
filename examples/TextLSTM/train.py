import torch
import torch.nn as nn
from torch import optim
from os.path import abspath, dirname, join
from model import TextLSTM
import dataloader
import time
from configs import *

train_dataloader = dataloader.create_dataloader(
    x_dir = config['train_x'], 
    y_dir = config['train_y'], 
    batch_size = config['batch_size']
)
print("---- dataloaders are initialized")

model = TextLSTM(
    batch_size = config['batch_size'], 
    output_size = config['output_size'], 
    hidden_size = config['hidden_size'],
    vocab_size = config['vocab_size'],
    embedding_length = config['embedding_length'],
    weight_npy_dir = config['embedding_dir']
    )
model.double()
model.cuda()
print("---- TextLSTM model is initialized with CUDA")

# train
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
history_f = open(join(common_dir, "history.txt"), 'w')
for epoch in range(config["epoch"] ):
    avg_loss = 0
    count = 0
    start = time.time()
    for batch_x, batch_y in train_dataloader:
        batch_size = batch_x.shape[0]
        output = model(batch_x, batch_size) 
        loss = loss_fn(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        count += 1
    end = time.time()
    elapsed = end - start
    history_f.write("{}\n".format(avg_loss))
    print("---- average loss for epoch {} : {}, time elapsed : {}".format((epoch+1), avg_loss, elapsed))

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), join(common_dir, "weights", "epoch_{}.pth".format(epoch+1)))