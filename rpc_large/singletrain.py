import time
import torch
import torch.nn as nn
import torch.optim as optim

from effnetv2_model_single import *
from effnetv2_model_single import effnetv2_l

num_batches = 3
batch_size = 120
image_w = 32
image_h = 32
num_classes = 1000

tik_tok = 0

device = torch.device("cuda:0")
model = effnetv2_l()
model = model.to(device)

loss_fn = nn.MSELoss().to(device)
opt = optim.SGD(model.parameters(), lr=0.05)

one_hot_indices = torch.LongTensor(batch_size) \
                    .random_(0, num_classes) \
                    .view(batch_size, 1)

def executionTime(num_batches):
    tik_tok = 0
    for i in range(num_batches):
        print(f"Processing batch {i}")
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
                    .scatter_(1, one_hot_indices, 1)
        tik = time.time()
        outputs = model(inputs.to(device))
        loss = loss_fn(outputs, labels.to(device))
        loss.backward()
        opt.step()
        tok = time.time()
        tik_tok += tok-tik
    print(f"execution time = {tik_tok}")
    tik_tok = 0
    
executionTime(num_batches)

