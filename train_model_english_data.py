from zmq import device
from englishDataset import englishDataset
from torch.utils.data import dataloader
from neuralnetwork1 import neuralnetwork1
from torch import nn
from torch.optim.sgd import SGD
from train_model import train
from test_model import test
import torch

train_dataset = englishDataset(
    root = 'data/english_dataset',
    train=True
)

test_dataset = englishDataset(
    root = 'data/english_dataset',
    train=False
)


# with dataloader 
batch_size = 64
train_dataloader = dataloader.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader = dataloader.DataLoader(dataset=test_dataset,batch_size=batch_size)
model_ = neuralnetwork1()
print(model_)
loss_fn = nn.CrossEntropyLoss()
#train
optimizer = SGD(model_.parameters(),lr = 1e-3)
epochs = 5
device = 'cuda'
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model_, loss_fn, optimizer,device)
    test(test_dataloader, model_, loss_fn,device)
print("Done!")