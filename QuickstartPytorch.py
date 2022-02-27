from pyexpat import model
import torch
from zmq import device
device = 'cuda'
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

from statistics import mode
from torch import nn
from torch.optim.sgd import SGD
from neuralnetwork1 import neuralnetwork1
from train_model import train
from test_model import test
from mimetypes import init
from torch.utils.data import dataloader
from torchvision import datasets
from torchvision.transforms import ToTensor

#download training data
trainig_data = datasets.FashionMNIST(
    root = 'data',
    train = 'True',
    transform = ToTensor(),
    download=True
)

#downlaod test data
testing_data = datasets.FashionMNIST(
    root = 'data',
    train = 'False',
    transform = ToTensor(),
    download=True
)

batch_size = 8
train_dataloader = dataloader.DataLoader(
    dataset=trainig_data,
    batch_size=batch_size
    )

test_dataloader = dataloader.DataLoader(
    dataset=testing_data,
    batch_size=batch_size
    )

for x,y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break



model_ = neuralnetwork1()
print(model_)
loss_fn = nn.CrossEntropyLoss()
#train
optimizer = SGD(model_.parameters(),lr = 1e-3)
# epoch = 5
# for i in range(epoch):
#     train(train_dataloader,model_,loss_fn,optimizer,device)

#save model
# torch.save(model_.state_dict(),"model.pth")

#load model
# model_.load_state_dict(torch.load("model.pth"))
# test(test_dataloader,model_,loss_fn,device)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model_, loss_fn, optimizer,device)
    test(test_dataloader, model_, loss_fn,device)
print("Done!")
        