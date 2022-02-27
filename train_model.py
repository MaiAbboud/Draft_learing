from torch import nn
from neuralnetwork1 import neuralnetwork1
from torch.optim.sgd import SGD


def train(dataloader,model,loss_fn ,optimizer,device):
    model.train()
    for batch , (x,y) in enumerate(dataloader):
        x,y,model = x.to(device),y.to(device),model.to(device)

        #forward propagation
        pred = model(x)
        #error
        loss = loss_fn(pred,y)
        #backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# def test(dataloader,model,loss_fn):
#     model.eval()

