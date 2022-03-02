from torch import nn
from neuralnetwork1 import neuralnetwork1
from torch.optim.sgd import SGD


def train(dataloader,model,loss_fn ,optimizer,device):
    model.train()
    size = len(dataloader.dataset)
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
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# def test(dataloader,model,loss_fn):
#     model.eval()

