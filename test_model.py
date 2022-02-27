import torch
def test_me(dataloader,model,loss_fn,device):
    loss_test , accurecy = 0 , 0
    data_size = len(dataloader.dataset)
    model.eval()
    with torch.no_grad():
        for x,y in dataloader:
            x,y,model = x.to(device),y.to(device),model.to(device)
            pred_test = model(x)
            loss_test = loss_fn(pred_test,y)
            # accurecy += (pred_test == y).sum
            # loss +=loss_fn(pred,y)
    # loss_fn = loss_fn/len(dataloader)
    # accurecy = accurecy/data_size
    print(loss_test)

def test(dataloader, model, loss_fn,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y, model = X.to(device), y.to(device) , model.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")