
from cProfile import label
from torch.utils.data import dataloader
from torchvision.transforms import ToTensor
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

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

classes = trainig_data.classes

# show training data
row ,col = 3,3
fig  = plt.figure()
rand_index = np.random.randint(1,len(trainig_data),size = row*col+1)

for i in range(1,row*col+1):
    j = rand_index[i]
    im,label_index = trainig_data[j]
    label_name = classes[label_index]
    fig.add_subplot(row,col,i)
    plt.imshow(im.squeeze() , cmap = 'gray')
    plt.title(label_name)
    plt.axis('off')
plt.show()

