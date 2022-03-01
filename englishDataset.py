import os
# from pyrsistent import optional
from torchvision.io import read_image
from torchvision.datasets import VisionDataset
# from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from typing import Optional , Callable , Any
from torchvision.transforms import ToTensor,Grayscale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class englishDataset(VisionDataset):
    def __init__(
        self, 
        root: str, 
        transforms: Optional[Callable] = None, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
        ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        #read .csv data file 
        data_csv_file = pd.read_csv(root+'/english.csv')
        data_size = data_csv_file.shape[0]

        #shuffle dataset
        np.random.seed(10)
        np.random.shuffle(data_csv_file.values)

        #split dataset to train & test (0.8/0.2)
        dataset_train_size = int(data_size * 0.8)
        self.dataset_train = data_csv_file.iloc[0:dataset_train_size,:]
        self.dataset_test = data_csv_file.iloc[dataset_train_size:,:]
    
    def __getitem__(self, index: int) -> Any:
        # read image for specific data
        img_pth = os.path.join(self.root , self.dataset_train['image'][index]) 
        img , label = read_image(img_pth) , self.dataset_train['label'][index]
        return img,label

    def __len__(self) -> int:
        train_size = self.dataset_train.shape[0]
        test_size = self.dataset_test.shape[0]
        return train_size,test_size


dataset = englishDataset(
    root = 'data/english_dataset',
)
train_size = dataset.dataset_train.shape[0]
test_size = dataset.dataset_test.shape[0]


# show training data
row ,col = 3,3
fig  = plt.figure()
rand_index = np.random.randint(1,train_size,size = row*col+1)

for i in range(1,col*row+1):
    j = rand_index[i]
    img,label = dataset[j]
    fig.add_subplot(row,col,i)
    plt.imshow(img[1,:,:] , cmap = 'gray')
    plt.title(label)
    plt.axis('off')
plt.show()