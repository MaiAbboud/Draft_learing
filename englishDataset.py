import os
from time import time
import torch
# from pyrsistent import optional
from torchvision.io import read_image
from torchvision.datasets import VisionDataset
# from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from typing import Optional , Callable , Any
from torchvision.transforms import ToTensor,Grayscale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import dataloader
import torchvision.transforms as T


class englishDataset(VisionDataset):
    def __init__(
        self, 
        root: str,
        train: bool = True, 
        transforms: Optional[Callable] = None, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
        ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.train = train
        #read .csv data file 
        data_csv_file = pd.read_csv(root+'/english.csv')
        data_size = data_csv_file.shape[0]

        #shuffle dataset
        np.random.seed(10)
        np.random.shuffle(data_csv_file.values)

        #split dataset to train & test (0.8/0.2)
        dataset_train_size = int(data_size * 0.8)
        if self.train:
            self.dataset_train = data_csv_file.iloc[0:dataset_train_size,:]
        else :
            self.dataset_test = data_csv_file.iloc[dataset_train_size:,:]
            self.dataset_test = self.dataset_test.reset_index()

        self.classes = ['0','1','2','3','4','5','6','7','8','9',
        'a','b','c','d','e','f',
        'g','h','i','j','k','l','m','n',
        'o','p','q','r','s','t','u','v','w','x','y','z',
        'A','B','C','D','E','F','G','H','I','J','K','L',
        'M','N','O','P','Q','R','S','T','U','V','W','X',
        'Y','Z'
        ]
        self.classes_dict = {self.classes[i]:i for i in range(len(self.classes))}

    
    def __getitem__(self, index: int) -> Any:
        # read image for specific data
        if self.train:
            img_pth = os.path.join(self.root , self.dataset_train['image'][index]) 
            label_name = self.dataset_train['label'][index]
        else:
            img_pth = os.path.join(self.root , self.dataset_test['image'][index]) 
            label_name = self.dataset_test['label'][index]

        img = read_image(img_pth)
        transform = T.Resize(size = (28,28))
        img = transform(img)  
        img = img[1,...]*1.0

        label_num = self.classes_dict[label_name]
        label =  label_num
        return img,label

    def __len__(self) -> int:
        if self.train:
            train_size = self.dataset_train.shape[0]
            return train_size
        else :
            test_size = self.dataset_test.shape[0]
            return test_size


# dataset = englishDataset(
#     root = 'data/english_dataset',
#     train=False
# )
# print(len(dataset.classes))
# test_size = len(dataset)
# data_10 = dataset[10]
# print(data_10)

# #load all dataset time without dataloader
# start_time= time.time()
# for sample in dataset.dataset_train:
#     print(sample)
# end_time = time.time()
# print(f"time without dataloader class is : {end_time - start_time}")

# # with dataloader 
# batch_size = 1
# start_time= time.time()
# train_dataloader = dataloader.DataLoader(dataset=dataset,batch_size=batch_size)
# end_time = time.time()
# print(f"time with dataloader class is : {end_time - start_time}")


# # show training data
# row ,col = 3,3
# fig  = plt.figure()
# rand_index = np.random.randint(1,train_size,size = row*col+1)

# for i in range(1,col*row+1):
#     j = rand_index[i]
#     img,label = dataset[j]
#     fig.add_subplot(row,col,i)
#     plt.imshow(img , cmap = 'gray')
#     plt.title(label)
#     plt.axis('off')
# plt.show()
# print(  )


# # show testing data
# row ,col = 3,3
# fig  = plt.figure()
# rand_index = np.random.randint(1,test_size,size = row*col+1)

# for i in range(1,col*row+1):
#     j = rand_index[i]
#     img,label = dataset[j]
#     fig.add_subplot(row,col,i)
#     plt.imshow(img, cmap = 'gray')
#     plt.title(dataset.classes[label])
#     plt.axis('off')
# plt.show()
# print()


