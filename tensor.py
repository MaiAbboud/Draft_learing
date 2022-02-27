import torch
import numpy as np
import time
data = [[1,2],[3,4]]
# print(type(data))
x_data = torch.tensor(data=data,device='cuda',requires_grad=False)
x_data.to('cuda')
np_data = np.array(data)
x_data_array = torch.from_numpy(np_data)
x_data_array = x_data_array.type(torch.float64)
x_data_array.requires_grad = True
# print(x_data)

rand = torch.rand((5,5))
print(f"Tensor : {rand}")
print(f"First row :{rand[0]}")
print(f"First column :{rand[:,0]}")
print(f"Last column :{rand[:,-1]}")
# vectorization
rand[:,0] = 0
print(f"Tensor : {rand}")

value = torch.tensor(10.5)
print(f"type of value is {type(value)}")
value = value.item()
print(f"type of value is {type(value)}")

# print(f"rand item is :{rand.item()}")

# cpu vs gpu speed 
tensor_cpu = torch.rand(10,10)
tensor_gpu = torch.rand(10,10)
tensor_gpu.to('cuda')
# cpu time for multiplication
start_time = time.time()
tensor_mult = tensor_cpu @ tensor_cpu
end_time = time.time()
print(f"cpu time fo tensor mult :{end_time - start_time} ")
#gpu time
start_time = time.time()
tensor_mult = tensor_gpu @ tensor_gpu
end_time = time.time()
print(f"gpu time fo tensor mult :{end_time - start_time} ")