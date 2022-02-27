
import torch
T_ = torch.tensor(0.5)
print(T_)
print(T_ + 1)
print(T_ + T_)
for i in range(10):
    T_ += T_
print(T_)