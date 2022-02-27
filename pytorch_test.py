from importlib.metadata import requires
import torch
import numpy as np
import torch.autograd
##
tf = torch.FloatTensor([[10,20.,20],[20,10,50]])
tf.cuda
ti = torch.Tensor([[10,20,20],[20,10,50]])
# print(tf == ti)
# print(tf + ti)
##
m = torch.rand(2,2)
mt = m.t()
# print(m)
# print(mt)
# pytorch Autograd

a = torch.tensor([3.0,2.0],requires_grad = True)
b = torch.tensor([4.0,7.0])
ab_sum = a + b
print('ab_sum is {}'.format(ab_sum))
ab_res = (ab_sum*8).sum()
print('ab_res is {}'.format(ab_res))
ab_res.backward()
print('ab_res after bw is {}'.format(ab_res))
print('a grad is {}'.format(a.grad))
print('b grad is {}'.format(b.grad))
