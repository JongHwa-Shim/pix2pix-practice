import numpy as np
import torch
from make_dataset import *

x = [[1,2,3,4,5],
    [2,4,6,8,10],
    [1,4,7,10,13]]

asdf = transform_processing()
data = asdf.MinMaxScale(x)
a=1

data = torch.randn((3,3,3))
data = (data - data.min()) / (data.max() - data.min())
"""
x = [[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]], [[13,14,15],[16,17,18]], [[19,20,21], [22,23,24]]]

a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.array([7,8,9])
d = np.array([10,11,12])
e = np.array([13,14,15])
f = np.array([16,17,18])
g = np.array([19,20,21])
h = np.array([22,23,24])

i = np.array([a, b])
j = np.array([c, d])
k = np.array([e, f])
l = np.array([g, h])

m = np.array([i,j,k,l])

x = 1
"""
"""
a = 3
b = a
b = b + 3
a = [x for i in range(6) for x in [1]]

b = 1
"""
"""
idx = [element for element in range(10)]
asdf = torch.zeros((10,10))
for i in idx:
    asdf[i][i] = 1
fixed_condition = [asdf for _ in range(10)]
fixed_condition = torch.cat(fixed_condition,0)
"""
"""
filter = transform_processing()
idx = [[element] for element in range(10)]
idx = filter.to_LongTensor(idx)
"""
