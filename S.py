import torch as th


c = th.tensor([1, 2, 3])
m = th.tensor([3, 4, 7])


print(th.cat([c, m]))
