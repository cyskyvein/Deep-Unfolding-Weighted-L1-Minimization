import random
import torch

x = torch.rand(3, 3)

y = x.clamp(min=-0.1, max=0.1)
print(x)
print(y)

