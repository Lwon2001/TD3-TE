import torch
import numpy as np

data = np.array([2, 3, 4, 4])

data = torch.from_numpy(data)
a = torch.max(data / 2)
print(a)

