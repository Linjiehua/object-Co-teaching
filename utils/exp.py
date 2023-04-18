import numpy as np
import torch
a=torch.from_numpy(np.array([1,1,1,1,1,2,2,2,2]))

print(torch.cat([a[0:5],3*a[5:]],0))