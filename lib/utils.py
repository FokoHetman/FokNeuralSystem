import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))
reLU = lambda x: int(x>0)*x



derivatives = {
  sigmoid: lambda x: x * (1 - x),
  reLU: lambda x: int(x>=0),
}
