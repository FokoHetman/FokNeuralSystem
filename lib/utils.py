import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))
reLU = lambda x: int(x>0)*x



derivatives = {
  sigmoid: lambda x: sigmoid(x)(1 - sigmoid(x)),
  reLU: lambda x: int(x>=0),
}
