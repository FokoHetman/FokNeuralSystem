import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def reLU(x):
  return  x * int(x>0)
