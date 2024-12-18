import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import lib.utils as utils
class Network:
  def __init__(self, layers=[], activation = utils.reLU):
    self.layers = layers
    self.activation = activation
  def display(self, axt=plt):
    x,y = (0,0)
    offset_x, offset_y = (20,5)
    for i in self.layers:
      y=0 
      ax = plt.gca()
      #ax.set_xlim([0, 1000])
      #ax.set_ylim([-1000, 1000])
      for ni in range(len(i.neurons)):
        plt.plot(x,y,'bo')
        plt.text(x,y+0.005, round(i.neurons[ni].activator, 2), color='b')
        for wi in range(len(i.neurons[ni].weights)):
          '''X = np.linspace(x, x+offset_x, 20)
          def f(a):
            return -(a-x)*wi/4
          Y = f(X)'''
          #print([x,y], [x+offset_x, y-offset_y*wi])
          axt.plot([x,x+offset_x], [y, y-offset_y*(wi-ni)], 'r')
          axt.text(x+offset_x/4,y-(offset_y*(wi-ni))/4 + 0.005, round(i.neurons[ni].weights[wi].weight,2), color='r')
        y-=offset_y
      x+=offset_x
    if axt==plt:
      plt.show()
    #plt.plot(X,Y, marker='o')

  def live(self, inputs: list, expected: list, delay=2.5, interval=1):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    def animate(i):
      i = i%len(expected)
      self.set_input(inputs[i])
      self.train(expected[i])
      self.run()
      ax1.clear()
      self.display(axt=ax1)
      #sleep(delay)
    ani = animation.FuncAnimation(fig, animate, interval=delay*interval)
    plt.show()

  def backpropagate(self, expected: list, interval: float=0.1):
    deriv = utils.derivatives[self.activation]
    

    for li in range(len(self.layers)-1,0,-1):
      #print("LAYER: ", li)
      n_expected = [n.activator for n in self.layers[li-1].neurons]
      for ni in range(len(self.layers[li].neurons)):
        #print("NEURON: ", ni)
        #print("LEN: ", len(expected))
        print(expected)
        
        index = 0
        for n in self.layers[li-1].neurons:
          #print("LAYER: ", li-1, "; NEURON: ", len(n_expected))
          delta = (expected[ni]-self.layers[li].neurons[ni].activator) * deriv(
                n.activator * n.weights[ni].weight + self.layers[li].neurons[ni].bias)
          n.weights[ni].weight += delta * interval # this is the gradient dummy
          
          #self.layers[li].neurons[ni].bias += interval * ader(n.activator * n.weights[ni].weight + self.layers[li].neurons[ni].bias) * (2 * (self.layers[li].neurons[ni].activator - expected[ni]))
          
          n_expected[index] += delta * n.activator 
      expected = n_expected
  def get_output(self) -> list:
    assert len(self.layers) > 0
    result = []
    for i in self.layers[-1].neurons:
      result.append(i.activator)
    return result

  def set_input(self, values: list) -> bool:
    assert len(self.layers) > 0
    assert len(values) == len(self.layers[0].neurons)

    for i in range(len(values)):
      self.layers[0].neurons[i].activator = values[i]
    return True

  def cost(self, expected: list) -> float:
    assert len(self.layers) > 0
    assert len(expected) == len(self.layers[-1].neurons)
    
    C = 0
    for i in range(len(self.layers[-1].neurons)):
      y = expected[i]
      a = self.layers[-1].neurons[i].activator
      C += (a-y)**2
    return C
  def train(self, expected: list, interval: float=0.1, debug: bool=False):
    self.run()
    cost = self.cost(expected)
    if debug:
      print("COST: ", cost)
    self.backpropagate(expected, interval)
  def run(self):
    assert len(self.layers) > 1
    for i in range(len(self.layers)-1):
      for ni in range(len(self.layers[i+1].neurons)):
        result = 0
        for n2 in self.layers[i].neurons:
          result += n2.activator * n2.weights[ni].weight + self.layers[i+1].neurons[ni].bias
        self.layers[i+1].neurons[ni].activator = self.activation(result)

class Layer:
  def __init__(self, neurons: list=[]):
    self.neurons = neurons

class Neuron:
  def __init__(self, weights: list=[], bias: float=0):
    self.activator = 0
    self.weights = weights
    self.bias = bias

class Weight:
  def __init__(self, weight):
    self.weight = weight

