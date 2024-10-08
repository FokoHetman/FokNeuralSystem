import math
import pyLib.utils as utils


class Weight:
  def __init__(self, weight):
    self.weight = weight


class Neuron:
  def __init__(self, activator, bias=0):
    self.activator = activator
    self.bias = bias
    self.weights = []
  def match_weights(self, next_layer):
    for _ in next_layer.neurons:
      self.weights.append(Weight(1))
  def match_weights_vec(self, vector):
    for i in vector:
      self.weights.append(Weight(i))
  def match_weights_preset(self, preset):
    self.weights = preset

class Layer:
  def __init__(self, neurons, index=0):
    self.neurons = neurons
    self.index = index
  def backpropagate(self, layers, cost):
    layers[self.index+1]


class Network:
  def __init__(self, layers, logging=False, optimisation=utils.reLU):
    self.layers = layers
    self.logging = logging
    self.optimisation = optimisation
  def correctify(self):
    for i in range(len(self.layers)):
      self.layers[i].index = i 
    for layeri in range(len(self.layers)-1):
      for neuron in self.layers[layeri].neurons:
        if neuron.weights == []:
          neuron.match_weights(self.layers[layeri+1])

    return True
  def display(self, mode="list"):
    result = ""
    for i in self.layers:
      result += "LAYER " + str(i.index) + ":\n"
      for n in i.neurons:
        result += "*" + str(n.activator)
        result += "\t" + str([i.weight for i in n.weights])
        result += "\t" + str(n.bias)
        result += "\n"
    print(result)
  def cost(self):
    pass
  def backpropagate(self, output_expects):
    pass
  def train(self, training_data):
    #for data in training_data[0]:
      data = training_data[0][0]
      print("DAT:", data)
      for i in range(len(self.layers[0].neurons)):
        w,h = (len(data), len(data[0]))
        self.layers[0].neurons[i].activator = data[int(i/w)][i%w]
      self.run()
      self.backpropagate(output_expectations)
  def run(self):
    for layeri in range(1, len(self.layers)):
      for neuroni in range(len(self.layers[layeri].neurons)):
        self.layers[layeri].neurons[neuroni].activator = self.optimisation(sum([neuron.weights[neuroni].weight * neuron.activator for neuron in self.layers[layeri-1].neurons]) + self.layers[layeri].neurons[neuroni].bias)

        #for weighti in range(len(neuron.weights)):
        #  rsum += neuron.activator * neuron.weights[weighti].weight + self.layers[layeri+1].neurons[weighti].bias
          #print("help: " , self.layers[layeri+1].neurons[weighti].activator)
        #self.layers[layeri+1].neurons[weighti].activator = reLU(rsum - self.layers[layeri)


      
