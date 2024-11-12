import cv2
import numpy as np
import glob
images = glob.glob("training/*")
training = []
input_size = 0
for img_path in images:
  img = cv2.imread(img_path, 0) / 255.0
  print(img)
  input_size = len(img) * len(img[0])
  data = img_path.split(".")[0].split("/")[1].split("_")
  training.append((img, int(data[1]), data[0])) # image, expected_result, test_pair_id

from lib.neurons import *


'''
output_layer = Layer([
    Neuron(),
    Neuron(),
])
middle_man2 = Layer([
    Neuron(weights=[Weight(1) for i in output_layer.neurons], bias=0)
])
middle_man1 = Layer([
    Neuron(weights=[Weight(1) for i in middle_man2.neurons], bias=0)
])
input_layer = Layer([
   Neuron(weights=[Weight(1) for i in middle_man1.neurons], bias=0),
])


simple_network = Network([
    input_layer,
    middle_man1,
    middle_man2,
    output_layer,
])'''



'''
output_layer = Layer([
    Neuron(),
    Neuron(),
])

input_layer = Layer([
   Neuron(weights=[Weight(1) for i in output_layer.neurons], bias=0),
])

multi_output_network = Network([
    input_layer,
    #middle_man1,
    #middle_man2,
    output_layer,
])
'''

'''output_layer = Layer([
    Neuron(),
    Neuron(),
])
middle_man1 = Layer([
    Neuron(weights=[Weight(1) for i in output_layer.neurons], bias=0)
    for i in range(64)
])
input_layer = Layer([
   Neuron(weights=[Weight(1) for i in middle_man1.neurons], bias=0)
   for i in range(input_size)
])

network = Network([
  input_layer,
  middle_man1,
  output_layer,
])'''


output_layer = Layer([
    Neuron(),
    Neuron(),
])
middle_man1 = Layer([
    Neuron(weights=[Weight(1) for i in output_layer.neurons], bias=0),
    Neuron(weights=[Weight(1) for i in output_layer.neurons], bias=0)
])
input_layer = Layer([
   Neuron(weights=[Weight(1) for i in middle_man1.neurons], bias=0),
   Neuron(weights=[Weight(1) for i in middle_man1.neurons], bias=0),
])


display_test_network = Network([
    input_layer,
    middle_man1,
    output_layer,
])
network = display_test_network



network.display()

network.set_input([1])

network.train([0,1], debug=True)

network.display()

#network.run()
#network.display()
network.live([0,1])
