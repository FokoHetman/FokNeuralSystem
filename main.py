import pyLib.lib as neurons
import pyLib.utils as nutils
import cv2
import numpy as np
import glob
import copy

images = glob.glob("training_data/*")
training_data = []
for img_path in images:
  img = cv2.imread(img_path, 0) / 255.0
  print(img)
  data = img_path.split(".")[0].split("/")[1].split("_")
  expected = [0 for _ in range(2)]
  expected[int(data[1])] = 1
  training_data.append((img, expected, data[0])) # image, expected_result, test_pair_id
print("TRAINING_DAT: ", training_data)

input_layer = neurons.Layer([neurons.Neuron(0) for i in range(256)])

hidden_layer = neurons.Layer([neurons.Neuron(0) for i in range(16)])

output_layer = neurons.Layer([neurons.Neuron(0) for i in range(2)])

net = neurons.Network([
      input_layer,
      hidden_layer,
      copy.deepcopy(hidden_layer),
      output_layer,
    ], True, nutils.sigmoid)
net.correctify()
net.train(training_data)
net.run()
net.display()
