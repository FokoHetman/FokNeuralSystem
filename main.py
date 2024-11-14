'''import cv2
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
'''
from lib.neurons import *
import numpy as np
import matplotlib.pyplot as plt


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

'''output_layer = Layer([
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
network = display_test_network'''

output_layer = Layer([
    Neuron(), # LEFT - RED
    Neuron(), # RIGHT - BLUE
])
hidden_layer = Layer([
    Neuron(weights=[Weight(1) for i in output_layer.neurons], bias=0),
    Neuron(weights=[Weight(1) for i in output_layer.neurons], bias=0)
])
input_layer = Layer([
   Neuron(weights=[Weight(1) for i in hidden_layer.neurons], bias=0), # X
   Neuron(weights=[Weight(1) for i in hidden_layer.neurons], bias=0), # Y
])

network = Network([
    input_layer,
    hidden_layer,
    output_layer,
])
network.run()
network.display()

cons = 40

rng = np.random.default_rng()
def gen_points(n, w=cons, h=cons):
  result = []
  for i in range(n):
    result.append([rng.random()*w,rng.random()*h])
  return result

points = gen_points(5)#[[.3, .7], [.7, .3]]#gen_points(1)

line = lambda x: x

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

'''
expects = []

for i in points:
  network.set_input([i[0]/200, i[1]/200])
  print([int(i[1] >= line(i[0])), int(i[1] < line(i[0]))])

  expects.append([int(i[1] >= line(i[0])), int(i[1] < line(i[0]))])

network.live(list(reversed(points)), list(reversed(expects)), delay=0.1, interval=0.0005)

'''
def animate(i):
  ax1.clear()
  for i in points:
    network.set_input([i[0]/cons, i[1]/cons])
    print([int(i[1] >= line(i[0])), int(i[1] < line(i[0]))])
    network.train([int(i[1] >= line(i[0])), int(i[1] < line(i[0]))], interval=0.1)
    network.run()
    
    #print(network.get_output())
    red = network.get_output()[0]
    blue = network.get_output()[1]

    print((red,0,blue))

    ax1.plot(i[0], i[1], 'o', color=(min(red, 1), 0, min(blue, 1)))
    X = np.linspace(0,cons,10)
    Y = line(X)
    ax1.plot(X,Y, "black")
  network.display()
  
ani = animation.FuncAnimation(fig, animate, interval=250)
plt.show()
#'''


#network.set_input([1,0])

#network.train([0,1], debug=True)

#network.display()

#network.run()
#network.display()
#network.live([[0,1]])






