from project_types import *
import numpy as np
from scipy.ndimage.filters import convolve

class CNN(Predictor):
    
    def __init__(self, input_size, iterations = 1, learning_rate = 1e-4):
        self.contains_output = False
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.output_size = []
        self.output_size.append(input_size)
        self.last_output_size = input_size
        self.layers = ['input']

    def add_convolution_layer(self, nodes, size):
        if self.contains_output:
            raise Exception ("Cannot add layers after output layers")

        x,y,c = self.last_output_size
        self.last_output_size = (x, y, nodes)
        self.output_size.append((x,y,nodes))

        kx, ky = size
        layer = ['conv']
        for _ in range(nodes):
            w = np.random.randn(kx*ky*c) / sqrt(kx*ky*c)
            w = np.resize((kx, ky, c))
            layer.append(w)

        self.layers.append(layer)

    def add_relu_layer(self):
        if self.contains_output:
            raise Exception ("Cannot add layers after output layers")

        self.layers.append(['relu'])

    def add_pool_layer(self, shape):
        if self.contains_output:
            raise Exception ("Cannot add layers after output layers")

        sx, sy, sz = shape
        x,y,c = self.last_output_size
        self.last_output_size = (x/sx, y/sy, c/sz)
        self.output_size.append((x/sx, y/sy, c/sz))

        self.layers.append(['pool', shape])

    def add_fc_output_layer(self, nodes):
        if self.contains_output:
            raise Exception ("Cannot have multiple output layers")

        self.contains_output = True
        x,y,c = self.last_output_size
        self.last_output_size = (nodes,1,1)
        self.output_size.append((nodes, 1, 1))

        layer = ['fc']
        for _ in range(nodes):
            w = np.zeros((x,y,c))
            layer.append(w)
        self.layers.append(w)

    def train(self, feature_converter):

        if not self.contains_output:
            raise Exception("Must add output layer before training")

        self.labels = set()
        for i in range(feature_converter.trainingInstancesSize()):
            label = feature_converter.getTrainingLabel(i)
            self.labels.add(str(label))
        self.labels = list(self.labels)

        self.label_dict = {}
        i = 0
        for label in self.labels:
            self.label_dict[str(label)] = i
            i += 1

        for iteration in range(self.iterations):
            instance_order = []
            for i in range(feature_converter.trainingInstancesSize()):
                instance_order.append(i)

            shuffle(instance_order)

            for i in instance_order:
                instance = feature_converter.getTrainingInstance(i)
                instance_label = feature_converter.getTrainingLabel(i)

                inputs, label = self.__feed_forward(instance.get_vector())
                if label != self.label_dict[str(instance_label)]:
                    self.__back_propagate(instance.get_vector(), label, -1/learning_rate, inputs)
                    self.__back_propagate(instance.get_vector(), self.label_dict[str(instance_label)], 1/learning_rate, inputs)


    def __feed_forward(self, vector):
        outputs = []
        outputs.append(['input', vector])
        last_output = vector

        for layer in self.layers:
            layer_type = layer[0]
            if layer_type == "conv":
                new_output = []
                for kernel in layer[1:]:
                    # find best mode
                    new_output.append(convolve(last_output, kernel, mode='constant'))
                new_output = np.array(new_output)
                outputs.append([layer_type, last_output])
                last_output = new_output

            elif layer_type == "relu":
                a,b,c = last_output.shape
                new_output = np.zeros((a,b,c))
                for i in range(a):
                    for j in range(b):
                        for k in range(c):
                            val = last_output.item((i,j,k))
                            if val < 0:
                                new_output.itemset((i,j,k), 0)
                            else:
                                new_output.itemset((i,j,k), val)

                outputs.append([layer_type,last_output])
                last_output = new_output

            elif layer_type == "pool":
                sx,sy,sz = layer[1]
                ox,oy,oz = last_output.shape
                new_output = np.zeros((ox/sx, oy/sy, oz/sz))
                locations = np.zeros((ox/sx, oy/sy, oz/sz))
                for i in range(0, ox, sx):
                    for j in range(0, oy, sy):
                        for k in range(0, oz, sz):
                            max_val = None
                            max_loc = None
                            for a in range(sx):
                                for b in range(sy):
                                    for c in range(sz):
                                        val = last_output.item((i+a, j+b, k+c))
                                        if max_val is None or val > max_val:
                                            max_val = val
                                            max_loc = (a,b,c)  
                            new_output.itemset((i,j,k), max_val)
                            locations.itemset((i,j,k), max_loc)

                outputs.append([layer_type, locations])
                last_output = new_output

            elif layer_type == "fc":
                new_output = layer[1].ravel().dot(last_output.ravel())
                outputs.append([layer_type,last_output])
                last_output = new_output
            elif layer_type == "input":
                continue
            else:
                raise Exception('Unrecognized layer')
        
        # input of each layer, label
        max_label = None
        max_val = None
        i = 0
        for val in last_output.ravel():
            if max_val is None or val > max_val:
                max_val = val
                max_label = i
            i += 1

        return outputs, max_label

    def __back_propagate(self, vector, node_index, constant, inputs):
        last_output = [0] * self.output_size[-1][0]
        last_output[node_index] = constant

        for i in range(len(self.inputs), 0, -1):
            input_i = self.inputs[i - 1]
            input_type = input_i[0]

            if input_type == "conv":
                pass
            elif input_type == "pool":
                locations = input_i[1]

                a,b,c = locations.shape
                new_output = np.zeros((a,b,c))
                for i in range(a):
                    for j in range(b):
                        for k in range(c):
                            location = locations.item((i,j,k))
                            new_output.itemset(location, last_output.item(location))
                last_output = new_output

            elif input_type == "relu":
                input_vector = input_i[1]

                a,b,c = input_vector.shape
                for i in range(a):
                    for j in range(b):
                        for k in range(c):
                            if input_vector.item((i,j,k)) < 0:
                                last_output.itemset((i,j,k), 0)

            elif input_type == "fc":
                input_vector = input_i[1]
                w = self.layers[-1][node_index]
                last_output = w
                w = w + constant * input_vector
                self.layers[-1][node_index] = w
                
            elif input_type == "input":
                continue
            else:
                raise Exception("Unrecognized layer")


    def predict(self, feature_converter):
        if not self.contains_output:
            raise Exception("Must add output layer before testing")
