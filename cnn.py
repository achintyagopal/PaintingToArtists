from project_types import *
import numpy as np
from scipy.ndimage import convolve
from random import shuffle
import cv2

class CNN(Predictor):
    
    def __init__(self, input_size, iterations = 5, learning_rate = 1e6):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.output_size = []
        self.output_size.append(input_size)
        self.last_output_size = input_size
        self.layers = [['input']]

    def add_convolution_layer(self, nodes, size):

        y,x,c = self.last_output_size
        self.last_output_size = (y, x, nodes)
        self.output_size.append((y,x,nodes))

        ky, kx = size
        layer = ['conv']
        for _ in range(nodes):
            w = np.ones(kx*ky*c) / np.sqrt(kx*ky*c)
            w = np.resize(w, (ky, kx, c))
            layer.append(w)

        self.layers.append(layer)

    def add_relu_layer(self):

        self.output_size.append(self.last_output_size)
        self.layers.append(['relu'])

    def add_pool_layer(self, shape):

        sx, sy, sz = shape
        x,y,c = self.last_output_size

        if x % sx != 0 or y % sy != 0 or c % sz != 0:
            raise Exception("Integer divison")

        self.last_output_size = (x/sx, y/sy, c/sz)
        self.output_size.append((x/sx, y/sy, c/sz))

        self.layers.append(['pool', shape])

    def add_fc_output_layer(self, nodes):

        x,y,c = self.last_output_size
        self.last_output_size = (nodes,1,1)
        self.output_size.append((nodes, 1, 1))

        layer = ['fc']
        for _ in range(nodes):
            w = np.zeros((x,y,c))
            layer.append(w)
        self.layers.append(layer)

    def train(self, feature_converter):

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
                    initial_deriv = np.zeros(len(self.labels))
                    initial_deriv[label] -= 1
                    initial_deriv[self.label_dict[str(instance_label)]] += 1
                    self.__back_propagate(initial_deriv, 1/self.learning_rate, inputs)



            total = 0 
            correct = 0
            for i in instance_order:
                instance = feature_converter.getTrainingInstance(i)
                instance_label = feature_converter.getTrainingLabel(i)

                inputs, label = self.__feed_forward(instance.get_vector())
                if label == self.label_dict[str(instance_label)]:
                    correct += 1
                total += 1

            print "Iteration", iteration + 1, ":", correct/float(total)


    def __feed_forward(self, vector):
        outputs = []
        outputs.append(['input'])
        last_output = vector

        kernel = self.layers[1][1]

        for layer in self.layers:
            layer_type = layer[0]
            if layer_type == "conv":
                new_output = []
                for kernel in layer[1:]:
                    _,_,depth = kernel.shape[:]
                    convolved = convolve(last_output, kernel, mode='constant')[:,:,depth/2]
                    new_output.append(convolve(last_output, kernel, mode='constant')[:,:,depth/2])
                
                new_output = np.array(new_output)
                new_output = new_output.swapaxes(0,1)
                new_output = new_output.swapaxes(1,2)

                outputs.append([layer_type, last_output])
                last_output = new_output

            elif layer_type == "relu":
                a,b,c = last_output.shape
                new_output = np.zeros((a,b,c))
                for i in range(a):
                    for j in range(b):
                        for k in range(c):
                            val = last_output.item((i,j,k))
                            if val >= 0:
                                new_output.itemset((i,j,k), val)

                outputs.append([layer_type,last_output])
                last_output = new_output

            elif layer_type == "pool":
                sx,sy,sz = layer[1]
                ox,oy,oz = last_output.shape
                new_output = np.zeros((ox/sx, oy/sy, oz/sz))
                locations = []
                for i in range(0, ox, sx):
                    locations_i = []
                    for j in range(0, oy, sy):
                        locations_j = []
                        for k in range(0, oz, sz):
                            locations_j.append(0)
                        locations_i.append(locations_j)
                    locations.append(locations_i)

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
                                            max_loc = (i+a, j+b, k+c)
                            new_output.itemset((i/sx,j/sy,k/sz), max_val)
                            locations[i/sx][j/sy][k/sz] =  max_loc

                outputs.append([layer_type, locations])
                last_output = new_output

            elif layer_type == "fc":
                new_output = []
                for w in layer[1:]:
                    new_output.append(w.ravel().dot(last_output.ravel()))

                new_output = np.array(new_output)
                outputs.append([layer_type,last_output])
                last_output = new_output
            elif layer_type == "input":
                continue
            else:
                # print layer_type
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

    def __back_propagate(self, initial_deriv, constant, inputs):

        last_output = initial_deriv

        for index in range(len(inputs), 0, -1):
            input_i = inputs[index - 1]
            input_type = input_i[0]
            if input_type == "conv":
                # dL/dX
                input_vector = input_i[1]
                _,_,depth = last_output.shape
                new_output = np.zeros((self.output_size[index - 2]))
                for i in range(depth):
                    img = last_output[:,:,i]
                    kernel = self.layers[index - 1][i + 1]
                    new_output = new_output + self.inverse_convolve(img, kernel)

                # dL/dw
                kernels = self.layers[index - 1][1:]

                x = inputs[index-1][1]
                for l in range(depth):
                    kernel = kernels[l]
                    kx, ky, kz = kernel.shape
                    y = last_output[:,:,l]
                    rows, cols = y.shape
                    for k in range(kz):
                        for i in range(-(kx/2), kx/2 + 1):
                            for j in range(-(ky/2), ky/2 + 1):
                                value = np.dot(y[max(0,i):min(rows, rows + i), max(0,j):min(cols, cols + j)].ravel(), x[max(0,-i):min(rows, rows - i), max(0, -j):min(cols, cols - j), kz - k - 1].ravel())
                                self.layers[index - 1][l + 1][i + kx/2][j + ky/2][k] += constant * value

                last_output = new_output

            elif input_type == "pool":
                locations = input_i[1]
                a,b,c = len(locations), len(locations[0]), len(locations[0][0])

                new_output = np.zeros(self.output_size[index - 2])
                sx, sy, sz = self.layers[index - 1][1]
                for i in range(a):
                    for j in range(b):
                        for k in range(c):
                            lx, ly, lz = locations[i][j][k]
                            new_output.itemset((lx, ly, lz), last_output.item((lx/sx, ly/sy, lz/sz)))
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

                new_output = np.zeros(last_output.shape)
                for i in range(last_output.shape[0]):
                    new_output = new_output + last_output.item(i) * self.layers[index - 1][i + 1]
                    self.layers[index - 1][i + 1] = self.layers[index - 1][i + 1] + last_output.item(i)*constant*input_i[1]
                last_output = new_output

            elif input_type == "input":
                continue
            else:
                raise Exception("Unrecognized layer")

    def inverse_convolve(self, img, kernel):
        # x,y in order
        # z in reverse order
        _,_,l = kernel.shape
        new_img = []
        for z in range(l):
            kernel_z = cv2.flip(kernel[:,:,l - z - 1], -1)
            new_img.append(convolve(img, kernel_z, mode="constant"))
        new_img = np.array(new_img)
        new_img = np.swapaxes(new_img, 0,1)
        new_img = np.swapaxes(new_img, 1,2)

        return new_img



    def predict(self, feature_converter):
        if not self.contains_output:
            raise Exception("Must add output layer before testing")
