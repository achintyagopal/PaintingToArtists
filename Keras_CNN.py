import cv2
import numpy as np
from OriginalImg import OriginalImg
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import activity_l2
import os
from project_types import *
import pickle
# from keras.layers import Dense, Activation

class Keras_CNN():

    def __init__(self):
        self.model = Sequential()
        self.model.add(Convolution2D(32, 5, 5, border_mode='same',
                        input_shape=(128,128, 3)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, 5, 5))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(6))
        self.model.add(Activation('sigmoid'))
        # let's train the model using SGD + momentum (how original).
        sgd = SGD(lr=1e-6, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(loss='hinge',
              optimizer=sgd,
              metrics=['accuracy'])

        # Y_train = np_utils.to_categorical([1,2], 3)
        # Y_train = np.array([1,2])
        # X_train =  np.array([[[[1]], [[2]], [[3]]], [[[6]], [[-1]], [[4]]]])
        # print X_train.shape
        # self.model.fit(x = X_train, y = Y_train)

    def add_convolutional(filters, con_size=(3,3), input_size=None):
        # self.model.add()
        # apply a 3x3 convolution with 64 output filters on a 256x256 image:
        # model = Sequential()
        # size should be (3, row, col) for RGB img
        if input_size is not None:
            self.model.add(keras.layers.convolutional.Convolution2D(filters, con_size[0], con_size[1], border_mode='same', input_shape=(input_size)))
        else:
            # add a 3x3 convolution on top, with 32 output filters:
            model.add(keras.layers.convolutional.Convolution2D(filters, con_size[0], con_size[1], border_mode='same'))

    def add_pool(self,size=(2,2)):
        self.model.add(keras.layers.pooling.MaxPooling2D(pool_size=size, strides=None, border_mode='valid', dim_ordering='default'))

    def add_relu(self):
        self.model.add(Activation('relu'))

    def compile_model(self):
        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    
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

        x = []
        y = []
        print feature_converter.trainingInstancesSize()
        for i in range(feature_converter.trainingInstancesSize()):
            inst = feature_converter.getTrainingInstance(i)
            vector = inst.get_vector()
            # vector = np.swapaxes(vector, 2, 1)
            # vector = np.swapaxes(vector, 0, 1)
            x.append(vector)
            y.append(self.label_dict[inst.get_label()])
            # break

        x = np.array(x)
        print x.shape
        y = np.array(y)
        y = np_utils.to_categorical(y, 6)
        self.model.fit(x, y, nb_epoch=10, batch_size = 32)

    def predict(self, feature_converter):
        labels = []
        total = 0
        correct = 0
        x = []
        y = []
        print self.labels[4]
        for i in range(feature_converter.testingInstancesSize()):
            instance = feature_converter.getTestingInstance(i)
            vector = instance.get_vector()
            # vector = np.swapaxes(vector, 2, 1)
            # vector = np.swapaxes(vector, 0, 1)
            x.append(vector)
            y.append(self.label_dict[instance.get_label()])
            # x = instance.get_vector()
            # lab = self.model.predict(x)
            # print self.labels[lab], str(instance.get_label())
            # # labels.append(self.labels[max_index])

            # if self.labels[lab] == str(instance.get_label()):
            #   correct += 1
            # total += 1
        x = np.array(x)
        y = np.array(y)
        y = np_utils.to_categorical(y, 6)
        score = self.model.evaluate(x, y)
        print self.model.predict_classes(x)
        print self.label_dict
        print score
        # for row in labs:
        #     i = 0
        #     for value in row:
        #         if value == 1:
        #             print i
        #         i += 1

        # # print labs
        # print "Testing Accuracy: ", (correct / float(total) * 100), "%"
        return labels

def get_files(folder_name):

    foldersTmp = os.listdir(folder_name)
    folders = []
    for folder in foldersTmp:
        if folder[0] == '.':
            continue
        folders.append(folder)

    imgs = []
    for folder in folders:
        path = folder_name + folder + '/'
        if not os.path.isdir(path):
            continue
            
        files = os.listdir(path)
        for file_str in files:
            complete_file_str = str((os.path.join(path, file_str)))
            if os.path.isfile(complete_file_str) and (complete_file_str.endswith('.jpg') or complete_file_str.endswith('.JPG')):
                imgs.append((os.path.join(path, file_str), folder))

    return imgs

feature_converter = OriginalImg()
training_files = get_files('data/train/')
testing_files = get_files('data/test/')
feature_converter.createTrainingInstances(training_files)
feature_converter.createTestingInstances(testing_files)
# with open('')
for _ in range(5):
    cnn = Keras_CNN()
    cnn.train(feature_converter)
    cnn.predict(feature_converter)
