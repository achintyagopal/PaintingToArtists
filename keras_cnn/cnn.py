import numpy as np
import pickle
import os

from images import *

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import Callback

class TestCallback(Callback):
    def __init__(self, testing_files):
        self.testing_files = testing_files

    def on_epoch_end(self, epoch, logs={}):

        labels = set()
        for _, label in self.testing_files:
            labels.add(str(label))

        labels = list(labels)

        label_dict = {}
        i = 0
        for label in labels:
            label_dict[str(label)] = i
            i += 1


        x = []
        y = []
        
        for filename, label in self.testing_files:
            vector = read_color_image(filename)
            x.append(vector)
            y.append(label_dict[label])

        x = np.array(x)
        y = np.array(y)
        y = np_utils.to_categorical(y, 6)
        predictions = self.model.predict_classes(x)

        distr = [0,0,0,0,0,0]
        old_filename = ""
        count = 0
        old_label = ""
        correct = 0
        total = 0
        prediction_string = ""

        for filename, label in self.testing_files:
            img = read_color_image(filename)
            file_pre = filename[:filename.rfind("_")]
            if file_pre != old_filename:

                label_index = distr.index(max(distr))
                
                if old_label != "" and old_label == labels[label_index]:
                    correct += 1
                if old_label != "":
                    prediction_string += str(old_label) + " " + str(labels[label_index]) + "\n"
                    total += 1
                distr = [0,0,0,0,0,0]
                old_label = label
                old_filename = file_pre

            distr[predictions[count]] += 1

            count += 1

        label_index = distr.index(max(distr))
        prediction_string += str(old_label) + " " + str(labels[label_index]) + "\n"
        if old_label == labels[label_index]:
            correct += 1
        total += 1

        filename = "epoch_" + str(epoch) + ".predictions"
        with open(filename, 'wb') as writer:
            pickle.dump(prediction_string, writer)

        print "Testing accuracy for epoch " + str(epoch) + ":", correct/float(total)



class Keras_CNN():

    def __init__(self):
        self.model = Sequential()
        self.model.add(Convolution2D(32, 5, 5, border_mode='same',
                        input_shape=(64,64,3)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, 5, 5))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(6))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    def train(self, files, testing_files):

        self.labels = set()
        for _, label in files:
            self.labels.add(str(label))

        self.labels = list(self.labels)

        self.label_dict = {}
        i = 0
        for label in self.labels:
            self.label_dict[str(label)] = i
            i += 1

        x = []
        y = []

        for filename, label in files:
            vector = read_color_image(filename)
            x.append(vector)
            y.append(self.label_dict[label])

        x = np.array(x)
        y = np.array(y)
        y = np_utils.to_categorical(y, 6)

        self.model.fit(x, y, nb_epoch=50, batch_size = 32, 
            callbacks=[TestCallback(testing_files)])

    # def predict(self, files):
    #     total = 0
    #     correct = 0
    #     x = []
    #     y = []
        
    #     for filename, label in files:
    #         vector = read_color_image(filename)
    #         x.append(vector)
    #         y.append(self.label_dict[label])

    #     x = np.array(x)
    #     y = np.array(y)
    #     y = np_utils.to_categorical(y, 6)
    #     predictions = self.model.predict_classes(x)

    #     with open('predictions_epoch10.file', 'wb') as writer:
    #         pickle.dump(predictions, writer)


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


cnn = Keras_CNN()

training_files = get_files('data/train/')
testing_files = get_files('data/test/')
cnn.train(training_files, testing_files)

# cnn.predict(testing_files)

# with open('predictions_epoch10.file', 'rb') as reader:
#     predictions = pickle.load(reader)

# testing_files = get_files('data/test/')

# labels = set()
# for _, label in testing_files:
#     labels.add(str(label))

# labels = list(labels)

# label_dict = {}
# i = 0
# for label in labels:
#     label_dict[str(label)] = i
#     i += 1

# distr = [0,0,0,0,0,0]
# old_filename = ""
# count = 0
# old_label = ""
# correct = 0
# total = 0
# for filename, label in testing_files:
#     img = read_color_image(filename)
#     file_pre = filename[:filename.rfind("_")]
#     if file_pre != old_filename:

#         label_index = distr.index(max(distr))
        
#         if old_label != "" and old_label == labels[label_index]:
#             correct += 1
#         if old_label != "":
#             print old_label, labels[label_index]
#             total += 1
#         distr = [0,0,0,0,0,0]
#         old_label = label
#         old_filename = file_pre

#     distr[predictions[count]] += 1

#     count += 1

# label_index = distr.index(max(distr))
# print old_label, labels[label_index]
# if old_label == labels[label_index]:
#     correct += 1
# total += 1

# print correct/float(total)
