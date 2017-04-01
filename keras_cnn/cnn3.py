import numpy as np
import pickle
import os
import random
import sys

from images import *

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import Callback

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

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
                        input_shape=(64,64, 3)))
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

        self.model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    def train(self, training_files, testing_files):

        labels = set()
        for _, label in training_files:
            labels.add(str(label))

        labels = list(labels)

        label_dict = {}
        i = 0
        for label in labels:
            label_dict[str(label)] = i
            i += 1

        files = []
        file_index = {}
        index = 0
        for filename, label in training_files:

            filename = filename[:filename.rfind("_")]
            files.append(filename)
            if file_index.get(filename) is None:
                file_index[filename] = [index, 1]
            else:
                file_index[filename][1] += 1

            index += 1

        files = list(set(files))

        for epoch in range(100):

            print "Epoch", epoch + 1
            print "0/", len(files), " - train_acc: 0"
            random.shuffle(files)

            training_correct = 0
            training_total = 0
            for img in files:

                training_total += 1

                index, size = file_index[img]
                x = []
                y = []
                old_label = None
                variances = []
                for i in range(index, index + size):

                    filename, label = training_files[i]
                    old_label = label

                    vector = read_color_image(filename)
                    x.append(vector)
                    y.append(label_dict[label])
                    variances.append(np.std(vector.ravel()))

                x = np.array(x)

                distr = [0,0,0,0,0,0]
                predictions = self.model.predict_classes(x)
                i = 0
                for prediction in predictions:
                    distr[prediction] += variances[i]
                    i += 1

                label_index = distr.index(max(distr))
                
                if old_label == labels[label_index]:
                    training_correct += 1
                else:
                    y = np.array(y)
                    y = np_utils.to_categorical(y, 6)
                    j = label_dict[old_label]
                    for i in range(y.shape[0]):
                        y.itemset((i,j), variances[i])

                    self.model.train_on_batch(x, y)

                sys.stdout.write(CURSOR_UP_ONE + CURSOR_UP_ONE + ERASE_LINE + ERASE_LINE)
                print training_total, "/", len(files), " - train_acc:", training_correct/float(training_total)

            self.testing_accuracy(testing_files, epoch)
            print ""

    def testing_accuracy(self, testing_files, epoch):
        labels = set()
        for _, label in testing_files:
            labels.add(str(label))

        labels = list(labels)

        label_dict = {}
        i = 0
        for label in labels:
            label_dict[str(label)] = i
            i += 1


        x = []
        y = []
        
        variances = []
        for filename, label in testing_files:
            vector = read_color_image(filename)
            x.append(vector)
            y.append(label_dict[label])
            variances.append(np.std(vector.ravel()))

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

        for filename, label in testing_files:
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

            distr[predictions[count]] += variances[count]

            count += 1

        label_index = distr.index(max(distr))
        prediction_string += str(old_label) + " " + str(labels[label_index]) + "\n"
        if old_label == labels[label_index]:
            correct += 1
        total += 1

        filename = "epoch_" + str(epoch) + ".predictions"
        with open(filename, 'wb') as writer:
            pickle.dump(prediction_string, writer)

        sys.stdout.write(CURSOR_UP_ONE + ERASE_LINE)
        print "test_acc:", correct/float(total)


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

