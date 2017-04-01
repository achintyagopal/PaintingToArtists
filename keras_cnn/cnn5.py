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
from keras import backend as K
from keras.regularizers import l2

def my_objective_function(y_true, y_pred):
    # use categorial cross entropy
    # basically normal cross entropy (> 0 -> 1, < 0 -> 0), multiply by y_true
    return y_true * K.categorical_crossentropy(y_pred, (K.sign(y_true) + 1) / 2)

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
        variances = []
        
        for filename, label in self.testing_files:
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
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss=my_objective_function,
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
        variances = []
        count = 0
        for filename, label in files:
            print filename
            vector = read_color_image(filename)            
            x.append(vector)
            y.append(self.label_dict[label])
            variances.append(np.std(vector.ravel()))



        x = np.array(x)
        y = np.array(y)
        z = np_utils.to_categorical(y, 6)
        for i in range(len(variances)):
            for j in range(6):
                if j == y[i]:
                    z[i][j] = variances[i]
                else:
                    z[i][j] = - variances[i]

        self.model.fit(x, z, nb_epoch=50, batch_size = 32, 
            callbacks=[TestCallback(testing_files)])

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

