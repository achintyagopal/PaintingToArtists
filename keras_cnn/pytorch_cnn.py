from __future__ import print_function
import random
import sys
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils

import numpy as np
import pickle
import os

from images import *

torch.manual_seed(500)

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


training_files = get_files('../data/train/')
testing_files = get_files('../data/test/')
# sort(testing_files)
testing_files.sort()
# traingi
print(len(training_files))
print(len(testing_files))


labels = set()
for _, label in training_files:
    labels.add(str(label))

labels = list(labels)

label_dict = {}
i = 0
for label in labels:
    label_dict[str(label)] = i
    i += 1


class PaintingDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, files):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files = files
        self.data = []
        i = 0
        for filename, label in files:
            if i % 100 == 0:
                print(i)

            i += 1
            # if i % 1000 == 0:
                # break
            vector = read_color_image(filename) / 255.
            self.data.append( \
                (torch.from_numpy(np.transpose(vector, (2,1,0))).type(torch.FloatTensor), \
                torch.from_numpy(np.array([label_dict[label]])), \
                torch.from_numpy(np.array([np.std(vector.ravel())])).type(torch.FloatTensor)  ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CNN_Model(nn.Module):

    def __init__(self):
        super(CNN_Model, self).__init__()

        self.seq = torch.nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 2, 2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
        )
        self.seq2 = torch.nn.Sequential(
            nn.Linear(32*8*8, 128),
            nn.Dropout(p=0.5),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        y = self.seq(x)
        y = y.view(x.size()[0], -1)
        return self.seq2(y)

def evaluate(testing_files, epoch):
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
    variances = []
    
    i = 0
    for filename, label in testing_files:
        i += 1
        if i % 100 == 0:
            print(i)
        # if i % 1000 == 0:
            # break
        vector = read_color_image(filename)
        x.append(vector)
        variances.append(np.std(vector.ravel()))

    x = Variable(torch.from_numpy(np.transpose(np.array(x), (0,3,2,1))).type(torch.FloatTensor), volatile=True)
    _, predictions = torch.max(model(x), dim=1)
    predictions = predictions.data.numpy()
    # print(predictions.shape)
    distr = [0,0,0,0,0,0]
    old_filename = ""
    count = 0
    old_label = ""
    correct = 0
    total = 0
    prediction_string = ""

    for filename, label in testing_files:
        # img = read_color_image(filename)
        # if count % 1000 == 0:
            # break

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

        # print(distr, predictions[count], variances[count])
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

    print("Testing accuracy for epoch " + str(epoch) + ":", correct/float(total))



model = CNN_Model()
# evaluate(testing_files, 0)
# sys.exit(0)
batch_size = 128
testloader = torch.utils.data.DataLoader(
    PaintingDataset(testing_files), batch_size=batch_size, shuffle=False)
# evaluate2(testloader, testing_files, batch_size, 0)
# sys.exit(0)

dataloader = torch.utils.data.DataLoader(
    PaintingDataset(training_files), batch_size=batch_size, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))

def train(epoch):
    criterion = torch.nn.CrossEntropyLoss()
    for i, (data, y_class, variance) in enumerate(dataloader):
        # print(data.size())
        # print(y_class.size())
        data = Variable(data)
        y_class = Variable(y_class)
        variance = Variable(variance)
        loss = criterion(variance * model(data), y_class.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 1 == 0:
            print("loss at batch {0}: {1}".format(i, loss.data[0]))

def evaluate2(epoch):

    distr = [0,0,0,0,0,0]
    old_filename = ""
    count = 0
    old_label = ""
    correct = 0
    total = 0
    prediction_string = ""

    for i, (data, y_class, variances) in enumerate(testloader):
        data = Variable(data)
        variances = variances.numpy()
        # y_class = Variable(y_class)
        # variance = Variable(variance)
        _, predictions = torch.max(model(data), dim=1)
        # print(predictions)
        predictions = predictions.data.numpy()
        for j, prediction in enumerate(predictions):
            # print(i*batch_size + j)
            filename, label = testing_files[i*batch_size + j]
            # print(filename)
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

            distr[prediction] += variances[j]

    label_index = distr.index(max(distr))
    prediction_string += str(old_label) + " " + str(labels[label_index]) + "\n"
    if old_label == labels[label_index]:
        correct += 1
    total += 1

    filename = "epoch_" + str(epoch) + ".predictions"
    with open(filename, 'w') as writer:
        writer.write(prediction_string)
        # pickle.dump(prediction_string, writer)
    # print(total)
    print("Testing accuracy for epoch " + str(epoch) + ":", correct/float(total))
    


epochs = 1
for epoch in range(epochs):
    print("Epoch {0}".format(epoch))
    train(epoch)
    evaluate2(epoch)
    torch.save(model.state_dict(), 'models/model_epoch_%d.pth' % (epoch))
    # g.load_state_dict(torch.load('netG.path'))

# train(0)
# sys.exit(0)


# for i, (data, y_class, variance) in enumerate(dataloader):
#     # print(data.size())
#     # print(y_class.size())
#     data = Variable(data)
#     y_class = Variable(y_class)
#     variance = Variable(variance)
#     loss = criterion(variance * model(data), y_class.view(-1))
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if i % 10 == 0:
#         print("loss at batch {0}: {1}".format(i, loss.data[0]))

    # evaluate(testing_files, )
# y = []

# for filename, label in files:
#     vector = read_color_image(filename)
#     x.append(vector)
#     y.append(self.label_dict[label])

# x = np.array(x)
# y = np.array(y)
