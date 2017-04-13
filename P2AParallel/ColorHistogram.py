from images import *
from project_types import FeatureConverter, Instance
import cv2
import numpy as np
import time
from multiprocessing import Pool
from functools import partial

class ColorHistogram(FeatureConverter):

    def __init__(self, bits = 12):
        self.bits = bits

    def createTrainingInstances(self, images):
        start = time.time()
        lst2 = []
        for image in images:
            # print image[0]
            instance = self.__createInstances(image)
            lst2.append(instance)
        print "TRAIN: ", time.time() - start
        
        start = time.time()
        lst1 = par(images, self.bits)
        print "TRAIN: ", time.time() - start
        
        for i in range(len(lst1)):
            if not np.array_equal(lst2[i].get_vector(), lst1[i].get_vector()):
                print "FAIL"
            if lst1[i].get_label() != lst2[i].get_label():
                print "FAIL"
        print "SUCCESS?"


    def createTestingInstances(self, images):
        start = time.time()
        lst2 = []
        for image in images:
            # print image[0]
            instance = self.__createInstances(image)
            lst2.append(instance)
        print "TEST: ", time.time() - start

        start = time.time()
        lst1 = par(images, self.bits)
        print "TEST: ", time.time() - start

        for i in range(len(lst1)):
            if not np.array_equal(lst2[i].get_vector(), lst1[i].get_vector()):
                print "FAIL"
            if lst1[i].get_label() != lst2[i].get_label():
                print "FAIL"
        print "SUCCESS?"
        # self.training_instances = 

    def __createInstances(self, image):
        
        img, label = image
        histogram = np.zeros(2 ** self.bits)
        img = read_color_image(img)
        rows, cols = img.shape[:2]
        count = 0
        for r in range(rows):
            for c in range(cols):
                blue = img.item((r,c,0))
                green = img.item((r,c,1))
                red = img.item((r,c,2))

                encoding = 0
                encoding += int(red * (2 ** (self.bits / 3.0)) /256)
                encoding <<= int(self.bits / 3.0)
                encoding += int(green * (2 ** (self.bits / 3.0)) /256)
                encoding <<= int(self.bits / 3.0)
                encoding += int(blue * (2 ** (self.bits / 3.0)) /256)

                histogram[encoding] += 1

        return Instance(histogram, label)

    def getTrainingInstance(self, index):
        return self.training_instances[index]
    
    def getTestingInstance(self, index):
        return self.testing_instances[index]
    
    def trainingInstancesSize(self):
        return len(self.training_instances)
    
    def testingInstancesSize(self):
        return len(self.testing_instances)

    def getTrainingLabel(self, index):
        return self.training_instances[index].get_label()

    def getTestingLabel(self, label):
        return self.testing_instances[index].get_label()



def local(image, bit):
    img, label = image
    # print img
    histogram = np.zeros(2 ** bit)
    img = read_color_image(img)
    rows, cols = img.shape[:2]
    count = 0
    for r in range(rows):
        for c in range(cols):
            blue = img.item((r,c,0))
            green = img.item((r,c,1))
            red = img.item((r,c,2))

            encoding = 0
            encoding += int(red * (2 ** (bit / 3.0)) /256)
            encoding <<= int(bit / 3.0)
            encoding += int(green * (2 ** (bit / 3.0)) /256)
            encoding <<= int(bit / 3.0)
            encoding += int(blue * (2 ** (bit / 3.0)) /256)

            histogram[encoding] += 1

    return Instance(histogram, label)

def par(images, bits):
    p = Pool()
    # bits = [bits] * len(images)
    print bits
    partial_local = partial(local, bit=bits)
    return p.map(partial_local, images)

