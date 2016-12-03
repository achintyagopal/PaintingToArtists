from images import *
from project_types import FeatureConverter, Instance
import cv2
import numpy as np

class ColorHistogram(FeatureConverter):

    def __init__(self, bits = 12):
        self.bits = bits

    def createTrainingInstances(self, images):
        self.training_images = images
    
    def createTestingInstances(self, images):
        self.testing_images = images

    def __createInstances(self, image):
        img, label = image
        histogram = np.zeros(2 ** self.bits)
        img = read_color_image(img)
        rows, cols = img.shape[:2]
        print rows, cols
        for r in range(rows):
            for c in range(cols):
                blue = img.item((r,c,0))
                green = img.item((r,c,1))
                red = img.item((r,c,2))

                encoding = 0
                encoding += red / (2 ** (self.bits / 3))
                encoding << self.bits / 3;
                encoding += green / (2 ** (self.bits / 3))
                encoding << self.bits / 3;
                encoding += blue / (2 ** (self.bits / 3))

                histogram[encoding] += 1
        return Instance(histogram, label)

    def getTrainingInstance(self, index):
        return self.__createInstances(self.training_images[index])
    
    def getTestingInstance(self, index):
        return self.__createInstances(self.testing_images[index])
    
    def trainingInstancesSize(self):
        return len(self.training_images)
    
    def testingInstancesSize(self):
        return len(self.testing_images)

    def getTrainingLabel(self, index):
        return self.training_images[index][1]

    def getTestingLabel(self, label):
        return self.testing_images[index][1]
