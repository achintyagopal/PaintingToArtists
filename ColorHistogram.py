from images import *
from project_types import FeatureConverter, Instance
import cv2
import numpy as np

class ColorHistogram(FeatureConverter):

    def __init__(self, bits = 12):
        self.bits = bits

    def createTrainingInstances(self, images):
        self.training_instances = []
        for image in images:
            print image[0]
            instance = self.__createInstances(image)
            self.training_instances.append(instance)


    def createTestingInstances(self, images):
        self.testing_instances = []
        for image in images:
            print image[0]
            instance = self.__createInstances(image)
            self.testing_instances.append(instance)

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
