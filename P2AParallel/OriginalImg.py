from project_types import FeatureConverter, Instance
import cv2
from images import *
import numpy as np
import pickle

class OriginalImg(FeatureConverter):

    def __init__(self):
        self.training_images = None
        self.testing_images = None

    def createTrainingInstances(self, images):
        self.training_images = images

    def createTestingInstances(self, images):
        self.testing_images = images

    def getTrainingInstance(self, index):
        img = read_color_image(self.training_images[index][0])
        img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
        return Instance(img, self.training_images[index][1])
    
    def getTestingInstance(self, index):
        img = read_color_image(self.testing_images[index][0])
        img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
        return Instance(img, self.testing_images[index][1])
    
    def trainingInstancesSize(self):
        return len(self.training_images)
    
    def testingInstancesSize(self):
        return len(self.testing_images)

    def getTrainingLabel(self, index):
        return self.training_images[index][1]

    def getTestingLabel(self, label):
        return self.testing_images[index][1]
