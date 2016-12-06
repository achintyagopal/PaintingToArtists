from project_types import FeatureConverter
import cv2
from images import *
import numpy as np

class BagOfWords(FeatureConverter):

    def __init__(self):
        self.training_instances = None
        self.testing_instances = None

    def createTrainingInstances(self, images):
        instances = []
        cv2.ocl.setUseOpenCL(False)
        orb = cv2.ORB_create()
        for img, label in images:
            img = read_color_image(img)
            pass

        self.training_instances = instances
    
    def createTestingInstances(self, images):
        instances = []
        cv2.ocl.setUseOpenCL(False)
        orb = cv2.ORB_create()
        for img, label in images:
            img = read_color_image(img)
            pass
            
        self.testing_instances = instances

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
