from project_types import FeatureConverter
import cv2
import numpy as np

def BagOfWords(FeatureConverter):

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
    
    def testingInstanceSize(self):
        return len(self.testing_instances)
