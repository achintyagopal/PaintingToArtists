from project_types import FeatureConverter
import cv2
import numpy as np

class ColorHistogram(FeatureConverter):

    def createTrainingInstances(self, images):
        self.training_instances = self.__createInstances(images)
    
    def createTestingInstances(self, images):
        self.testing_instances = self.__createInstances(images)

    def __createInstances(self, images):
        instances = []
        for img, label in images:
            histogram = np.zeros(2 ** bits)
            img = read_color_image(img)
            rows, cols = img.shape[:2]
            for r in range(rows):
                for c in range(cols):
                    blue = img.item((r,c,0))
                    green = img.item((r,c,1))
                    red = img.item((r,c,2))

                    encoding = 0
                    encoding += red / (2 ** (bits / 3))
                    encoding << bits / 3;
                    encoding += green / (2 ** (bits / 3))
                    encoding << bits / 3;
                    encoding += blue / (2 ** (bits / 3))

                    histogram[encoding] += 1
            instances.append(Instance(histogram, label))
        return instances

    def getTrainingInstance(self, index):
        return self.training_instances[index]
    
    def getTestingInstance(self, index):
        return self.testing_instances[index]
    
    def trainingInstancesSize(self):
        return len(self.training_instances)
    
    def testingInstanceSize(self):
        return len(self.testing_instances)
