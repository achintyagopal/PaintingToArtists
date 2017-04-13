import numpy as np 
import cv2
import pickle

from project_types import *
from DualSVM import DualSVM
from structuredSVM import StructuredSVM

class AccuracyBooster(Predictor):

    def __init__(self, iterations = 5):
        self.iterations = iterations

    def train(self, color_feature_converter, hog_feature_converter):
        # instances = []
        # for i in range(color_feature_converter.trainingInstancesSize()):
            # instance = color_feature_converter.getTrainingInstance(i)
            # if instance.getLabel() == "Rembrant":
                # instances.append(instance)
            # else:
                # instances.append(Instance(instance.get_vector(), "Not"))
            # instances.append(instance)

        # self.dual_svm = DualSVM("Rembrant", iterations = 10)
        self.dual_svm = DualSVM(iterations = 10)
        self.dual_svm.train(color_feature_converter)

        self.struct_svm = StructuredSVM(iterations = 10, lambda_fn=lambda x,y:(x.dot(y)) ** 2)
        self.struct_svm.train(hog_feature_converter)

    def predict(self, color_feature_converter, hog_feature_converter):
        labels = []
        total = 0
        correct = 0
        for i in range(color_feature_converter.testingInstancesSize()):
            total += 1
            instance = color_feature_converter.getTestingInstance(i)
            score = self.dual_svm.score(instance)
            if score > 0:
                labels.append("Rembrant")
                if str(instance.get_label()) == "Rembrant":
                    correct += 1
            else:
                instance = hog_feature_converter.getTestingInstance(i)
                max_index, _ = self.struct_svm.score(instance)
                labels.append(self.struct_svm.labels[max_index])
                if labels[-1] == str(instance.get_label()):
                    correct += 1

        print "Testing Accuracy: ", (correct / float(total) * 100), "%"
        return labels

with open('color.feature.file', 'rb') as reader:
    color_feature_converter = pickle.load(reader)

with open('hog.feature.file', 'rb') as reader:
    hog_feature_converter = pickle.load(reader)

with open('hog2.feature.file', 'rb') as reader:
    hog_feature_converter_2 = pickle.load(reader)

print "Hog"
model = AccuracyBooster()
model.train(color_feature_converter, hog_feature_converter)
model.predict(color_feature_converter, hog_feature_converter_2)