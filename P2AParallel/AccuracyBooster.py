import numpy as np 
import cv2
import dill as pickle
import math

from project_types import *
from DualSVM import DualSVM
from structuredSVM import StructuredSVM

class CustomFeatureConverter(FeatureConverter):

    def createTrainingInstances(self, instances):
        self.training_instances = instances
    
    def createTestingInstances(self, instances):
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


class AccuracyBooster(Predictor):

    def __init__(self, iterations = 5):
        self.iterations = iterations

    def train(self, color_feature_converter, hog1_feature_converter, hog2_feature_converter):
        
        feature_converter = CustomFeatureConverter()
        instances = []
        for i in range(color_feature_converter.trainingInstancesSize()):
            instance = color_feature_converter.getTrainingInstance(i)
            label = instance.get_label()
            if str(label) == "Rembrant":
                instance = Instance(instance.get_vector(), "0")
            elif str(label) in ("Dali", "Picasso"):
                instance = Instance(instance.get_vector(), "1")
            else:
                instance = Instance(instance.get_vector(), "2")
            
            instances.append(instance)

        feature_converter.createTrainingInstances(instances)
        instances = []
        for i in range(color_feature_converter.testingInstancesSize()):
            instance = color_feature_converter.getTestingInstance(i)
            label = instance.get_label()
            if str(label) == "Rembrant":
                instance = Instance(instance.get_vector(), "0")
            elif str(label) in ("Dali", "Picasso"):
                instance = Instance(instance.get_vector(), "1")
            else:
                instance = Instance(instance.get_vector(), "2")
            
            instances.append(instance)

        feature_converter.createTestingInstances(instances)

        self.color_svm = StructuredSVM(iterations = 10, lambda_fn = lambda x,y: math.e ** (-np.linalg.norm(x-y) ** 2/(2e9)))
        self.color_svm.train(feature_converter)


        self.svm_nature = StructuredSVM(iterations = 10, lambda_fn = lambda x,y: math.e ** (-np.linalg.norm(x-y) ** 2/(2000)))
        self.svm_nature.train(hog1_feature_converter)

        self.svm_cubist = StructuredSVM(iterations = 10, lambda_fn = lambda x,y: math.e ** (-np.linalg.norm(x-y) ** 2/(3000)))
        self.svm_cubist.train(hog2_feature_converter)


    def predict(self, color_feature_converter, hog_feature_converter):
        labels = []
        total = 0
        correct = 0
        for i in range(hog_feature_converter.testingInstancesSize()):
            total += 1
            # print i
            instance = color_feature_converter.getTestingInstance(i)
            index, score = self.color_svm.score(instance)
            # print self.color_svm.labels[index]
            if str(self.color_svm.labels[index]) == "0":
                # print "Rembrant"
                labels.append("Rembrant")
                if str(instance.get_label()) == "Rembrant":
                    correct += 1
            elif str(self.color_svm.labels[index]) == "1":
                instance = hog_feature_converter.getTestingInstance(i)
                max_index, _ = self.svm_nature.score(instance)
                # print max_index
                labels.append(self.svm_nature.labels[max_index])
                # print labels[-1]
                if labels[-1] == str(instance.get_label()):
                    correct += 1
            else:
                instance = hog_feature_converter.getTestingInstance(i)
                max_index, _ = self.svm_cubist.score(instance)
                # print max_index
                labels.append(self.svm_cubist.labels[max_index])
                # print labels[-1]
                if labels[-1] == str(instance.get_label()):
                    correct += 1


        print "Testing Accuracy: ", (correct / float(total) * 100), "%"
        return labels

with open('color9.feature.file', 'rb') as reader:
    color_feature_converter = pickle.load(reader)

with open('hog.feature.file', 'rb') as reader:
    hog_feature_converter = pickle.load(reader)

with open('hog2.feature.file', 'rb') as reader:
    hog_feature_converter_2 = pickle.load(reader)

with open('hog3.feature.file', 'rb') as reader:
    hog_feature_converter_3 = pickle.load(reader)

print "Hog"

for iteration in range(6, 12):
    model = AccuracyBooster()
    model.train(color_feature_converter, hog_feature_converter_2, hog_feature_converter_3)

    labels = model.predict(color_feature_converter, hog_feature_converter)

    filename = 'boost.' + str(iteration) + '.predictions'
    with open(filename, 'w') as writer:
        for i in range(len(labels)):
            label = labels[i]
            instance = color_feature_converter.getTestingInstance(i)

            writer.write(str(label))
            writer.write(' ')
            writer.write(instance.get_label())
            writer.write('\n')
