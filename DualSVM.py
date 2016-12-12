import numpy as np
from random import shuffle
from project_types import *

class DualSVM(Predictor):

    def __init__(self, label = 1, lambda_fn = lambda x,y: x.dot(y), SVM_lam = 1e-4, iterations = 5):
        self.label = label
        self.alphas = None
        self.lam = SVM_lam
        self.iterations = iterations
        self.lambda_fn = lambda_fn

    def train(self, feature_converter):
        t = 1
        constant = 1
        self.feature_converter = feature_converter
        self.alphas = np.zeros(feature_converter.trainingInstancesSize())

        j = 0
        for j in range(self.iterations):
            print j
            instance_order = []
            for i in range(feature_converter.trainingInstancesSize()):
                instance_order.append(i)

            shuffle(instance_order)

            for i in instance_order:

                instance = feature_converter.getTrainingInstance(i)
                yBar =  self.score(instance)

                if self.label != instance.get_label():
                    y = -1
                else:
                    y = 1

                learningRate = 1.0 / (self.lam * t)
                constant *= 1 - 1.0/t

                if yBar * y < 1:
                    self.alphas = constant * self.alphas
                    if self.label == instance.get_label():
                        self.alphas[i] += learningRate * y
                    else:
                        self.alphas[i] += learningRate * y
                    constant = 1

                t += 1

            # total = 0
            # correct = 0
            # for i in instance_order:

            #     instance = feature_converter.getTrainingInstance(i)
            #     score = self.score(instance)

            #     if score > 0 and self.label == str(instance.get_label()):
            #         correct += 1
            #     elif score < 0 and self.label != str(instance.get_label()):
            #         correct += 1
            #     total += 1

            # print "Training Accuracy for ", j + 1, " iterations: ", (correct / float(total) * 100), "%"

            # j += 1
            # self.predict(feature_converter)
            # if correct / float(total) * 100 > 95:
                # break


        if constant != 1:
            self.alphas = constant * self.alphas


    def predict(self, feature_converter):
        labels = []

        total = 0
        correct = 0
        for i in range(feature_converter.testingInstancesSize()):
            instance = feature_converter.getTestingInstance(i)

            score_value = self.score(instance)
            if score_value >= 0.0:
                if self.label == str(instance.get_label()):
                    correct += 1
                labels.append(1)
            else:
                if self.label != str(instance.get_label()):
                    correct += 1
                labels.append(0)
            total += 1
        # print "Testing Accuracy: ", (correct / float(total) * 100), "%"
        return labels

    def score(self, instance):
        vector = instance.get_vector()
        ret = 0
        for i in range(len(self.alphas)):
            if self.alphas[i] != 0:
                training_instance = self.feature_converter.getTrainingInstance(i).get_vector()
                ret += self.alphas[i] * self.lambda_fn(vector, training_instance)

        return ret



