import numpy as np
from project_types import *
from random import shuffle

class StructuredSVM():

    def __init__(self, iterations = 5, SVM_lam=1e-4, lambda_fn = lambda x, y: x.dot(y)):
        self.svm = {}
        self.lam = SVM_lam
        self.iterations = iterations
        self.lambda_fn = lambda_fn


    def train(self, feature_converter):
        self.labels = set()
        for i in range(feature_converter.trainingInstancesSize()):
            label = feature_converter.getTrainingLabel(i)
            self.labels.add(str(label))
        self.labels = list(self.labels)

        self.label_dict = {}
        i = 0
        for label in self.labels:
            self.label_dict[str(label)] = i
            i += 1


        self.alphas_all = []
        for _ in range(len(self.labels)):
            self.alphas_all.append(np.zeros(feature_converter.trainingInstancesSize()))

        self.alphas_all = np.array(self.alphas_all)

        self.feature_converter = feature_converter

        t = 1
        for iteration in range(self.iterations):
            print iteration
            instance_order = []
            for i in range(feature_converter.trainingInstancesSize()):
                instance_order.append(i)

            shuffle(instance_order)

            for i in instance_order:
                instance = feature_converter.getTrainingInstance(i)
                instance_label = feature_converter.getTrainingLabel(i)
                # find max score
                max_index, max_score = self.score(instance)
                self.alphas_all = self.alphas_all * (1 - 1.0/t)
                learningRate  = 1.0 / (self.lam * t)

                # print self.labels[max_index], str(instance.get_label())

                if self.labels[max_index] != str(instance_label):
                    self.alphas_all[max_index][i] -= learningRate
                    self.alphas_all[self.label_dict[instance_label]][i] += learningRate
                else:
                    if max_score < 1:
                        self.alphas_all[max_index][i] += learningRate
                t += 1


            total = 0
            correct = 0
            for i in instance_order:

                instance = feature_converter.getTrainingInstance(i)
                max_index, max_score = self.score(instance)

                if self.labels[max_index] == str(instance.get_label()):
                    correct += 1
                total += 1

            print "Training Accuracy for ", iteration, " iterations: ", (correct / float(total) * 100), "%"
            self.predict(feature_converter)

    def score(self, instance):
        
        scores = [0] * len(self.alphas_all)

        for i in range(len(self.alphas_all[0])):
            training_instance = None
            for j in range(len(self.alphas_all)):
                alpha = self.alphas_all.item((j,i))
                if alpha == 0:
                    continue
                elif training_instance is None:
                    training_instance = self.feature_converter.getTrainingInstance(i)

                scores[j] += alpha * self.lambda_fn(instance.get_vector(), training_instance.get_vector())

        max_index = None
        max_score = None
        i = 0
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score
                max_index = i
            i += 1

        return max_index, max_score

    def predict(self, feature_converter):
        labels = []
        total = 0
        correct = 0
        for i in range(feature_converter.testingInstancesSize()):
            instance = feature_converter.getTestingInstance(i)
            max_index, max_score = self.score(instance)
            labels.append(self.labels[max_index])
            if self.labels[max_index] == str(instance.get_label()):
                correct += 1
            total += 1

        print "Testing Accuracy: ", (correct / float(total) * 100), "%"
        return labels
