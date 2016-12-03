import numpy as np
from project_types import *
from random import shuffle

class StructuredSVM():

    def __init__(self, lambda_fn = lambda x, y: x.dot(y) , SVM_lam=1e-4, iterations = 5):
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
        for _ in range(self.iterations):
            instance_order = []
            for i in range(feature_converter.trainingInstancesSize()):
                instance_order.append(i)

            shuffle(instance_order)

            for i in instance_order:
            # for i in range(feature_converter.trainingInstancesSize()):
                instance = feature_converter.getTrainingInstance(i)

                # find max score
                max_index, max_score = self.score(instance)
                self.alphas_all = self.alphas_all * (1 - 1.0/t)
                learningRate  = 1.0 / (self.lam * t)

                if self.labels[max_index] != str(instance.get_label()):
                    self.alphas_all[max_index][i] -= learningRate
                    self.alphas_all[self.label_dict[instance.get_label()]][i] += learningRate
                else:
                    if max_score < 1:
                        self.alphas_all[max_index][i] += learningRate
                t += 1


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

                scores[j] += alpha * instance.get_vector().dot(training_instance.get_vector())

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
        for i in range(feature_converter.testingInstancesSize()):
            instance = feature_converter.getTestingInstance(i)
            instance = feature_converter.getTrainingInstance(i)
            max_index, max_score = self.score(instance)
            labels.append(self.labels[max_index])
        return labels
