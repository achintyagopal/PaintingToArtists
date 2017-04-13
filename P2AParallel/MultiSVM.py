import numpy as np
from project_types import *
from DualSVM import DualSVM

class MultiSVM(Predictor):

    def __init__(self, lambda_fn = lambda x, y: x.dot(y) , SVM_lam=1e-4, iterations = 10):
        self.svm = {}
        self.lam = SVM_lam
        self.iterations = iterations
        self.lambda_fn = lambda_fn


    def train(self, feature_converter):
        labels = set()
        for i in range(feature_converter.trainingInstancesSize()):
            label = feature_converter.getTrainingLabel(i)
            labels.add(label)
        self.labels = list(labels)

        for l in labels:
            self.svm[l] = DualSVM(l, self.lambda_fn, self.lam, self.iterations)
            self.svm[l].train(feature_converter)

        print "Trained"

        total = 0
        correct = 0
        for i in range(feature_converter.trainingInstancesSize()):

            instance = feature_converter.getTrainingInstance(i)
            max_label = None
            max_val = None
            for label, d_svm in self.svm.iteritems():
                score = d_svm.score(instance)
                if max_val is None or score > max_val:
                    max_val = score
                    max_label = label

            print max_label
            print instance.get_label()
            if max_label == str(instance.get_label()):
                correct += 1
            total += 1

        print "Training Accuracy for ", self.iterations, " iterations: ", (correct / float(total) * 100), "%"
        self.predict(feature_converter)

    def predict(self, feature_converter):
        labels = []

        total = 0
        correct = 0

        for i in range(feature_converter.testingInstancesSize()):
            instance = feature_converter.getTestingInstance(i)
            max_label = None
            max_val = None
            for label, d_svm in self.svm.iteritems():
                score = d_svm.score(instance)
                if max_val is None or score > max_val:
                    max_val = score
                    max_label = label
            labels.append(max_label)
            if max_label == str(instance.get_label()):
                correct += 1
            total += 1

        print "Testing Accuracy: ", (correct / float(total) * 100), "%"
        return labels
