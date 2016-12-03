import numpy as np
from project_types import *
from DualSVM import DualSVM

class MultiSVM(Predictor):

	def __init__(self, SVM_lam=1e-4, iterations = 5):
		self.svm = {}
		self.lam = SVM_lam
		self.iterations = iterations


	def train(self, feature_converter):
		labels = set()
		for i in range(feature_converter.trainingInstancesSize()):
			label = feature_converter.getTrainingLabel(i)
			labels.add(label)

		for l in labels:
			self.svm[l] = DualSVM(l, self.lam, self.iterations)
			self.svm[l].train(feature_converter)

	def predict(self, feature_converter):
		labels = []
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
		return labels
