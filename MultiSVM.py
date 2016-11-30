import numpy as np
from project_types import *


class MultiSVM(Predictor):

	def __init__(self, SVM_lam=1e-4, iterations = 5):
		self.svm = {}
		self.lam = SVM_lam
		self.iterations = iterations


	def train(self, instances):
		labels = set()
		for vector, label in instances:
				labels.add(label)
		for l in labels:
			self.svm[l] = DualSVM(l, self.lam, self.iterations)
			self.svm[l].train(instances)

	def predict(self, instance):
		max_label = None
		max_val = None
		for label, d_svm in self.svm.iteritems():
			score = d_svm.score(instance)
			if max_val is None or score > max_val:
				max_val = score
				max_label = label
		return max_label

# END OF FILE