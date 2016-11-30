import numpy as np
from project_types import *

class DualSVM(Predictor):

	def __init__(self, label = 1, SVM_lam = 1e-4, iterations = 5):
		self.label = label
		self.alphas = None
		self.lam = SVM_lam
		self.iterations = iterations


	def train(self, instances):
		self.instances = instances
		pass


	def predict(self, instance):
		score_value = self.score(instance)
		if score_value >= 0.0:
            return 1
        return 0



	def score(self, instance):
		vector = instance.get_vector()
		ret = 0
		for i in range(len(self.alphas)):
			if self.alphas[i] != 0:
				if self.label == self.instances[i].get_label():
					ret += vector.dot(self.instances[i].get_vector())
				else:
					ret -= vector.dot(self.instances[i].get_vector())
		return ret



