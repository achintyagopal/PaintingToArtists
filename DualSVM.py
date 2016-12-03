import numpy as np
from random import shuffle
from project_types import *

class DualSVM(Predictor):

	def __init__(self, label = 1, SVM_lam = 1e-4, iterations = 5):
		self.label = label
		self.alphas = None
		self.lam = SVM_lam
		self.iterations = iterations

	def train(self, feature_converter):
		t = 1
		constant = 1
		self.feature_converter = feature_converter
		self.alphas = np.zeros(feature_converter.trainingInstancesSize())

		for j in range(self.iterations):
			print j
			instance_order = []
			for i in range(feature_converter.trainingInstancesSize()):
				instance_order.append(i)

			shuffle(instance_order)

			for i in range(feature_converter.trainingInstancesSize()):
				print i
				instance = feature_converter.getTrainingInstance(i)
				yBar =  self.score(instance)

				if self.label == instance.get_label():
					y = -1
				else:
					y = 1

				learningRate = 1.0 / (self.lam * t)
				constant *= 1 - 1.0/t

				if yBar * y < 1:
					self.alphas = constant * self.alphas
					self.alphas[i] += learningRate * y
					constant = 1

				t += 1

		if constant != 1:
			self.alphas = constant * self.alphas


	def predict(self, feature_converter):
		labels = []

		for i in range(feature_converter.testingInstancesSize()):
			instance = feature_converter.getTestingInstance(i)

			score_value = self.score(instance)
			if score_value >= 0.0:
				labels.append(1)
	        else:
	        	labels.append(0)
		return labels

	def score(self, instance):
		vector = instance.get_vector()
		ret = 0
		for i in range(len(self.alphas)):
			if self.alphas[i] != 0:
				training_instance = self.feature_converter.getTrainingInstance(i)
				ret += self.alphas[i] * vector.dot(training_instance.get_vector())
		return ret



