from project_types import FeatureConverter, Instance
import numpy as np
from images import *
import cv2


class HOG(FeatureConverter):
	
	def __init__(self):
		self.training_instances = None
		self.testing_instances = None


	def createTrainingInstances(self, images):
		hog = cv2.HOGDescriptor()
		instances = []
		for img, label in images:
			img = read_color_image(img)
			descriptor = hog.compute(img)
			if descriptor is None:
				descriptor = []
			else:
				descriptor = descriptor.ravel()
			pairing = Instance(descriptor, label)
			instances.append(pairing)
		self.training_instances = instances


	def createTestingInstances(self, images):
		hog = cv2.HOGDescriptor()
		instances = []
		for img, label in images:
			img = read_color_image(img)
			descriptor = hog.compute(img)
			if descriptor is None:
				descriptor = []
			else:
				descriptor = descriptor.ravel()
			pairing = Instance(descriptor, label)
			instances.append(pairing)
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
