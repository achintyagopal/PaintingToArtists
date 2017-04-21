from project_types import FeatureConverter, Instance
import numpy as np
from images import *
import cv2
from multiprocessing import Pool
from functools import partial
import time



class HOG(FeatureConverter):
	
	def __init__(self):
		self.training_instances = None
		self.testing_instances = None
		self.hog = cv2.HOGDescriptor()

	def local(self, image):
		img, label = image
		print img
		img = read_color_image(img)
		img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
		descriptor = self.hog.compute(img)
		if descriptor is None:
			descriptor = []
		else:
			descriptor = descriptor.ravel()
		pairing = Instance(descriptor, label)
		# print "HI"
		return pairing

	def createTrainingInstances(self, images):
		start = time.time()
		lst1 = par(images)
		print "TRAIN PAR: ", time.time()-start
		# 3.805

		start = time.time()
		hog = cv2.HOGDescriptor()
		instances = []
		for img, label in images:
			# print img
			img = read_color_image(img)
			img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
			descriptor = hog.compute(img)
			if descriptor is None:
				descriptor = []
			else:
				descriptor = descriptor.ravel()
			pairing = Instance(descriptor, label)
			instances.append(pairing)
		lst2 = instances

		print "TRAIN SER: ", time.time()-start
		for i in range(len(lst1)):
			if not np.array_equal(lst2[i].get_vector(),  lst1[i].get_vector()):
				raise Exception()
			if lst1[i].get_label() != lst2[i].get_label():
				raise Exception()


	def createTestingInstances(self, images):
		start = time.time()
		self.testing_instances = par(images)
		print "TEST PAR: ", time.time()-start
		
		# 1.3682
		start = time.time()
		hog = cv2.HOGDescriptor()
		instances = []
		for img, label in images:
			# print img
			img = read_color_image(img)
			img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
			descriptor = hog.compute(img)
			if descriptor is None:
				descriptor = []
			else:
				descriptor = descriptor.ravel()
			pairing = Instance(descriptor, label)
			instances.append(pairing)
		# self.testing_instances = instances
		print "TEST SER: ", time.time()-start


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


HOGDESC = cv2.HOGDescriptor()

def local(image):
	img, label = image
	img = read_color_image(img)
	img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
	descriptor = HOGDESC.compute(img)
	if descriptor is None:
		descriptor = []
	else:
		descriptor = descriptor.ravel()
	pairing = Instance(descriptor, label)
	return pairing

def par(images):
	p = Pool()
	return p.map(local, images)