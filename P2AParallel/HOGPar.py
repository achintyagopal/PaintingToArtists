from project_types import FeatureConverter, Instance
import numpy as np
from images import *
import cv2
from multiprocessing import Pool
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
		print "TRAIN: ", time.time()-start
		# 3.805
		# p = Pool()
		# print images
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
		print "TRAIN: ", time.time()-start
		# lst2 = p.map(self.local, images)
		# print lst2[0].get_vector()
		# print lst1[0].get_vector()
		# print np.array_equal(lst2[0].get_vector(),  lst1[0].get_vector())
		# for i in range(len(lst1)):
			# if lst1[i].get_label() != lst2[i].get_label():
				# print "FAIL"
		# print "SUCCESS?"


	def createTestingInstances(self, images):
		start = time.time()
		self.testing_instances = par(images)
		print "TEST: ", time.time()-start
		# 1.3682
		# hog = cv2.HOGDescriptor()
		# instances = []
		# for img, label in images:
		# 	print img
		# 	img = read_color_image(img)
		# 	img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
		# 	descriptor = hog.compute(img)
		# 	if descriptor is None:
		# 		descriptor = []
		# 	else:
		# 		descriptor = descriptor.ravel()
		# 	pairing = Instance(descriptor, label)
		# 	instances.append(pairing)
		# self.testing_instances = instances


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

hog = cv2.HOGDescriptor()

def local(image):
	img, label = image
	# print img
	img = read_color_image(img)
	img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
	descriptor = hog.compute(img)
	if descriptor is None:
		descriptor = []
	else:
		descriptor = descriptor.ravel()
	pairing = Instance(descriptor, label)
	# print "HI"
	return pairing

def par(images):
	p = Pool()
	
	return p.map(local, images)
