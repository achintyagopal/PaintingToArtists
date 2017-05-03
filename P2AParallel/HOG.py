from project_types import FeatureConverter, Instance
import numpy as np
from images import *
import cv2
from multiprocessing import Pool
# from joblib import Parallel, delayed
from ipyparallel import Client
from functools import partial
# import threading
# import Queue
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
		end = time.time() - start
		self.training_instances = instances
		print "HOG TRAIN SERIAL: %d images -> %d" % (len(images), end)

	def createTestingInstances(self, images):
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
		end = time.time() - start
		self.testing_instances = instances
		print "HOG TEST SERIAL: %d images -> %d" % (len(images), end)


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

	def setTestingInstances(self, inst):
		self.testing_instances = inst

	def setTrainingInstances(self, inst):
		self.training_instances = inst


HOGDESC = cv2.HOGDescriptor()


def local_hog(image):
	# HOGDESC = cv2.HOGDescriptor()
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


def native_par_hog(images, procs):
	start = time.time()
	p = Pool(procs)
	ret = p.map(local_hog, images)
	end = time.time() - start
	print "HOG NATIVE: %d images -> %d" % (len(images), end)
	return ret


def ipython_par_hog(images):
	start = time.time()
	c = Client()
	dview = c[:]
	ret = dview.map_sync(local_hog, images)
	end = time.time() - start
	print "HOG IPYTHON: %d images -> %d" % (len(images), end)
	return ret


# def ipypar(images):
# 	pass
	# Job Lib
	# start = time.time()
	# ret = Parallel(n_jobs=2)(delayed(local)(i) for i in images)
	# print "JL", time.time() - start
	
	# Threading
	# start = time.time()
	# threads = []
	# q = Queue.Queue()
	# for i in range(len(images)):
	# 	t = threading.Thread(target=localthr, args=(images[i],q,))
	# 	threads.append(t)
	# 	t.start()
	# for i in threads:
	# 	i.join()	
	# print "JLTH", time.time() - start
	# print q.qsize()
	# return list(q.queue)

	# IPY
	# c = Client()
	# print c.ids
	# dview = c[:]
	# # dview.push(HOGDESC)
	# # with dview.sync_imports():
   		# # import sys
   		# #sys.path[:] = ["/home/bill/Desktop/PaintingToArtists/P2AParallel"]
	# print "LOC", sys.path
	# print dview.map_sync(par, range(1))
	# dview.block = False
	# c.TaskScheduler.hwm = 1
	# dview = c[:] #c.load_balanced_view()
	# # dview.block = True
	# start = time.time()
	
	# ret = dview.map(local, images, block=True)
	# # while not ret.ready():
	# # 	pass
	# # while not ret.ready():
	# 	# time.sleep(.1)
	# print "DV", time.time() - start
	
	# dview = c.load_balanced_view()
	# dview.block = False
	# start = time.time()
	# ret = dview.map(local, images)
	# while not ret.ready():
		# time.sleep(1)
	# print "LB", time.time() - start
	# print ret
	# return ret

# def localthr(image, q):
# 	HOGDESC = cv2.HOGDescriptor()
# 	img, label = image
# 	img = read_color_image(img)
# 	img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
# 	descriptor = HOGDESC.compute(img)
# 	if descriptor is None:
# 		descriptor = []
# 	else:
# 		descriptor = descriptor.ravel()
# 	pairing = Instance(descriptor, label)
# 	q.put(pairing)