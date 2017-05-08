from project_types import FeatureConverter, Instance
import numpy as np
from images import *
import cv2
from multiprocessing import Pool
# from joblib import Parallel, delayed
from ipyparallel import Client
import ipyparallel as ipp
from functools import partial
# import distributed as dask
# import threading
# import Queue
import time
import itertools
# from numba import jit
# import pp



class HOG(FeatureConverter):
	
	def __init__(self):
		self.training_instances = None
		self.testing_instances = None

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
		print "HOG TRAIN SERIAL: %d images -> %f" % (len(images), end)

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
		print "HOG TEST SERIAL: %d images -> %f" % (len(images), end)


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

# @jit
def local_hog(image):
	HOGDESC = cv2.HOGDescriptor()
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

def local_par_hog(images):
	inst = list()
	HOGDESC = cv2.HOGDescriptor()

	for image in images:
		img, label = image
		img = read_color_image(img)
		img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
		descriptor = HOGDESC.compute(img)
		if descriptor is None:
			descriptor = []
		else:
			descriptor = descriptor.ravel()
		pairing = Instance(descriptor, label)
		inst.append(pairing)
	return inst

def native_partition(images, procs, n):
	p = Pool(procs)
	partitions = [images[i:i + n] for i in range(0, len(images), n)]

	start = time.time()
	ret1 = [p.apply(local_par_hog, args=(i,)) for i in partitions]
	end = time.time() - start
	ret1 = list(itertools.chain.from_iterable(ret1))
	print "HOG NATIVE APP: %d images -> %f" % (len(ret1), end)

	start = time.time()
	ret2 = p.map(local_par_hog, partitions)
	end = time.time() - start
	ret2 = list(itertools.chain.from_iterable(ret2))
	print "HOG NATIVE MAP: %d images -> %f" % (len(ret2), end)
	
	start = time.time()
	resu = [p.apply_async(local_par_hog, args=(i,)) for i in partitions]
	ret3 = [r.get() for r in resu]
	end = time.time() - start
	ret3 = list(itertools.chain.from_iterable(ret3))
	print "HOG NATIVE ASY: %d images -> %f" % (len(ret3), end)
	
	# serial equivalence
	# for i in range(len(ret1)):
	# 	if not np.array_equal(ret2[i].get_vector(),  ret1[i].get_vector()) or not np.array_equal(ret2[i].get_vector(),  ret3[i].get_vector()):
	# 		raise Exception()
	# 	if ret1[i].get_label() != ret2[i].get_label() or ret2[i].get_label() != ret3[i].get_label():
	# 		raise Exception()
	return ret3



def native_par_hog(images, procs):
	p = Pool(procs)

	start = time.time()
	ret1 = [p.apply(local_hog, args=(i,)) for i in images]
	end = time.time() - start
	print "HOG NATIVE APP: %d images -> %f" % (len(ret1), end)

	start = time.time()
	ret2 = p.map(local_hog, images, chunksize=2)
	end = time.time() - start
	print "HOG NATIVE MAP: %d images -> %f" % (len(ret2), end)
	
	start = time.time()
	resu = [p.apply_async(local_hog, args=(i,)) for i in images]
	ret3 = [r.get() for r in resu]
	end = time.time() - start
	print "HOG NATIVE ASY: %d images -> %f" % (len(ret3), end)
	
	# serial equivalence
	# for i in range(len(ret1)):
	# 	if not np.array_equal(ret2[i].get_vector(),  ret1[i].get_vector()) or not np.array_equal(ret2[i].get_vector(),  ret3[i].get_vector()):
	# 		raise Exception()
	# 	if ret1[i].get_label() != ret2[i].get_label() or ret2[i].get_label() != ret3[i].get_label():
	# 		raise Exception()
	return ret3


def ipython_partition(images, direct, n):
	c = Client()
	partitions = [images[i:i + n] for i in range(0, len(images), n)]
	if direct:
		dview = c[:]
		dview.block = False
		num_clients = len(c.ids)


		start = time.time()
		ret1 = [c[i % num_clients].apply_sync(local_par_hog, partitions[i]) for i in xrange(len(partitions))]
		end = time.time() - start
		ret1 = list(itertools.chain.from_iterable(ret1))
		print "HOG IPYTHON DIRECT APP: %d images -> %f" % (len(ret1), end)

		start = time.time()
		ret2 = dview.map_sync(local_par_hog, partitions)
		end = time.time() - start
		ret2 = list(itertools.chain.from_iterable(ret2))
		print "HOG IPYTHON DIRECT MAP: %d images -> %f" % (len(ret2), end)

		start = time.time()
		rets = [c[i % num_clients].apply_async(local_par_hog, partitions[i]) for i in xrange(len(partitions))]
		ret3 = [r.get() for r in rets]
		end = time.time() - start
		ret3 = list(itertools.chain.from_iterable(ret3))
		print "HOG IPYTHON DIRECT ASY: %d images -> %f" % (len(ret3), end)
		
		return ret3
	else:
		dview = c.load_balanced_view()
		dview.block = False
		
		start = time.time()
		ret1 = [dview.apply_sync(local_par_hog, i) for i in partitions]
		end = time.time() - start
		ret1 = list(itertools.chain.from_iterable(ret1))
		print "HOG IPYTHON LBV APP: %d images -> %f" % (len(ret1), end)

		start = time.time()
		ret2 = dview.map_sync(local_par_hog, partitions)
		end = time.time() - start
		ret2 = list(itertools.chain.from_iterable(ret2))
		print "HOG IPYTHON LBV MAP: %d images -> %f" % (len(ret2), end)
		
		start = time.time()
		rets = [dview.apply_async(local_par_hog, i) for i in partitions]
		ret3 = [r.get() for r in rets]
		end = time.time() - start
		ret3 = list(itertools.chain.from_iterable(ret3))
		print "HOG IPYTHON LBV ASY: %d images -> %f" % (len(ret3), end)
		return ret3		


def ipython_par_hog(images, direct):
	c = Client()
	clients = len(c.ids)
	# num_per_call = 10 #int(len(images) / float(clients))
	# partitions = [images[i:i + num_per_call] for i in range(0, len(images), num_per_call)]

	if direct:
		dview = c[:]
		dview.block = False
		num_clients = len(c.ids)

		start = time.time()
		ret1 = [c[i % num_clients].apply_sync(local_hog, images[i]) for i in xrange(len(images))]
		end = time.time() - start
		print "HOG IPYTHON DIRECT APP: %d images -> %f" % (len(ret1), end)
		# ret1 = [c[i % num_clients].apply_sync(local_par_hog, partitions[i]) for i in xrange(len(partitions))]
		# end = time.time() - start
		# print "HOG IPYTHON DIRECT APP: %d images -> %f" % (len(ret1), end)


		start = time.time()
		ret2 = dview.map_sync(local_hog, images)
		end = time.time() - start
		print "HOG IPYTHON DIRECT MAP: %d images -> %f" % (len(ret2), end)

		start = time.time()
		rets = [c[i % num_clients].apply_async(local_hog, images[i]) for i in xrange(len(images))]
		ret3 = [r.get() for r in rets]
		end = time.time() - start
		print "HOG IPYTHON DIRECT ASY: %d images -> %f" % (len(ret3), end)

		# for i in range(len(ret1)):
		# 	if not np.array_equal(ret2[i].get_vector(),  ret1[i].get_vector()) or not np.array_equal(ret2[i].get_vector(),  ret3[i].get_vector()):
		# 		raise Exception()
		# 	if ret1[i].get_label() != ret2[i].get_label() or ret2[i].get_label() != ret3[i].get_label():
		# 		raise Exception()
		return ret3
	else:
		dview = c.load_balanced_view()
		dview.block = False
		
		start = time.time()
		ret1 = [dview.apply_sync(local_hog, i) for i in images]
		end = time.time() - start
		print "HOG IPYTHON LBV APP: %d images -> %f" % (len(ret1), end)

		start = time.time()
		ret2 = dview.map_sync(local_hog, images, chunksize=2)
		end = time.time() - start
		print "HOG IPYTHON LBV MAP: %d images -> %f" % (len(ret2), end)
		
		start = time.time()
		rets = [dview.apply_async(local_hog, i) for i in images]
		ret3 = [r.get() for r in rets]
		end = time.time() - start
		print "HOG IPYTHON LBV ASY: %d images -> %f" % (len(ret3), end)
		
		# for i in range(len(ret1)):
		# 	if not np.array_equal(ret2[i].get_vector(),  ret1[i].get_vector()) or not np.array_equal(ret2[i].get_vector(),  ret3[i].get_vector()):
		# 		raise Exception()
		# 	if ret1[i].get_label() != ret2[i].get_label() or ret2[i].get_label() != ret3[i].get_label():
		# 		raise Exception()
		return ret3
