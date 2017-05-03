from project_types import FeatureConverter, Instance
import numpy as np
from images import *
import cv2
from multiprocessing import Pool
# from joblib import Parallel, delayed
from ipyparallel import Client
import ipyparallel as ipp
from functools import partial
import distributed as dask
# import threading
# import Queue
import time
from numba import jit
import pp



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


def native_par_hog(images, procs):
	start = time.time()
	p = Pool(procs)
	ret = p.map(local_hog, images, chunksize=2)
	# ret = map(local_hog, images)
	end = time.time() - start
	print "HOG NATIVE: %d images -> %f" % (len(images), end)
	return ret


def ipython_par_hog(images, direct):
	if direct:
		start = time.time()
		# ipp.register_joblib_backend()
		c = Client()
		dview = c[:]
		# dview.register_joblib_backend()
		# asyn_ret = list()
		# for i in images:
			# ret = dview.map(local_hog, images, block=False, chunksize=100)
			# ret = dview.apply_async(local_hog, i)
			# asyn_ret.append(ret)
		# c.wait(asyn_ret)
		# ch = int(len(images)/3.0)
		ret = dview.map_sync(local_hog, images)
		end = time.time() - start
		# print ret
		print "HOG IPYTHON DIRECT: %d images -> %f" % (len(ret), end)
		return ret
	else:
		start = time.time()
		c = Client()
		# num = int(len(images) / float(len(c.ids)))
		# ipp.register_joblib_backend()
		dview = c.load_balanced_view()
		# asyn_ret = list()
		# for i in images:
		ret = dview.map_sync(local_hog, images, chunksize=int(len(images) / float(len(c.ids))))
			# ret = dview.apply_async(local_hog, i)
			# asyn_ret.append(ret)
		# c.wait(asyn_ret)
		end = time.time() - start
		print "HOG IPYTHON LBV: %d images -> %f" % (len(images), end)
		return ret

# #PP
# 	start = time.time()
# 	# HOGDESC = cv2.HOGDescriptor()
# 	job_server = pp.Server()
# 	print job_server.get_ncpus()
# 	jobs = list()
# 	for i in images:
# 		f = job_server.submit(local_hog, (i,),(read_color_image,Instance,),("numpy", "cv2",))
# 		jobs.append(f)
# 	ret = list()
# 	for f in jobs:
# 		ret.append(f())
# 	end = time.time() - start
# 	print len(ret), end

# DASK
# start = time.time()
	# client = dask.Client()
	# print client
	# time.sleep(5)
	# ret = client.map(local_hog, images)
	# ret = client.gather(ret)
	# end = time.time() - start
	# print end
	# print len(ret)
	# print ret
	# return ret
	# time.sleep(5)
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