from images import *
from project_types import FeatureConverter, Instance
import cv2
import numpy as np
import time
from multiprocessing import Pool
# import threading
# import Queue
import itertools
from ipyparallel import Client
from functools import partial

class ColorHistogram(FeatureConverter):

    def __init__(self, bits = 12):
        self.bits = bits

    def createTrainingInstances(self, images):
        start = time.time()
        tmp = list()
        for image in images:
            # print image[0]
            instance = self.__createInstances(image)
            tmp.append(instance)
        end = time.time() - start
        self.training_instances = tmp
        print "COLOR SERIAL: %d images -> %d" % (len(images), end)
        
    def createTestingInstances(self, images):
        start = time.time()
        tmp = list()
        for image in images:
            # print image[0]
            instance = self.__createInstances(image)
            tmp.append(instance)
        end = time.time() - start
        self.testing_instances = tmp 
        print "COLOR SERIAL: %d images -> %f" % (len(images), end)

    def __createInstances(self, image):
        
        img, label = image
        histogram = np.zeros(2 ** self.bits)
        img = read_color_image(img)
        rows, cols = img.shape[:2]
        for r in xrange(rows):
            for c in xrange(cols):
                blue = img.item((r,c,0))
                green = img.item((r,c,1))
                red = img.item((r,c,2))

                encoding = 0
                encoding += int(red * (2 ** (self.bits / 3.0)) /256)
                encoding <<= int(self.bits / 3.0)
                encoding += int(green * (2 ** (self.bits / 3.0)) /256)
                encoding <<= int(self.bits / 3.0)
                encoding += int(blue * (2 ** (self.bits / 3.0)) /256)

                histogram[encoding] += 1

        return Instance(histogram, label)

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


def local_color(image, bit):
    img, label = image
    histogram = np.zeros(2 ** bit)
    img = read_color_image(img)
    rows, cols = img.shape[:2]
    for r in xrange(rows):
        for c in xrange(cols):
            blue = img.item((r,c,0))
            green = img.item((r,c,1))
            red = img.item((r,c,2))

            encoding = 0
            encoding += int(red * (2 ** (bit / 3.0)) /256)
            encoding <<= int(bit / 3.0)
            encoding += int(green * (2 ** (bit / 3.0)) /256)
            encoding <<= int(bit / 3.0)
            encoding += int(blue * (2 ** (bit / 3.0)) /256)

            histogram[encoding] += 1
    return Instance(histogram, label)

def local_color_partition(images, bit):
    inst = list()
    for image in images:
        img, label = image
        histogram = np.zeros(2 ** bit)
        img = read_color_image(img)
        rows, cols = img.shape[:2]
        for r in xrange(rows):
            for c in xrange(cols):
                blue = img.item((r,c,0))
                green = img.item((r,c,1))
                red = img.item((r,c,2))

                encoding = 0
                encoding += int(red * (2 ** (bit / 3.0)) /256)
                encoding <<= int(bit / 3.0)
                encoding += int(green * (2 ** (bit / 3.0)) /256)
                encoding <<= int(bit / 3.0)
                encoding += int(blue * (2 ** (bit / 3.0)) /256)

                histogram[encoding] += 1
        inst.append(Instance(histogram, label))
    return inst


def native_par_color(images, bits, procs):
    p = Pool(procs)
    partial_local = partial(local_color, bit=bits)
    
    # start = time.time()
    # ret1 = [p.apply(partial_local, args=(i,)) for i in images]
    # end = time.time() - start
    # print "COLOR NATIVE APP: %d images -> %f" % (len(ret1), end)

    start = time.time()
    ret2 = p.map(partial_local, images, chunksize=2)
    end = time.time() - start
    print "COLOR NATIVE MAP: %d images -> %f" % (len(ret2), end)
    
    start = time.time()
    resu = [p.apply_async(partial_local, args=(i,)) for i in images]
    ret3 = [r.get() for r in resu]
    end = time.time() - start
    print "COLOR NATIVE ASY: %d images -> %f" % (len(ret3), end)

    # serial equivalence
    # for i in range(len(ret1)):
    #     if not np.array_equal(ret2[i].get_vector(),  ret1[i].get_vector()) or not np.array_equal(ret2[i].get_vector(),  ret3[i].get_vector()):
    #         raise Exception()
    #     if ret1[i].get_label() != ret2[i].get_label() or ret2[i].get_label() != ret3[i].get_label():
    #         raise Exception()
    return ret3

def native_partition(images, bits, procs, n):
    p = Pool(procs)
    partial_local = partial(local_color_partition, bit=bits)
    partitions = [images[i:i + n] for i in range(0, len(images), n)]
    
    # start = time.time()
    # ret1 = [p.apply(partial_local, args=(i,)) for i in partitions]
    # end = time.time() - start
    # print "COLOR NATIVE APP: %d images -> %f" % (len(ret1), end)

    start = time.time()
    ret2 = p.map(partial_local, partitions)
    end = time.time() - start
    ret2 = list(itertools.chain.from_iterable(ret2))
    print "COLOR NATIVE MAP: %d images -> %f" % (len(ret2), end)
    
    start = time.time()
    resu = [p.apply_async(partial_local, args=(i,)) for i in partitions]
    ret3 = [r.get() for r in resu]
    end = time.time() - start
    ret3 = list(itertools.chain.from_iterable(ret3))
    print "COLOR NATIVE ASY: %d images -> %f" % (len(ret3), end)
    return ret3


def ipython_par_color(images, bits, direct):
    c = Client()   
    partial_local = partial(local_color, bit=bits)
    
    if direct:
        dview = c[:]
        dview.block = False
        num_clients = len(c.ids)

        # start = time.time()
        # ret1 = [c[i % num_clients].apply_sync(partial_local, images[i]) for i in xrange(len(images))]
        # end = time.time() - start
        # print "COLOR IPYTHON DIRECT APP: %d images -> %f" % (len(ret1), end)


        start = time.time()
        ret2 = dview.map_sync(partial_local, images)
        end = time.time() - start
        print "COLOR IPYTHON DIRECT MAP: %d images -> %f" % (len(ret2), end)

        start = time.time()
        ret = [c[i % num_clients].apply_async(partial_local, images[i]) for i in xrange(len(images))]
        ret3 = [r.get() for r in ret]
        end = time.time() - start
        print "COLOR IPYTHON DIRECT ASY: %d images -> %f" % (len(ret3), end)
        
        # for i in range(len(ret1)):
        #     if not np.array_equal(ret2[i].get_vector(),  ret1[i].get_vector()) or not np.array_equal(ret2[i].get_vector(),  ret3[i].get_vector()):
        #         raise Exception()
        #     if ret1[i].get_label() != ret2[i].get_label() or ret2[i].get_label() != ret3[i].get_label():
        #         raise Exception()
        return ret3
    else:
        dview = c.load_balanced_view()
        dview.block = False
        
        # start = time.time()
        # ret1 = [dview.apply_sync(partial_local, i) for i in images]
        # end = time.time() - start
        # print "COLOR IPYTHON LBV APP: %d images -> %f" % (len(ret1), end)

        start = time.time()
        ret2 = dview.map_sync(partial_local, images, chunksize=2)
        end = time.time() - start
        print "COLOR IPYTHON LBV MAP: %d images -> %f" % (len(ret2), end)
        
        start = time.time()
        ret = [dview.apply_async(partial_local, i) for i in images]
        ret3 = [r.get() for r in ret]
        end = time.time() - start
        print "COLOR IPYTHON LBV ASY: %d images -> %f" % (len(ret3), end)
        
        # for i in range(len(ret1)):
        #     if not np.array_equal(ret2[i].get_vector(),  ret1[i].get_vector()) or not np.array_equal(ret2[i].get_vector(),  ret3[i].get_vector()):
        #         raise Exception()
        #     if ret1[i].get_label() != ret2[i].get_label() or ret2[i].get_label() != ret3[i].get_label():
        #         raise Exception()
        return ret3

def ipython_partition(images, bits, direct, n):
    c = Client()   
    partial_local = partial(local_color_partition, bit=bits)
    partitions = [images[i:i + n] for i in range(0, len(images), n)]
    
    if direct:
        dview = c[:]
        dview.block = False
        num_clients = len(c.ids)

        # start = time.time()
        # ret1 = [c[i % num_clients].apply_sync(partial_local, partitions[i]) for i in xrange(len(partitions))]
        # end = time.time() - start
        # print "COLOR IPYTHON DIRECT APP: %d images -> %f" % (len(ret1), end)


        start = time.time()
        ret2 = dview.map_sync(partial_local, images)
        end = time.time() - start
        ret2 = list(itertools.chain.from_iterable(ret2))
        print "COLOR IPYTHON DIRECT MAP: %d images -> %f" % (len(ret2), end)

        start = time.time()
        ret = [c[i % num_clients].apply_async(partial_local, partitions[i]) for i in xrange(len(partitions))]
        ret3 = [r.get() for r in ret]
        end = time.time() - start
        ret3 = list(itertools.chain.from_iterable(ret3))
        print "COLOR IPYTHON DIRECT ASY: %d images -> %f" % (len(ret3), end)
        
        # for i in range(len(ret1)):
        #     if not np.array_equal(ret2[i].get_vector(),  ret1[i].get_vector()) or not np.array_equal(ret2[i].get_vector(),  ret3[i].get_vector()):
        #         raise Exception()
        #     if ret1[i].get_label() != ret2[i].get_label() or ret2[i].get_label() != ret3[i].get_label():
        #         raise Exception()
        return ret3
    else:
        dview = c.load_balanced_view()
        dview.block = False
        
        # start = time.time()
        # ret1 = [dview.apply_sync(partial_local, i) for i in partitions]
        # end = time.time() - start
        # print "COLOR IPYTHON LBV APP: %d images -> %f" % (len(ret1), end)

        start = time.time()
        ret2 = dview.map_sync(partial_local, partitions)
        end = time.time() - start
        ret2 = list(itertools.chain.from_iterable(ret2))
        print "COLOR IPYTHON LBV MAP: %d images -> %f" % (len(ret2), end)
        
        start = time.time()
        ret = [dview.apply_async(partial_local, i) for i in partitions]
        ret3 = [r.get() for r in ret]
        end = time.time() - start
        ret3 = list(itertools.chain.from_iterable(ret3))
        print "COLOR IPYTHON LBV ASY: %d images -> %f" % (len(ret3), end)
        
        # for i in range(len(ret1)):
        #     if not np.array_equal(ret2[i].get_vector(),  ret1[i].get_vector()) or not np.array_equal(ret2[i].get_vector(),  ret3[i].get_vector()):
        #         raise Exception()
        #     if ret1[i].get_label() != ret2[i].get_label() or ret2[i].get_label() != ret3[i].get_label():
        #         raise Exception()
        return ret3

