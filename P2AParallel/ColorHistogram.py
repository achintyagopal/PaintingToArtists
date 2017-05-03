from images import *
from project_types import FeatureConverter, Instance
import cv2
import numpy as np
import time
from multiprocessing import Pool
# import threading
# import Queue
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
        print "COLOR SERIAL: %d images -> %d" % (len(images), end)

    def __createInstances(self, image):
        
        img, label = image
        histogram = np.zeros(2 ** self.bits)
        img = read_color_image(img)
        rows, cols = img.shape[:2]
        for r in range(rows):
            for c in range(cols):
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
    for r in range(rows):
        for c in range(cols):
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


def native_par_color(images, bits, procs):
    start = time.time()
    p = Pool(procs)
    partial_local = partial(local_color, bit=bits)
    ret = p.map(partial_local, images)
    end = time.time() - start
    print "COLOR NATIVE: %d images -> %d" % (len(images), end)
    return ret


def ipython_par_color(images, bits):
    start = time.time()
    c = Client()
    dview = c[:]
    partial_local = partial(local_color, bit=bits)
    ret = dview.map_sync(partial_local, images)
    end = time.time() - start
    print "COLOR IPYTHON: %d images -> %d" % (len(images), end)
    return ret





    # partial_local = partial(localThr, bit=bits)
    # threads = []
    # q = Queue.Queue()
    # start = time.time()
    
    # for i in range(len(images)):
    #     t = threading.Thread(target=partial_local, args=(images[i],q,))
    #     threads.append(t)
    #     t.start()

    # for i in threads:
    #     i.join()    
    # print "JLTH", time.time() - start
    # print q.qsize()

    # c = Client()
    # # print c.ids
    # dview = c[:]
    # # dview.push(HOGDESC)
    # # with dview.sync_imports():
    #     # import sys
    #     # sys.path[:] = ["/home/bill/Desktop/PaintingToArtists/P2AParallel"]
    # # print "LOC", sys.path
    # # print dview.map_sync(par, range(1))
    # partial_local = partial(local, bit=bits)
    # start = time.time()
    # ret = dview.map_sync(partial_local, images)
    # print "DIRECT:", time.time() - start
    # dview = c.load_balanced_view()
    # start = time.time()
    # ret = dview.map_sync(partial_local, images)
    # print "LB", time.time() - start

    # print ret
    # return list(q.queue)

# def localThr(image, q, bit):
#     img, label = image
#     # print img
#     histogram = np.zeros(2 ** bit)
#     img = read_color_image(img)
#     rows, cols = img.shape[:2]
#     count = 0
#     for r in range(rows):
#         for c in range(cols):
#             blue = img.item((r,c,0))
#             green = img.item((r,c,1))
#             red = img.item((r,c,2))

#             encoding = 0
#             encoding += int(red * (2 ** (bit / 3.0)) /256)
#             encoding <<= int(bit / 3.0)
#             encoding += int(green * (2 ** (bit / 3.0)) /256)
#             encoding <<= int(bit / 3.0)
#             encoding += int(blue * (2 ** (bit / 3.0)) /256)

#             histogram[encoding] += 1

#     q.put(Instance(histogram, label))
