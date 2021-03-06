from project_types import FeatureConverter, Instance
import cv2
from images import *
import numpy as np
from multiprocessing import Pool
# from ipyparallel import Client
from functools import partial
import pickle

class BagOfWords(FeatureConverter):

    def __init__(self, center_num = 800):
        self.training_instances = None
        self.testing_instances = None
        self.center_num = center_num

    def createTrainingInstances(self, images):
        instances = []
        img_descriptors = []
        master_descriptors = []
        cv2.ocl.setUseOpenCL(False)
        orb = cv2.ORB_create()
        for img, label in images:
            print img
            img = read_color_image(img)
            keypoints = orb.detect(img, None)
            keypoints, descriptors = orb.compute(img, keypoints)
            if descriptors is None:
            	descriptors = []

            img_descriptors.append(descriptors)
            for i in descriptors:
            	master_descriptors.append(i)
        

        master_descriptors = np.float32(master_descriptors)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        ret, labels, centers = cv2.kmeans(master_descriptors, self.center_num, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.ravel()

        count = 0
        img_num = 0
        for img, label in images:
            histogram = np.zeros(self.center_num)
            feature_vector = img_descriptors[img_num]
            for f in xrange(len(feature_vector)):
                index = count + f
                histogram.itemset(labels[index], 1 + histogram.item(labels[index]))
            count += len(feature_vector)
            pairing = Instance(histogram, label)
            instances.append(pairing)

        self.training_instances = instances
        self.centers = centers


    def createTestingInstances(self, images):

        cv2.ocl.setUseOpenCL(False)
        orb = cv2.ORB_create()
        instances = []

        for img, label in images:
            print img
            img = read_color_image(img)
            
            keypoints = orb.detect(img, None)
            keypoints, descriptors = orb.compute(img, keypoints)

            if descriptors is None:
            	descriptors = []

            histogram = np.zeros(self.center_num)
            for d in descriptors:
                min_val = None
                min_index = None
                for j in xrange(len(self.centers)):
                    distance = np.linalg.norm(d - self.centers[j])
                    if min_val is None or distance < min_val:
                        min_val = distance
                        min_index = j
                histogram.itemset(min_index, 1 + histogram.item(min_index))
            instances.append(Instance(histogram, label))

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

    def setTrainingInstances(self, inst):
        self.training_instances = inst

    def setTestingInstances(self, inst):
        self.testing_instances = inst


def local_bow_train(image):
        instances = []
        img_descriptors = []
        master_descriptors = []
        cv2.ocl.setUseOpenCL(False)
        orb = cv2.ORB_create()
        for img, label in images:
            print img
            img = read_color_image(img)
            keypoints = orb.detect(img, None)
            keypoints, descriptors = orb.compute(img, keypoints)
            if descriptors is None:
                descriptors = []

            img_descriptors.append(descriptors)
            for i in descriptors:
                master_descriptors.append(i)
        

        master_descriptors = np.float32(master_descriptors)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        ret, labels, centers = cv2.kmeans(master_descriptors, self.center_num, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.ravel()

        count = 0
        img_num = 0
        for img, label in images:
            histogram = np.zeros(self.center_num)
            feature_vector = img_descriptors[img_num]
            for f in xrange(len(feature_vector)):
                index = count + f
                histogram.itemset(labels[index], 1 + histogram.item(labels[index]))
            count += len(feature_vector)
            pairing = Instance(histogram, label)
            instances.append(pairing)

        self.training_instances = instances
        self.centers = centers