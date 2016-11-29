import numpy as np
from project_types import Instance
from images import *
import gc

def raster(images):

    # img = read_color_image(filename)
    instances = []
    for img, label in images:
        img = read_color_image(img)
        rows, cols, channels = img.shape
        instances.append(Instance(np.flatten(img), label))
    return instances

def color_histogram(images, bits = 12):
    instances = []
    for img, label in images:
        histogram = np.zeros(2 ** bits)
        img = read_color_image(img)
        rows, cols = img.shape[:2]
        for r in range(rows):
            for c in range(cols):
                blue = img.item((r,c,0))
                green = img.item((r,c,1))
                red = img.item((r,c,2))

                encoding = 0
                encoding += red / (2 ** (bits / 3))
                encoding << bits / 3;
                encoding += green / (2 ** (bits / 3))
                encoding << bits / 3;
                encoding += blue / (2 ** (bits / 3))

                histogram[encoding] += 1
        instances.append(Instance(histogram, label))
    return instances



    def bow(images):
        
        instances = []
        cv2.ocl.setUseOpenCL(False)
        orb = cv2.ORB_create()

        for img, label in images:
            img = read_color_image(img)
            keypoints = orb.detect(img, None)
            keypoints, descriptors = orb.compute(img, keypoints)


        return instances