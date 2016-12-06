import numpy as np
import cv2
import os

lst = ["Dali", "Durer", "Monet", "Picasso", "Rembrant", "VanGogh"]

for item in lst:
	for filename in os.listdir(item):
		if filename.endswith(".jpg") or filename.endswith('.JPG'):

			path = item + "/" + filename
			print path
			
			img = cv2.imread(path)
			h, w = img.shape[:2]

			height = None
			width = None
			interpolation = None
			if w > h:
				if w > 500:
					interpolation = cv2.INTER_AREA
				else:
					interpolation = cv2.INTER_CUBIC

				width = 500
				scale = float(h)/float(w)
				height = int(500 * scale)
			else:
				if h > 500:
					interpolation = cv2.INTER_AREA
				else:
					interpolation = cv2.INTER_CUBIC

				height = 500
				scale = float(w)/float(h)
				width = int(height * scale)

			img = cv2.resize(img, (width, height), interpolation = interpolation)
			cv2.imwrite(path, img)

