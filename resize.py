import numpy as np
import cv2
import os

lst = ["Dali", "Durer", "Monet", "Picasso", "Rembrant", "VanGogh"]
num = [191, 111, 208, 192, 213, 205]

count = 0
for item in lst:
	n = num[count]
	for i in xrange(n):
		path = "%s/f%06d.jpg" % (item, i)
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
		# if h <= 500 or w <= 500:
			# img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
			# enlarge
		# else:
			# img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
			#shink
		cv2.imwrite(path, img)
	count += 1