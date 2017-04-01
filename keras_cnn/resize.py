from images import *
import cv2
import os

def get_files(folder_name):

    foldersTmp = os.listdir(folder_name)
    folders = []
    for folder in foldersTmp:
        if folder[0] == '.':
            continue
        folders.append(folder)

    imgs = []
    for folder in folders:
        path = folder_name + folder + '/'
        if not os.path.isdir(path):
            continue
            
        files = os.listdir(path)
        for file_str in files:
            complete_file_str = str((os.path.join(path, file_str)))
            if os.path.isfile(complete_file_str) and (complete_file_str.endswith('.jpg') or complete_file_str.endswith('.JPG')):
                imgs.append((os.path.join(path, file_str), folder))

    return imgs

training_files = get_files('data/train/')
testing_files = get_files('data/test/')

for folder in (training_files, testing_files):
    for file in folder:
        # print file[0]
        img = read_color_image(file[0])
        max_size = max(img.shape[0], img.shape[1])
        if max_size != 512:
            h, w = img.shape[:2]

            height = None
            width = None
            interpolation = None
            if w > h:
                if w > 512:
                    interpolation = cv2.INTER_AREA
                else:
                    interpolation = cv2.INTER_CUBIC

                width = 512
                scale = float(h)/float(w)
                height = int(512 * scale)
            else:
                if h > 512:
                    interpolation = cv2.INTER_AREA
                else:
                    interpolation = cv2.INTER_CUBIC

                height = 512
                scale = float(w)/float(h)
                width = int(height * scale)

            img = cv2.resize(img, (width, height), interpolation = interpolation)

            print file[0]
            # save_image(img, file[0])


