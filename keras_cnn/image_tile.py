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

for file in training_files:
    print file[0]
    img = read_color_image(file[0])
    rows , cols = img.shape[:2]
    count = 0
    for r in range(0, rows - 63, 64):
        for c in range(0, cols - 63, 64):
            img_cropped = img[r:r+64,c:c+64, :]
            filename = file[0][:-4] + "_" + str(count) + ".jpg"
            save_image(img_cropped, filename)
            count += 1

    os.remove(file[0])

for file in testing_files:
    print file[0]
    img = read_color_image(file[0])
    rows , cols = img.shape[:2]
    count = 0
    for r in range(0, rows - 63, 64):
        for c in range(0, cols - 63, 64):
            img_cropped = img[r:r+64,c:c+64, :]
            filename = file[0][:-4] + "_" + str(count) + ".jpg"
            save_image(img_cropped, filename)
            count += 1

    os.remove(file[0])
