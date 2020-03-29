import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
import random

X_SIZE, Y_SIZE = 28, 28
DATA_PATH, DATA_X, DATA_Y = None, None, None

def load_data(dir="datasets/train", exclude_labels = []):
    dataset_path = dir

    imgs_paths = []
    cnt = 0
    subdirs = os.listdir(dataset_path)
    for subdir in subdirs:
        subdir = os.path.join(dataset_path, subdir)
        for fn in os.listdir(subdir):
            ind = int(fn.split(".", 1)[0].split("_")[-1]) - 3
            labelstr = fn.split(".", 1)[0].split("_")[-2]
            if ind < 0 or ind >= len(labelstr): continue
            if labelstr[ind] > '9' or labelstr[ind] < '0': continue
            imgs_paths.append(os.path.join(subdir, fn))

    random.seed(2333)
    random.shuffle(imgs_paths)

    imgs, labels = np.zeros((len(imgs_paths), X_SIZE, Y_SIZE), dtype=np.int16), np.zeros(len(imgs_paths), dtype=np.int8)
    print("load data, waiting...")

    if len(imgs_paths) > 500000:
        print("Use load_large_data function, test data too large")
        raise ValueError

    for i, path in tqdm(enumerate(imgs_paths)):
        fn = path.split('/').pop()
        ind = int(fn.split(".", 1)[0].split("_")[-1]) - 3
        labelstr = fn.split(".", 1)[0].split("_")[-2]
        # print(labelstr, ind, path)
        label = int(labelstr[ind])
        if label == 0: cnt += 1
        labels[i] = label
        # read figures
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = cv.resize(img, (X_SIZE, Y_SIZE))
        img = img / 255.0
        # print(img)
        imgs[i] = img

    print("finish loading data")
    print("label 0 nums: ", cnt)
    return imgs_paths, imgs, labels

def load_data_once(dir = "train"):
    global DATA_PATH, DATA_X, DATA_Y
    if DATA_PATH is None:
        print("load data once")
        DATA_PATH, DATA_X, DATA_Y = load_data(dir)
    else:
        print("load data twice")
    return DATA_PATH, DATA_X, DATA_Y

def load_infer_data(dir = "infer"):
    dataset_path = dir

    imgs_paths = []
    cnt = 0
    subdirs = os.listdir(dataset_path)
    for subdir in subdirs:
        subdir = os.path.join(dataset_path, subdir)
        for fn in os.listdir(subdir):
            if os.path.isfile(os.path.join(subdir, fn)):
                imgs_paths.append(os.path.join(subdir, fn))
            else:
                print("warning: ignore " + os.path.join(subdir, fn))

    # create img blocks
    imgs = np.zeros((len(imgs_paths), X_SIZE, Y_SIZE), dtype=np.int16)

    for i, path in tqdm(enumerate(imgs_paths)):
        # read figures
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = cv.resize(img, (X_SIZE, Y_SIZE))
        img = img / 255.0
        imgs[i] = img

    return imgs_paths, imgs, None


if __name__ == "__main__":
    imgs_paths, imgs, labels = load_data()
    print(imgs_paths)
    print(len(imgs_paths), imgs.shape, labels.shape)
