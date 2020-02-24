import os
import numpy as np
import cv2 as cv
from tqdm import tqdm

DATASET_DIR = "datasets"

X_SIZE, Y_SIZE = 28, 28
DATA_PATH, DATA_X, DATA_Y = None, None, None

def load_data(dir = "train"):
    dataset_path = os.path.join(DATASET_DIR, dir)

    fns = os.listdir(dataset_path)
    cnt = 0
    # fns = fns[:100]
    imgs_paths, imgs, labels = [], np.zeros((len(fns), X_SIZE, Y_SIZE), dtype=np.int16), np.zeros(len(fns), dtype=np.int8)
    print("load data, waiting...")
    if len(fns) > 500000:
        print("Use load_large_data function, test data too large")
        raise ValueError
    for i, fn in tqdm(enumerate(fns)):
        # parse label
        labelstr = fn.split(".", 1)[0].split("_")[-1]
        # try:
        #     if int(labelstr) == 0: continue
        # except ValueError:
        #     print(fn)
        if int(labelstr) == 0: cnt += 1
        labels[i] = int(labelstr)
        # save figure path
        image_path = os.path.join(dataset_path, fn)
        imgs_paths.append(image_path)
        # read figures
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        # FIXME:
        # print(fn, img.shape)
        # print(img)
        img = cv.resize(img, (X_SIZE, Y_SIZE))
        # cv.imshow("test", img)
        # cv.waitKey(0)
        img = img / 255.0
        # print(img)
        imgs[i] = img
        # imgs = np.append(imgs, [img], axis=0)
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


if __name__ == "__main__":
    imgs_paths, imgs, labels = load_data()
    print(imgs_paths)
    print(len(imgs_paths), imgs.shape, labels.shape)
