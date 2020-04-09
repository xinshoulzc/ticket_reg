import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
from enum import Enum
import random
from logger import logger


X_SIZE, Y_SIZE = 28, 28
DATA_PATH, DATA_X, DATA_Y = None, None, None

TRAIN_MODE = Enum('MODE', ('NORMAL', 'PRICE', 'PRINT_DATASET', 'BARCODE'))

def load_train_data(dirs=["datasets/train"], train_modes=[TRAIN_MODE.PRICE]):
    img_paths_with_mode = load_img_paths(dirs, train_modes)

    random.seed(2333)
    random.shuffle(img_paths_with_mode)

    imgs, labels = np.zeros((len(img_paths_with_mode), X_SIZE, Y_SIZE), dtype=np.int16), np.zeros(len(img_paths_with_mode), dtype=np.int8)

    # label zero count
    cnt = 0
    logger.info("load data {} lines, waiting...".format(len(img_paths_with_mode)))

    if len(img_paths_with_mode) > 500000:
        print("Use load_large_data function, test data too large")
        raise ValueError

    for i, path_mode in tqdm(enumerate(img_paths_with_mode)):
        path, mode = path_mode
        fn = path.split('/').pop()
        label = parse_fn(fn, mode)
        if label == 0: cnt += 1

        labels[i] = label
        imgs[i] = img_process(path, mode)

    logger.info("finish loading data")
    # logger.info("label 0 nums: {:d}".format(cnt))
    imgs_paths = [path for path, mode in img_paths_with_mode]
    return imgs_paths, imgs, labels

def isbadImg(filename):
    bads = ["03304_102000", "00945_144000", "00639_95000", "01681_79000", "00752_138000", "03329_13900"]
    for bad in bads:
        if bad in filename: return True
    return False

def load_img_paths(dirs=[], train_modes=[]):
    assert len(dirs) == len(train_modes)
    img_paths_with_mode = []
    for i, dataset_path in enumerate(dirs):
        subdirs = os.listdir(dataset_path)
        for subdir in subdirs:
            subdir = os.path.join(dataset_path, subdir)
            for fn in os.listdir(subdir):
                if isbadImg(fn): continue
                label = parse_fn(fn, train_modes[i])
                if label == -1: continue
                img_paths_with_mode.append((os.path.join(subdir, fn), train_modes[i]))

    return img_paths_with_mode

def load_infer_data(dir = "infer"):
    dataset_path = dir
    imgs_paths = []
    cnt = 0
    for fn in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, fn)):
            imgs_paths.append(os.path.join(dir, fn))
        else:
            print("warning: ignore " + os.path.join(dir, fn))

    # create img blocks
    imgs = np.zeros((len(imgs_paths), X_SIZE, Y_SIZE), dtype=np.int16)

    for i, path in tqdm(enumerate(imgs_paths)):
        # read figures
        imgs[i] = img_process(path, TRAIN_MODE.NORMAL)

    return imgs_paths, imgs, None

def img_process(fp: str, train_mode: str) -> (np.array, str):
    img = cv.imread(fp)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    if train_mode == TRAIN_MODE.PRICE:
        pass
    elif train_mode == TRAIN_MODE.PRINT_DATASET:
        img = 255 - img
    elif train_mode == TRAIN_MODE.BARCODE:
        pass
    else:
        pass
    # padding
    if img.shape[0] > img.shape[1]:
        pad = (img.shape[0] - img.shape[1]) // 2
        img = cv.copyMakeBorder(img,0,0,pad,pad,cv.BORDER_CONSTANT,value=0)
    elif img.shape[1] > img.shape[0]:
        pad = (img.shape[1] - img.shape[0]) // 2
        img = cv.copyMakeBorder(img,pad,pad,0,0,cv.BORDER_CONSTANT,value=0)

    img = cv.resize(img, (X_SIZE, Y_SIZE))
    img = img / 255.0
    return img

def parse_fn(fn: str, train_mode: str) -> int:
    if "png" not in fn:
        logger.warning("ignore file: ", fn)
        return -1
    if train_mode == TRAIN_MODE.PRICE:
        ind = int(fn.split(".", 1)[0].split("_")[-1]) - 3
        labelstr = fn.split(".", 1)[0].split("_")[-2]
        if ind < 0 or ind >= len(labelstr): return -1
        if labelstr[ind] > '9' or labelstr[ind] < '0': return -1
        return int(labelstr[ind])
    elif train_mode == TRAIN_MODE.PRINT_DATASET:
        fn = fn.split('-')[-2][-2:]
        return int(fn) - 1
    elif train_mode == TRAIN_MODE.BARCODE:
        ind = int(fn.split(".", 1)[0].split("_")[-1])
        labelstr = fn.split(".", 1)[0].split("_")[-2]
        if ind < 0 or ind >= len(labelstr): return -1
        if labelstr[ind] > '9' or labelstr[ind] < '0': return -1
        return int(labelstr[ind])
    else:
        raise ValueError("UNKNOWN TRAIN_MODE DON'T HAVE LABEL")

if __name__ == "__main__":
    imgs_paths, imgs, labels = load_train_data(
                                ["datasets/train", "datasets/english"], [TRAIN_MODE.PRICE, TRAIN_MODE.PRINT_DATASET])
    print(len(imgs_paths), imgs.shape, labels.shape)
    imgs_paths, imgs, labels = load_train_data(["datasets/train"], [TRAIN_MODE.PRICE])
    print(len(imgs_paths), imgs.shape, labels.shape)
    imgs_paths, imgs, _ = load_infer_data("datasets/debug")
    print(len(imgs_paths), imgs.shape)
