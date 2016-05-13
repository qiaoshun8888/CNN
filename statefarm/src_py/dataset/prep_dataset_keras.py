from __future__ import print_function
from __future__ import division

import os
import glob
import pickle
import random
import time
import cv2
import math
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from skimage.io import imread, imsave
from scipy.misc import imresize

SUBSET = False
DOWNSAMPLE = 224
NUM_CLASSES = 10

# WIDTH, HEIGHT, NB_CHANNELS = 640 // DOWNSAMPLE, 480 // DOWNSAMPLE, 3
WIDTH, HEIGHT, NB_CHANNELS = 224, 224, 3

def load_image(path):
    # Load as grayscale
    if NB_CHANNELS == 1:
        img = cv2.imread(path, 0)
    elif NB_CHANNELS == 3:
        img = cv2.imread(path)
    # Reduce size
    img = cv2.resize(img, (WIDTH, HEIGHT))
    img = normalize(img)
    return img


def normalize(img):
    mean_pixel = [103.939, 116.799, 123.68]
    img = img.astype(np.float32, copy=False)
    for c in range(3):
       img[:, :, c] = img[:, :, c] - mean_pixel[c]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def get_driver_data():
    drivers = dict()
    classes = dict()
    print('Read drivers data')
    f = open('driver_imgs_list.csv', 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        array = line.strip().split(',')
        drivers[array[2]] = array[0]
        if array[0] not in classes.keys():
            classes[array[0]] = [(array[1], array[2])]
        else:
            classes[array[0]].append((array[1], array[2]))
    f.close()
    return drivers, classes

def load_train(base):
    X_train = []
    X_train_id = []
    y_train = []
    driver_ids = []
    driver_data, driver_class = get_driver_data()
    start_time = time.time()

    print('Reading train images...')
    for j in range(NUM_CLASSES):
        print('Loading folder c{}...'.format(j))
        path = os.path.join(base, 'c{}/'.format(j), '*.jpg')
        files = glob.glob(path)
        for file in files:
            flbase = os.path.basename(file)
            img = load_image(file)
            # img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(j)
            driver_ids.append(driver_data[flbase])

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    unique_drivers = sorted(list(set(driver_ids)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, X_train_id, driver_ids, unique_drivers


def load_test(base):
    print('Read test images')
    path = os.path.join(base, '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/NUM_CLASSES)
    for file in files:
        flbase = os.path.basename(file)
        img = load_image(file)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id


X_train, y_train, X_train_id, driver_ids, unique_drivers = load_train('imgs/train/')
X_test, X_test_ids = load_test('imgs/test/')

if SUBSET:
    dest = 'data_{}_subset_keras.pkl'.format(DOWNSAMPLE)
else:
    dest = 'data_{}_keras.pkl'.format(DOWNSAMPLE)

with open(dest, 'wb') as f:
    pickle.dump((X_train, y_train, X_test, X_test_ids, driver_ids), f)
