from __future__ import print_function
from __future__ import division

import os
import time
import pickle
import sys
import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

from sklearn.metrics import log_loss
from sklearn.cross_validation import LabelShuffleSplit

from utilities import write_submission, calc_geom, calc_geom_arr, mkdirp


sys.path.insert(0, '/Users/sqiao/Desktop/kaggle/statefarm/src_py/dataset')
from prep_dataset_keras import load_train, load_test

TESTING = False
USING_CHECKPOINT = False

DOWNSAMPLE = 224
NB_EPOCHS = 50 if not TESTING else 10
MAX_FOLDS = 8 if not TESTING else 2

DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/data_%d_keras.pkl' % DOWNSAMPLE if not TESTING else 'dataset/data_%d_subset_keras.pkl' % DOWNSAMPLE)

CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
SUMMARY_PATH = os.environ.get('SUMMARY_PATH', 'summaries/')
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/')

mkdirp(CHECKPOINT_PATH)
mkdirp(SUMMARY_PATH)
mkdirp(MODEL_PATH)

# WIDTH, HEIGHT, NB_CHANNELS = 640 // DOWNSAMPLE, 480 // DOWNSAMPLE, 3
WIDTH, HEIGHT, NB_CHANNELS = 224, 224, 3
NUM_CLASSES = 10
BATCH_SIZE = 64
PATIENCE = 5

predictions_total = [] # accumulated predictions from each fold
scores_total = [] # accumulated scores from each fold
num_folds = 0

def vgg_bn():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights('models/vgg16_weights.h5')


    # Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def read_data(path, using_cache=False):
    if using_cache:
        with open(path, 'rb') as f:
            X_train_raw, y_train_raw, X_test, X_test_ids, driver_ids = pickle.load(f)
        _, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)
    else:
        X_train_raw, y_train_raw, _, driver_ids, unique_drivers = load_train('dataset/imgs/train/')  ## _: X_train_id
        X_test, X_test_ids = load_test('dataset/imgs/test/')

    train_data = np.array(X_train_raw, dtype=np.uint8)
    train_target = np.array(y_train_raw, dtype=np.uint8)
    test_data = np.array(X_test, dtype=np.uint8)

    if NB_CHANNELS == 1:
        train_data = train_data.reshape(train_data.shape[0], NB_CHANNELS, HEIGHT, WIDTH)
        test_data = test_data.reshape(test_data.shape[0], NB_CHANNELS, HEIGHT, WIDTH)
    else:
        train_data = train_data.transpose((0, 3, 1, 2))
        test_data = test_data.transpose((0, 3, 1, 2))

    train_target = np_utils.to_categorical(train_target, 10)
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return train_data, train_target, driver_ids, unique_drivers, driver_indices, test_data, X_test_ids

def run_cross_validation():
    train_data, train_target, driver_ids, unique_drivers, driver_indices, test_data, X_test_ids = read_data(DATASET_PATH)

    for train_index, valid_index in LabelShuffleSplit(driver_indices, n_iter=MAX_FOLDS, test_size=0.2, random_state=67):
        print('Fold {}/{}'.format(num_folds + 1, MAX_FOLDS))

        # skip fold if a checkpoint exists for the next one
        # next_checkpoint_path = os.path.join(CHECKPOINT_PATH, 'model_{}.h5'.format(num_folds + 1))
        # if os.path.exists(next_checkpoint_path):
        #     print('Checkpoint exists for next fold, skipping current fold.')
        #     continue

        X_train, y_train = train_data[train_index,...], train_target[train_index,...]
        X_valid, y_valid = train_data[valid_index,...], train_target[valid_index,...]

        model = vgg_bn()

        model_path = os.path.join(MODEL_PATH, 'model_{}.json'.format(num_folds))
        with open(model_path, 'w') as f:
            f.write(model.to_json())

        # restore existing checkpoint, if it exists
        checkpoint_path = os.path.join(CHECKPOINT_PATH, 'model_{}.h5'.format(num_folds))
        if USING_CHECKPOINT and os.path.exists(checkpoint_path):
            print('Restoring fold from checkpoint.')
            model.load_weights(checkpoint_path)

        summary_path = os.path.join(SUMMARY_PATH, 'model_{}'.format(num_folds))
        mkdirp(summary_path)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, mode='auto'),
            ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            # TensorBoard(log_dir=summary_path, histogram_freq=0)
        ]
        model.fit(
            X_train, y_train, 
            batch_size=BATCH_SIZE, 
            nb_epoch=NB_EPOCHS, 
            shuffle=True, 
            verbose=1, 
            validation_data=(X_valid, y_valid), 
            callbacks=callbacks
        )

        predictions_valid = model.predict(X_valid, batch_size=BATCH_SIZE * 2, verbose=1)
        score_valid = log_loss(y_valid, predictions_valid)
        scores_total.append(score_valid)

        print('Score: {}'.format(score_valid))

        predictions_test = model.predict(test_data, batch_size=BATCH_SIZE * 2, verbose=1)
        predictions_total.append(predictions_test)

        num_folds += 1

    score_geom = calc_geom(scores_total, MAX_FOLDS)
    predictions_geom = calc_geom_arr(predictions_total, MAX_FOLDS)

    print('Writing submission for {} folds, score: {}...'.format(num_folds, score_geom))
    submission_path = os.path.join(SUMMARY_PATH, 'submission_{}_{:.2}.csv'.format(int(time.time()), score_geom))
    write_submission(predictions_geom, X_test_ids, submission_path)

    print('Done.')


def main():
    run_cross_validation()


main()






