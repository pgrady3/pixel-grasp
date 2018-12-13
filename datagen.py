import numpy as np
import keras
import glob

import matplotlib.pyplot as plt
from skimage import io
import cv2
import random
import timeit

from dataset_processing.image import Image, DepthImage
from dataset_processing import grasp


OUTPUT_IMG_SIZE = (300, 300, 1)

def get_data_list(train=True):
    'Returns train/test lists of file prefixes'

    if train:
        ims = glob.glob("./data/train/*/*.npz")
    else:
        ims = glob.glob("./data/test/*/*.npz")

    return ims


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=32, train=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = get_data_list(train)
        self.n_channels = 1
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X =     np.empty((self.batch_size, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], self.n_channels))
        Y_pts = np.empty((self.batch_size, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], self.n_channels))
        Y_wid = np.empty((self.batch_size, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], self.n_channels))
        Y_sin = np.empty((self.batch_size, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], self.n_channels))
        Y_cos = np.empty((self.batch_size, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], self.n_channels))
        
        suffixes = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data = np.load(ID)

            X[i, :, :, 0] = data['depth']
            Y_pts[i, :, :, 0] = data['pos']
            Y_wid[i, :, :, 0] = np.clip(data['width'], 0, 150)/150.0
            Y_sin[i, :, :, 0] = np.sin(2*data['ang'])
            Y_cos[i, :, :, 0] = np.cos(2*data['ang'])

        return X, [Y_pts, Y_cos, Y_sin, Y_wid]

    def getTrain(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = np.empty((self.batch_size, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], self.n_channels))
        RGB = np.empty((self.batch_size, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], 3))
        bbs = []

        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data = np.load(ID)

            X[i, :, :, 0] = data['depth']
            RGB[i, :, :, :] = data['rgb']
            bbs.append(data['bbs'])

        return X, RGB.astype(np.uint8), bbs


if __name__ == "__main__":
    gen = DataGenerator(batch_size = 8, train=True)

    for i in range(len(gen)):
        print i
        x, y = gen[i]