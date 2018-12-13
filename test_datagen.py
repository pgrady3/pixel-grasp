import numpy as np
import keras
import glob

import matplotlib.pyplot as plt
from skimage import io
import cv2
import random

from dataset_processing.image import Image, DepthImage
from dataset_processing import grasp
import datagen

'''train, test = datagen.get_data_list()

gen = datagen.DataGenerator(train)

batches = len(gen)

for idx, img_id in enumerate(train):

    if idx % 100 == 0:
        print idx, len(train)

    file_img = img_id + "_RGB.png"
    rgb_img_base = Image(io.imread(file_img))
    #im = keras.preprocessing.image.load_img(file_img)
    #im = cv2.imread(file_img)'''


dataList = glob.glob("./data/train/*/*.npz")

for idx, img_id in enumerate(dataList):
    if idx % 100 == 0:
        print idx, len(dataList)

    data = np.load(img_id)
    #print data.files