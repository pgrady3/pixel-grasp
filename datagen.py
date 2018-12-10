import numpy as np
import keras
import glob

import matplotlib.pyplot as plt
from skimage import io
import cv2
import random

from dataset_processing.image import Image, DepthImage
from dataset_processing import grasp

TRAIN_SPLIT = 0.8
RANDOM_ZOOM = True
OUTPUT_IMG_SIZE = (300, 300, 1)

def get_data_list():
    'Returns train/test lists of file prefixes'

    ims = glob.glob("./data/jacquard/*/*_RGB*")
    roots = []
    for d in ims:
        roots.append(d[:-8]) #cut off the suffix filename

    #random.shuffle(roots)

    splitIdx = int(len(roots) * TRAIN_SPLIT)

    train = roots[:splitIdx]
    test = roots[splitIdx:]

    return train, test



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = 1
        self.shuffle = shuffle
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
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], self.n_channels))
        Y_pts = np.empty((self.batch_size, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], self.n_channels))
        Y_wid = np.empty((self.batch_size, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], self.n_channels))
        Y_sin = np.empty((self.batch_size, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], self.n_channels))
        Y_cos = np.empty((self.batch_size, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            depth, pts, ang, wid = self.gen_single_image(ID)

            X[i, :, :, 0] = depth
            Y_pts[i, :, :, 0] = pts
            Y_wid[i, :, :, 0] = wid
            Y_sin[i, :, :, 0] = np.sin(2*ang)
            Y_cos[i, :, :, 0] = np.cos(2*ang)

        return X, [Y_pts, Y_cos, Y_sin, Y_wid]

    def gen_single_image(self, img_id):
        'Generate ground truth from a single image'

        file_img = img_id + "_RGB.png"
        file_grasps = img_id + "_grasps.txt"
        file_depth = img_id + "_perfect_depth.tiff"

        rgb_img_base = Image(io.imread(file_img))
        depth_img_base = io.imread(file_depth)
        depth_img_base = depth_img_base - depth_img_base.min()
        depth_img_base = depth_img_base / depth_img_base.max()#normalize to 1
        depth_img_base = DepthImage(depth_img_base)

        # Load Grasps.
        bounding_boxes_base = grasp.BoundingBoxes()
        bbDict = dict()
        with open(file_grasps) as f:
            while True:
                line = f.readline().rstrip()
                if not line:    #hit EOF
                    break

                split = line.split(";")
                x = float(split[0])
                y = float(split[1])
                theta = float(split[2]) / 180.0 * np.pi
                opening = float(split[3])   #opening of the jaws. useful
                jaws_size = float(split[4]) #width of the jaws. not that useful

                key = (x, y, theta, opening)
                if key in bbDict:
                    if jaws_size > bbDict[key]:#just keep the biggest jaws size
                        bbDict[key] = jaws_size
                else:
                    bbDict[key] = jaws_size

        for bb in bbDict:
            x, y, theta, opening = bb
            jaws_size = bbDict[bb]

            points = np.array([[0, 0], [0, opening], [jaws_size, opening], [jaws_size, 0]])
            bb = grasp.BoundingBox(points)
            bb.offset([-0.5 * jaws_size, -0.5 * opening])#center it
            bb.rotate(-theta, [0, 0])#rotate it
            bb.offset([y, x])#offset it

            bounding_boxes_base.append(bb)

        rgb_img_base.resize((rgb_img_base.shape[0] / 2, rgb_img_base.shape[1] / 2))#cut size in HALF!!!
        depth_img_base.resize((depth_img_base.shape[0] / 2, depth_img_base.shape[1] / 2))
        bounding_boxes_base.zoom(2, [0, 0])

        center = bounding_boxes_base.center

        #rotate.....
        angle = np.random.random() * 2 * np.pi - np.pi
        rgb = rgb_img_base.rotated(angle, center)
        depth = depth_img_base.rotated(angle, center)

        bbs = bounding_boxes_base.copy()
        bbs.rotate(angle, center)

        left = max(0, min(center[1] - OUTPUT_IMG_SIZE[1] // 2, rgb.shape[1] - OUTPUT_IMG_SIZE[1]))
        right = min(rgb.shape[1], left + OUTPUT_IMG_SIZE[1])

        top = max(0, min(center[0] - OUTPUT_IMG_SIZE[0] // 2, rgb.shape[0] - OUTPUT_IMG_SIZE[0]))
        bottom = min(rgb.shape[0], top + OUTPUT_IMG_SIZE[0])

        rgb.crop((top, left), (bottom, right))
        depth.crop((top, left), (bottom, right))
        bbs.offset((-top, -left))

        if RANDOM_ZOOM:
            zoom_factor = np.random.uniform(0.8, 1.0)
            rgb.zoom(zoom_factor)
            depth.zoom(zoom_factor)
            bbs.zoom(zoom_factor, (OUTPUT_IMG_SIZE[0]//2, OUTPUT_IMG_SIZE[1]//2))

        #depth.normalise()
        depth.img = (depth.img - np.mean(depth.img)) / np.std(depth.img)
        depth.img = np.clip(depth.img, -2, 2)#normalizing our way

        pos_img, ang_img, width_img = bbs.draw(depth.shape)

        #ds['img_id'].append(img_id)
        #ds['rgb'].append(rgb.img)
        #ds['depth_inpainted'].append(depth.img)
        #ds['bounding_boxes'].append(bbs.to_array(pad_to=200))#if the list is variable length, hd5 will crash
        #ds['grasp_points_img'].append(pos_img)
        #ds['angle_img'].append(ang_img)
        #ds['grasp_width'].append(width_img)

        width_img = np.clip(width_img, 0, 150)/150.0

        return depth.img, pos_img, ang_img, width_img
