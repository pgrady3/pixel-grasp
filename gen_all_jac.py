import datetime
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import random

from dataset_processing.image import Image, DepthImage
from dataset_processing import grasp

OUTPUT_IMG_SIZE = (300, 300)
BB_SAVE = 600 #length of bounding box padding array

TRAIN_SPLIT = 0.8
VISUALISE_ONLY = False



def get_image_ids():
    # Get all the input files, extract the numbers.
    ims = glob.glob("./data/jacquard/*/*_RGB*")

    roots = []
    for d in ims:
        roots.append(d[:-8]) #cut off the suffix filename

    random.shuffle(roots)

    return roots

def load_bbs(file_grasps):
    bounding_boxes_base = grasp.BoundingBoxes()
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

            points = np.array([[0, 0], [0, opening], [jaws_size, opening], [jaws_size, 0]])
            bb = grasp.BoundingBox(points)
            bb.offset([-0.5 * jaws_size, -0.5 * opening])#center it
            bb.rotate(-theta, [0, 0])#rotate it
            bb.offset([y, x])#offset it

            bounding_boxes_base.append(bb)

    return bounding_boxes_base

def save_subset(ids, outfile):

    len_subset = len(ids)
    rgb_arr = np.zeros((len_subset, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], 3), dtype=np.uint8)
    depth_arr = np.zeros((len_subset, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1]), dtype=np.float32)
    bbs_arr = np.zeros((len_subset, BB_SAVE, 4, 2))

    for i, img_id in enumerate(ids):
        print('%d / %d Processing: %s' % (i, len_subset, img_id))

        # Load the image
        file_img = img_id + "_RGB.png"
        file_grasps = img_id + "_grasps.txt"
        file_depth = img_id + "_perfect_depth.tiff"
        suffix = img_id.split("/")[-1]

        rgb = Image(io.imread(file_img))
        depth = io.imread(file_depth)
        depth = DepthImage(depth)

        # Load Grasps.
        bbs = load_bbs(file_grasps)
        center = bbs.center

        rgb.resize((rgb.shape[0] / 2, rgb.shape[1] / 2))
        depth.resize((depth.shape[0] / 2, depth.shape[1] / 2))
        bbs.zoom(2, [0, 0])

        left = max(0, min(center[1] - OUTPUT_IMG_SIZE[1] // 2, rgb.shape[1] - OUTPUT_IMG_SIZE[1]))
        right = min(rgb.shape[1], left + OUTPUT_IMG_SIZE[1])

        top = max(0, min(center[0] - OUTPUT_IMG_SIZE[0] // 2, rgb.shape[0] - OUTPUT_IMG_SIZE[0]))
        bottom = min(rgb.shape[0], top + OUTPUT_IMG_SIZE[0])

        rgb.crop((top, left), (bottom, right))
        depth.crop((top, left), (bottom, right))
        bbs.offset((-top, -left))

        depth.img = (depth.img - np.mean(depth.img)) / np.std(depth.img)
        depth.img = np.clip(depth.img, -2, 2)#normalizing our way
        
        bbSave = bbs.to_array(pad_to=BB_SAVE)

        rgb_arr[i, :, :, :] = rgb.img
        depth_arr[i, :, :] = depth.img
        bbs_arr[i, :, :, :] = bbSave

    np.savez(outfile, rgb_arr=rgb_arr, depth_arr=depth_arr, bbs=bbSave, bbs_arr=bbs_arr, suffix=suffix)

if __name__ == '__main__':
    all_ids = get_image_ids()
    split_point = int(len(all_ids) * TRAIN_SPLIT)
    train = all_ids[:split_point]
    test = all_ids[split_point:]

    save_subset(train, "data/jacAll/train")
    save_subset(test, "data/jacAll/test")