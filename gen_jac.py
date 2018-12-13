import datetime
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import cv2
import random

from joblib import Parallel, delayed

from dataset_processing.image import Image, DepthImage
from dataset_processing import grasp


OUTPUT_TEST_DIR = 'data/test/'
OUTPUT_TRAIN_DIR = 'data/train/'
OUTPUT_IMG_SIZE = (300, 300)
RANDOM_ROTATIONS = 1
RANDOM_ZOOM = False

TRAIN_SPLIT = 0.8
# OR specify which images are in the test set.
VISUALISE_ONLY = False

def get_image_ids():
    # Get all the input files, extract the numbers.
    ims = glob.glob("./data/jacquard/*/*_RGB*")

    roots = []
    for d in ims:
        roots.append(d[:-8]) #cut off the suffix filename

    random.shuffle(roots)

    return roots

def preprocessFiles(all_ids):
    for idx, img_id in enumerate(all_ids):
        print idx, "/", len(all_ids), img_id

        suffix = img_id.split("/")[-1]

        # Decide whether this is train or test.
        output_folder = OUTPUT_TRAIN_DIR + suffix
        if np.random.rand() > TRAIN_SPLIT:
            output_folder = OUTPUT_TEST_DIR + suffix

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


        # Load the image
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

        rgb_img_base.resize((rgb_img_base.shape[0] / 2, rgb_img_base.shape[1] / 2))
        depth_img_base.resize((depth_img_base.shape[0] / 2, depth_img_base.shape[1] / 2))
        bounding_boxes_base.zoom(2, [0, 0])

        center = bounding_boxes_base.center
        depth = depth_img_base
        rgb = rgb_img_base
        bbs = bounding_boxes_base.copy()

        left = max(0, min(center[1] - OUTPUT_IMG_SIZE[1] // 2, rgb.shape[1] - OUTPUT_IMG_SIZE[1]))
        right = min(rgb.shape[1], left + OUTPUT_IMG_SIZE[1])

        top = max(0, min(center[0] - OUTPUT_IMG_SIZE[0] // 2, rgb.shape[0] - OUTPUT_IMG_SIZE[0]))
        bottom = min(rgb.shape[0], top + OUTPUT_IMG_SIZE[0])

        rgb.crop((top, left), (bottom, right))
        depth.crop((top, left), (bottom, right))
        bbs.offset((-top, -left))

        depth.img = (depth.img - np.mean(depth.img)) / np.std(depth.img)
        depth.img = np.clip(depth.img, -2, 2)#normalizing our way

        pos_img, ang_img, width_img = bbs.draw(depth.shape)

        if VISUALISE_ONLY:
            print img_id
            f = plt.figure()
            ax = f.add_subplot(2, 2, 1)
            rgb.show(ax)
            #rgb_img_base.show(ax)
            #bbs.show(ax)
            ax = f.add_subplot(2, 2, 2)
            depth.show(ax)
            bbs.show(ax)

            #ax = f.add_subplot(1, 5, 3)
            #ax.imshow(pos_img)

            ax = f.add_subplot(2, 2, 3)
            ax.imshow(ang_img)

            ax = f.add_subplot(2, 2, 4)
            ax.imshow(width_img)

            #mng = plt.get_current_fig_manager()
            #mng.resize(*mng.window.maxsize())

            plt.show()
            continue

        i = 1
        
        bbSave = bbs.to_array(pad_to=150)

        #io.imsave(output_folder + "/" + str(i) + "_RGB.png", rgb.img)
        np.savez(output_folder + "/" + str(i) + "_data", depth=depth.img, pos=pos_img, 
            ang=ang_img, width=width_img, bbs=bbSave, suffix=suffix, rgb=rgb.img)

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

if __name__ == '__main__':
    # Create the output directory
    if not os.path.exists(OUTPUT_TEST_DIR):
        os.makedirs(OUTPUT_TEST_DIR)
    if not os.path.exists(OUTPUT_TRAIN_DIR):
        os.makedirs(OUTPUT_TRAIN_DIR)

    all_ids = get_image_ids()
    preprocessFiles(all_ids)
    
    '''PROCESSES = 4
    id_chunks = chunkIt(all_ids, PROCESSES)

    Parallel(n_jobs=PROCESSES, verbose=50)(delayed(preprocessFiles)(i)for i in id_chunks)'''