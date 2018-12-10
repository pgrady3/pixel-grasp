import datetime
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import cv2
import random

from dataset_processing.image import Image, DepthImage
from dataset_processing import grasp


DATASET_NAME = 'jacquard'
OUTPUT_DIR = 'data/datasets'
RAW_DATA_DIR = 'data/jacquard'
OUTPUT_IMG_SIZE = (300, 300)
RANDOM_ROTATIONS = 3
RANDOM_ZOOM = True

TRAIN_SPLIT = 0.8
# OR specify which images are in the test set.
TEST_IMAGES = None
VISUALISE_ONLY = False

# File name patterns for the different file types.  _ % '<image_id>'
#_rgb_pattern = os.path.join(RAW_DATA_DIR, 'pcd%sr.png')
#_pcd_pattern = os.path.join(RAW_DATA_DIR, 'pcd%s.txt')
#_pos_grasp_pattern = os.path.join(RAW_DATA_DIR, 'pcd%scpos.txt')
#_neg_grasp_pattern = os.path.join(RAW_DATA_DIR, 'pcd%scneg.txt')


def get_image_ids():
    # Get all the input files, extract the numbers.

    ims = glob.glob("./data/jacquard/*/*_RGB*")

    roots = []
    for d in ims:
        roots.append(d[:-8]) #cut off the suffix filename

    random.shuffle(roots)

    return roots


if __name__ == '__main__':
    # Create the output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Label the output file with the date/time it was created
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    outfile_name = os.path.join(OUTPUT_DIR, '%s_%s.hdf5' % (DATASET_NAME, dt))

    fields = [
        'img_id',
        'rgb',
        'depth_inpainted',
        'bounding_boxes',
        'grasp_points_img',
        'angle_img',
        'grasp_width'
    ]

    # Empty datatset.
    dataset = {
        'test':  dict([(f, []) for f in fields]),
        'train': dict([(f, []) for f in fields])
    }

    temp = ["./data/jacquard/5c832b4698e78caeb8b2226f41012fcd/4_5c832b4698e78caeb8b2226f41012fcd", "./data/jacquard/7cc4096fb7eb6abc6e71174964d90e49/0_7cc4096fb7eb6abc6e71174964d90e49"]

    all_ids = get_image_ids()

    for idx, img_id in enumerate(all_ids):
    #for img_id in temp:
        print idx, "/", len(all_ids), img_id

        # Decide whether this is train or test.
        ds_output = 'train'
        if TEST_IMAGES:
            if int(img_id) in TEST_IMAGES:
                print("This image is in TEST_IMAGES")
                ds_output = 'test'
        elif np.random.rand() > TRAIN_SPLIT:
            ds_output = 'test'
        ds = dataset[ds_output]

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

        for i in range(RANDOM_ROTATIONS):
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

            if VISUALISE_ONLY:
                print img_id
                f = plt.figure()
                ax = f.add_subplot(1, 5, 1)
                rgb.show(ax)
                #rgb_img_base.show(ax)
                #bbs.show(ax)
                ax = f.add_subplot(1, 5, 2)
                depth.show(ax)
                bbs.show(ax)

                ax = f.add_subplot(1, 5, 3)
                ax.imshow(pos_img)

                ax = f.add_subplot(1, 5, 4)
                ax.imshow(ang_img)

                ax = f.add_subplot(1, 5, 5)
                ax.imshow(width_img)

                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())

                plt.show()
                continue

            ds['img_id'].append(img_id)
            ds['rgb'].append(rgb.img)
            ds['depth_inpainted'].append(depth.img)
            ds['bounding_boxes'].append(bbs.to_array(pad_to=200))#if the list is variable length, hd5 will crash
            ds['grasp_points_img'].append(pos_img)
            ds['angle_img'].append(ang_img)
            ds['grasp_width'].append(width_img)

    # Save the output.
    if not VISUALISE_ONLY:
        with h5py.File(outfile_name, 'w') as f:
            for tt_name in dataset:
                for ds_name in dataset[tt_name]:
                    f.create_dataset('%s/%s' % (tt_name, ds_name), data=np.array(dataset[tt_name][ds_name]))