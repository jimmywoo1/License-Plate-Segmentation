import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mrcnn.model import MaskRCNN, mold_image
from SegmentUtils import PredictionConfig, evaluate, prep_dataset, DATA_PATH, WEIGHT_PATH


def detect(model, img, config):
    '''
    detect bounding boxes of an image using the provided model and config

    :param
    - model <mrcnn.model.MaskRCNN>: Mask R-CNN model for segmentation
    - img <np.array>: image of interest
    - cfg <mrcnn.config.Config>: configurations for the Mask R-CNN model

    :return
    - <dict>: dictionary containing regions of interest
    '''
    # preprocess image
    img_scaled = mold_image(img, config)
    img_scaled = np.expand_dims(img_scaled, 0)

    # object detection
    return model.detect(img_scaled, verbose=0)[0]


def add_mask(ax, img, mask):
    '''
    add image and apply mask to provided axis

    :param
    - ax <matplotlib.axes._subplots.AxesSubplot>: axis to display results
    - img <np.array>: image of interest
    - mask <np.array>: mask highlighting region of interst in img
    '''
    # ground truth image
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Groud truth')

    # add bounding rectangles for groud truths
    for idx in range(mask.shape[-1]):
        # numpy boolean mask
        x = np.any(mask[:, :, idx], axis=0)
        y = np.any(mask[:, :, idx], axis=1)

        # first occurence of mask
        x1 = np.argmax(x)
        y1 = np.argmax(y)

        # size - position of last occurence
        x2 = len(x) - np.argmax(x[::-1])
        y2 = len(y) - np.argmax(y[::-1])

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, color='green',
                                 fill=False, linewidth=3.0)
        ax.add_patch(rect)


def add_detected_img(ax, img, pred):
    '''
    add image and draw bounding rectangle aroud regions of interest
    on the provided axis

    :param
    - ax <matplotlib.axes._subplots.AxesSubplot>: axis to display results
    - img <np.array>: image of interest
    - pred <dict>: dictionary containing regions of interest
    '''
    # detection image
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Detected bbox')

    # add bounding rectangles for detected bboxes
    for bbox in pred['rois']:
        y1, x1, y2, x2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, color='red',
                                 fill=False, linewidth=3.0)
        ax.add_patch(rect)


def main(img_id, img_path):
    # holdout validation dataset
    validation_set = prep_dataset(os.path.join(DATA_PATH, 'validation'))

    # generate model
    config = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir=os.path.join(WEIGHT_PATH, 'log/'),
                     config=config)

    # load pre-trained weights for LP dataset
    model.load_weights(os.path.join(WEIGHT_PATH, 'mask_rcnn_lp.h5'),
                       by_name=True)

    # no images, evaluate using PASCAL VOC mAP on holdout validation set
    if img_id is None and img_path is None:
        val_mAP = evaluate(validation_set, model, config)
        print('mAP on validation set: {:.4f}'.format(val_mAP))
        return
    # demo on validation image
    elif img_id is not None:
        img_id = int(img_id)

        # load image and corresponding mask
        img = validation_set.load_image(img_id)
        mask = validation_set.load_mask(img_id)[0]

        # object detection
        pred = detect(model, img, config)

        # generate images
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        add_mask(axes[0], img, mask)
        add_detected_img(axes[1], img, pred)
    else:
        img_path = os.path.abspath(img_path)

        # read and preprocess image
        img = cv2.imread(img_path)

        # object detection
        pred = detect(model, img, config)

        # generate image
        fig, ax = plt.subplots(figsize=(10, 5))
        add_detected_img(ax, img, pred)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_id', help='ID of image in validation set')
    parser.add_argument('-p', '--img_path', help='path to image to run detection on')    
    args = parser.parse_args()

    main(**vars(args))
