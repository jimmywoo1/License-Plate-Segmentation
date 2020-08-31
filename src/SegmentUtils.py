'''
SegmentUtils.py

This module contains the utility classes and functions required to use the
matterport implementation of Mask R-CNN (https://github.com/matterport/Mask_RCNN)
for object detection and instance segmentation of the Car License Plates Dataset.

Classes:
    - LicensePlateDataset(mrcnn.utils.Dataset)
    - LicensePlateConfig(mrcnn.config.Config)
    - PredictionConfig(LicensePlateConfig)

Functions:
    - prep_dataset(path)
    - evaluate(dataset, model, cfg)

Attributes:
    - DATA_PATH
        - path to directory with the following structure:

          root
          ---- train
          -------- images
          -------- annotations
          ---- test
          -------- images
          -------- annotations
          ---- validation
          -------- images
          -------- annotations

    - WEIGHT_PATH
        - path to directory containing the following files:

        mask_rcnn_coco.h5
            - weights pre-trained on MS COCO dataset
        mask_rcnn_lp.hr
            - weights from transfer learning on License Plates Dataset
'''
import os
import re
import xmltodict
import numpy as np
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import compute_ap

DATA_PATH = 'ENTER DATA PATH HERE'
WEIGHT_PATH = 'ENTER WEIGHTS PATH HERE'


class LicensePlateDataset(Dataset):
    def __init__(self, path):
        '''
        child class of mrcnn.utils.Dataset
        sets the paths to images and corresponding annotations
        of a mrcnn.utils.Dataset object

        :param
        - path <str>: path to dataset containing an 'images' subdirectory
                      and an 'annotations' subdirectory
        '''
        super().__init__()
        self.__img_dir = os.path.join(path, 'images')
        self.__annot_dir = os.path.join(path, 'annotations')

    def get_bbox(self, img_id):
        '''
        parses an xml file containing the bounding box locations of
        within an image corresponding to the img_id

        :param
        - img_id <int>: ID corresponding to the image of interest

        :return
        - curr_dict <dict>: dictionary containing key, value pairs
                            for size of image and bounding box locations
        '''
        file_name = '{}.xml'.format(self.image_info[img_id]['id'])

        # read xml contents
        with open(os.path.join(self.__annot_dir, file_name), 'r') as f:
            xml = xmltodict.parse(f.read())

        curr_dict = {}
        bboxes = {}

        curr_dict['size'] = dict(xml['annotation']['size'])

        # xml file contains one bbox annotation
        try:
            bboxes[1] = list(xml['annotation']['object']['bndbox'].values())
        # multiple bboxes in xml file
        except TypeError:
            for idx, bbox in enumerate(xml['annotation']['object']):
                bboxes[idx+1] = list(bbox['bndbox'].values())

        # cast bbox coordinates to int
        bboxes = {k: [int(x) for x in v] for k, v in bboxes.items()}
        curr_dict['bboxes'] = bboxes

        return curr_dict

    def load_dataset(self):
        '''
        load paths to images and their corresponding annotations
        '''
        # add class for detection
        self.add_class('dataset', 1, 'license plate')

        # add individual observations to dataset
        for img_name in os.listdir(self.__img_dir):
            img_id = img_name.split('.')[0]
            img_path = os.path.join(self.__img_dir, img_name)
            annotation_path = os.path.join(
                self.__annot_dir, '{}.xml'.format(img_id))
            self.add_image('dataset', image_id=img_id,
                           path=img_path, annotation=annotation_path)

        # sort images by their indices
        self.image_info.sort(key=lambda x: int(re.findall(r'\d+', x['id'])[0]))

    def load_mask(self, img_id):
        '''
        generate a mask matching the bounding boxes of an image
        corresponding to the img_id

        :param
        - img_id <int>: ID corresponding to the image of interest

        :return
        - masks <np.array>: an array corresponding to the bounding box 
                            locations of the image of interest
        - ids <np.array>: an array containing the integer representation
                          of classes
        '''
        obs = self.get_bbox(img_id)
        size = obs['size']
        bboxes = obs['bboxes']

        # placeholder for masks
        masks = np.zeros(
            [int(size['height']), int(size['width']), len(bboxes)])
        ids = np.array([self.class_names.index('license plate')
                        for i in range(len(bboxes))])

        # generate masks
        for idx, bbox in bboxes.items():
            masks[bbox[1]:bbox[3], bbox[0]:bbox[2], idx-1] = 1

        return masks, ids

    def image_reference(self, img_id):
        return self.image_info[img_id]['path']

    def size(self):
        return len(self.image_info)


class LicensePlateConfig(Config):
    '''
    child class of mrcnn.config.Config
    '''
    NAME = 'licenseplate_cfg'
    NUM_CLASSES = 2  # license plate, background
    STEPS_PER_EPOCH = 303
    LEARNING_RATE = 0.001


class PredictionConfig(LicensePlateConfig):
    '''
    child class of LicensePlateConfig
    '''
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def prep_dataset(path):
    '''
    generates and processes a LicensePlateDataset object

    :param
    - path <str>: path to directory containing subdirectories
                  'images' and 'annotations'

    :return
    - dataset <LicensePlateDataset>: loaded dataset containing
                                     license plates images/annotations
    '''
    dataset = LicensePlateDataset(path)
    dataset.load_dataset()
    dataset.prepare()

    return dataset


def evaluate(dataset, model, cfg):
    '''
    computes the mean average precision (mAP) of a provided dataset
    using the provided model and configuration

    :param
    - dataset <LicensePlateDataset>: dataset to evaluate mAP on
    - model <mrcnn.model.MaskRCNN>: model used to evalute mAP
    - cfg <mrcnn.config.Config>: configurations for the mrcnn model 

    :return
    - <float>: value for mAP
    '''

    APs = []

    for idx, img_id in enumerate(dataset.image_ids):
        img, img_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, img_id,
                                                                     use_mini_mask=False)
        sample = np.expand_dims(mold_image(img, cfg), 0)
        pred = model.detect(sample)

        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 pred[0]['rois'], pred[0]['class_ids'],
                                 pred[0]['scores'], pred[0]['masks'])

        APs.append(AP)

    return np.mean(APs)
