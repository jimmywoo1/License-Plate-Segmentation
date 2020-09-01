# License Plate Segmentation using Mask R-CNN

The goal of this project is to use an implementation of Mask R-CNN ([source](https://github.com/matterport/Mask_RCNN)) to perform object segmentation on the [Car License Plate Dataset](https://www.kaggle.com/andrewmvd/car-plate-detection) to accurately detect the bounding boxes of license plates contained in each of the images of the dataset. We will cover the basics behind Mask R-CNN, the description of the dataset, the approach taken in this project, and the implementation details of this project.

## Mask R-CNN

Mask R-CNN is an object instance segmentation framework that is an extension of Faster R-CNN ([paper](https://arxiv.org/abs/1703.06870)). Faster R-CNN utilizes Region Proposal Network (RPN) for region of interest (ROI) proposal, replacing the selective search method used in the previous iterations of the R-CNN family. Mask R-CNN extends Faster R-CNN by decoupling the classification task from the mask prediction task. This has been achieved by adding a third branch in the network for mask prediction in addition to the existing branches for classification and localization. The mask prediction branch contains a small fully connected dense layer for each proposed ROI, generating predictions for the segmentation mask at the pixel level. The architecture of Mask R-CNN is shown below:

![Mask R-CNN ARchitecture](/imgs/mask_r_cnn_architecture.png)

THe ROIAlign layer is an improvement to the ROI pooling layer that deals with the misalignment caused by the quantization in the pooling layer, by using bilinear interpolation instead.

## Car License Plate Dataset

The Car License Plate Dataset is comprised of 433 distinct images of cars, with the corresponding bounding box annotations of the car license plates within each of the images. Each image contains one or more cars, with their license plate areas visible in the image.

![test](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F793761%2Fc15e812b3ab9aad2c0694a2e1f7548e9%2FUntitled.png?generation=1590981584876269&alt=media)

## Approach

In this project, a Mask R-CNN implementation built on FPN and ResNet101 was used. We initially started with the pre-trained weights on [MS COCO](https://cocodataset.org/#home), removing the classification, bounding box, mask layers and training the model with all of the weights freezed (i.e. transfer learning). Once the initial training phase was completed, finetuning was performed for all layers in the model.

**Note:** The annotations in this dataset do not contain the coordinates for the masks of license plates, and thus the scope of this project will exclude mask segmentation, and will only contain bounding box detection.

## Model Implementation Results

### Prerequisites

The [Mask R-CNN implementation](https://github.com/matterport/Mask_RCNN) must be installed, along with the other modules specified in [requirements.txt](/requirements.txt)

The [Car License Plate Dataset](https://www.kaggle.com/andrewmvd/car-plate-detection) must use the following structure. This project used a 75/15/15 split for training, testing, and holdout validation sets respectively. 

```bash
├───test
│   ├───annotations
│   └───images
├───train
│   ├───annotations
│   └───images
└───validation
    ├───annotations
    └───images
```

Prior to running the [train.py](/src/train.py) and/or [evaluate.py](/src/evaluate.py), *DATA_PATH* and *WEIGHT_PATH* must be specified in [SegmentUtils.py](/src/SegmentUtils.py), where *DATA_PATH* is the root path displayed above, and the *WEIGHT_PATH* is the directory containing the [weight files](/weights/).

### Detection of Bounding Boxes and Evaluation of the Trained Model

To perform detection of license plates on one of the images from the validation, use the following command:

```bash
python evaluation.py --img_id $IMG_ID
```
![Demo](/imgs/demo.png)

To perfom detection on images outside of the dataset, use the following command:

```bash
python evaluation.py --img_path $IMG_PATH
```
![Demo](/imgs/new_img.png)

### Training from scratch

To train the model from scratch, run the following command:

```bash
python train.py
```

If you want to train the model on your own dataset, modify the *Dataset* and *Config* classes in [train.py](/src/train.py), and modify the architecture/training scheme in [train.py](/src/train.py) as needed. 

## References

1. [Mask R-CNN implementation](https://github.com/matterport/Mask_RCNN)
2. [Car License Plate Dataset](https://www.kaggle.com/andrewmvd/car-plate-detection)
3. [Mask R-CNN paper](https://arxiv.org/abs/1703.06870)
4. [Faster R-CNN paper](https://arxiv.org/abs/1506.01497)
5. [Model architecture image](https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272)
6. [MS COCO](https://cocodataset.org/#home)
7. [Mask R-CNN example](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon)
