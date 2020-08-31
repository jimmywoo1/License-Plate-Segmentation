import os
from mrcnn.model import MaskRCNN
from SegmentUtils import prep_dataset, LicensePlateConfig, DATA_PATH, WEIGHT_PATH


def main():
    # load training set (75/15/15 split between train/test/validation)
    train_set = prep_dataset(os.path.join(DATA_PATH, 'train'))
    test_set = prep_dataset(os.path.join(DATA_PATH, 'test'))

    # generate model
    config = LicensePlateConfig()
    model = MaskRCNN(mode='training', model_dir=os.path.join(
        WEIGHT_PATH, 'log/'), config=config)

    # load pre-trained MS COCO weights
    model.load_weights(os.path.join(WEIGHT_PATH, 'mask_rcnn_coco.h5'), by_name=True,
                       exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

    # train top layer
    model.train(train_set, test_set,
                learning_rate=config.LEARNING_RATE, epochs=10, layers='heads')

    # adjust learning rate for finetuning to avoid overfitting
    config.LEARNING_RATE = 1e-5

    # finetune all layers
    model.train(train_set, test_set,
                learning_rate=config.LEARNING_RATE, epochs=5, layers='all')


if __name__ == '__main__':
    main()
