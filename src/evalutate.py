import os
from mrcnn.model import MaskRCNN
from SegmentUtils import PredictionConfig, evaluate, prep_dataset, DATA_PATH, WEIGHT_PATH


def main():
    # holdout validation dataset
    validation_set = prep_dataset(os.path.join(DATA_PATH, 'validation'))

    # generate model
    config = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir=os.path.join(WEIGHT_PATH, 'log/'),
                     config=config)

    # load pre-trained weights for LP dataset
    model.load_weights(os.path.join(WEIGHT_PATH, 'log/weights/mask_rcnn_lp.h5'),
                       by_name=True)

    # evaluate using PASCAL VOC mAP on holdout validation set
    val_mAP = evaluate(validation_set, model, config)
    print('mAP on validation set: {:.4f}'.format(val_mAP))


if __name__ == '__main__':
    main()
