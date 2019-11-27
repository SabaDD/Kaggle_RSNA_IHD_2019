# -*- coding: utf-8 -*-


##CHANGE THIS DIRECTORY TO YOURS
HOME_DIR = '/run/user/1000/gvfs/smb-share:server=fileserver,share=public/sdadsetan/RSNA-Competition/'


STAGE1_TRAIN_DIR = HOME_DIR + 'stage_1_train_images/'
STAGE1_TEST_DIR = HOME_DIR + 'stage_1_test_images/'

TRAIN_CSV = HOME_DIR + 'stage_1_train.csv'
TEST_CSV = HOME_DIR + 'stage_1_sample_submission.csv'


TRAIN_NORMAL = HOME_DIR + 'stage_1_train_noraml_png/'
TRAIN_ABNORMAL = HOME_DIR + 'stage_1_train_abnormal_png/'
TRAIN_PNG = HOME_DIR + 'stage_1_train_bsb_png/'
TEST_PNG = HOME_DIR + 'stage_1_test_bsb_png/'


PRETRAINED_MODELS_PATH = HOME_DIR + 'keras_pretrained_models/'
pretrained_models = {
    "inception_resnet_v2" : PRETRAINED_MODELS_PATH + "inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5",
    "resnet_50": PRETRAINED_MODELS_PATH + "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                  
    "vgg16": PRETRAINED_MODELS_PATH + "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
              
}


MODELOUTPUT_ANY_PATH = HOME_DIR + 'Scripts/any_best_model.h5'
MODELOUTPUT_ANY_SUB_PATH = HOME_DIR + 'Scripts/anysubtype_best_model.hdf5'

EPOCHS = 3
STEPS = None