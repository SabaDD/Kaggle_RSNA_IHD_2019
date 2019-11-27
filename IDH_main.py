# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import os
import csv
from tqdm import tqdm
import tensorflow as tf

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ShuffleSplit , StratifiedKFold
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.models import load_model


import Constants as c
import Data_handler as dh
import Models as md
import resnext as rs

from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

def my_auc(y_true, y_pred):
    h_auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return h_auc

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def _find_class_weights(y):
    return compute_class_weight(y)

def weighted_log_loss_metric(trues, preds): # add 0.2
    class_weights = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
    # higher epsilon than competition metric
    epsilon = 1e-7
    
    preds = np.clip(preds, epsilon, 1-epsilon)
    loss_subtypes = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)
    loss_weighted = np.average(loss_subtypes, axis=1, weights=class_weights)

    return - loss_weighted.mean()

def run(model, df, train_idx, valid_idx, test_df, epochs):
    
#    valid_predictions = []
#    test_predictions = []
    focal_loss_min = 2
    focal_loss = md.multilabel_focal_loss(class_weights=None, alpha=0.5, gamma=2)
    final_model = None
    for global_epoch in range(epochs):

        model.fit(df, train_idx,valid_idx, c.STAGE1_TRAIN_DIR, global_epoch)
        
        
#        model.evaluate(df, valid_idx, c.TRAIN_PNG)
        valid_predictions = model.predict(df, valid_idx, c.STAGE1_TRAIN_DIR)
        
        valid_predictions = np.concatenate((valid_predictions[0],valid_predictions[1]),axis = 1)
#        test_predictions.append(model.predict(test_df, range(test_df.shape[0]), c.TEST_PNG))
        
        
        valid_focal_loss = focal_loss(df.iloc[valid_idx].values, valid_predictions)
        value = K.get_value(valid_focal_loss).item(0)
        print("validation loss: %.4f" % value)
        
        if(value <= focal_loss_min):
            model.save(c.HOME_DIR+'Scripts/model_latest.h5') #model = load_model('my_model.h5') use this to load.
            focal_loss_min = value
            final_model = model

    return final_model
#    return test_predictions, valid_predictions
#        return valid_predictions
    
from skimage.color import rgb2gray

def find_skull_air_threshold(path, save_path):
    avg = []
    std = []
    gray_avg = []
    gray_std =[]
    area = []
    file_names = []
    all_info = []

    for filename in tqdm(os.listdir(path)):
        img,area1 = dh._read(c.STAGE1_TRAIN_DIR+filename.replace('png','dcm'))

        area.append(area1)
        file_names.append(filename)

    df = pd.DataFrame(data={"Filename": file_names , "area":area })
    df.to_csv('./'+save_path+'.csv', sep=',',index=False)
    
def _see_how_many_cases(normal_path, abnormal_path, test_path):
    normal = pd.read_csv(normal_path)
    abnormal= pd.read_csv(abnormal_path)
    test = pd.read_csv(test_path)

    norm_area = normal["area"].tolist()
    abnorm_area = abnormal["area"].tolist()
    test_area = test["area"].tolist()
    filename_norm = normal["Filename"].tolist()
    filename_abnorm = abnormal["Filename"].tolist()
    filename_test = test["Filename"].tolist()
    
    norm_cases = [i for j, i in enumerate(filename_norm) if norm_area[j] < 20000 ]
    abnorm_cases = [ i for j, i  in enumerate(filename_abnorm) if abnorm_area[j]< 20000 ]
    test_cases = [ i for j, i  in enumerate(filename_test) if test_area[j]< 20000 ]
    return norm_cases, abnorm_cases, test_cases

#find_skull_air_threshold(c.HOME_DIR + 'stage_1_train_normal_png', 'normal_info')
#
w, k , l = _see_how_many_cases('normal_info.csv','abnormal_info.csv', 'test_info.csv')

filename_dic = {}
for i in w:
    if i not in filename_dic.keys():
        filename_dic[i] = 1


if __name__ == '__main__':

    
    try:
        test_df = pd.read_pickle('./test.pkl')
        train_df = pd.read_pickle('./train.pkl')
    except (OSError, IOError):
        test_df = dh.read_testset(c.TEST_CSV)
        train_df = dh.read_trainset(c.TRAIN_CSV)
        #dump files for next use       
        test_df.to_pickle('./test.pkl')
        train_df.to_pickle('./train.pkl')
      
    train_df = train_df.sample(frac=0.001, replace=False, random_state=1)     
    test_df = test_df.sample(frac = 0.001, replace = False, random_state = 1)    
  
    split_seed = 1
    kfold = StratifiedKFold(n_splits=5,shuffle = True, random_state=split_seed).split(np.arange(train_df.shape[0]), train_df[('Label','any')].values)
    
    train_idx, valid_idx = next(kfold)
    
    # obtain model
    
    
    model_any_1 = md.anyDeepModel(engine=ResNet50, input_dims=(224, 224), batch_size=16, loss_fun=None, metrics_list= ['acc'],
                             checkpoint_path = c.MODELOUTPUT_ANY_PATH,
                             epochs = c.EPOCHS,
                             steps = c.STEPS,
                             learning_rate=1e-3, 
                             weights=c.pretrained_models["resnet_50"], verbose=2)
    
    model_any_1 = load_model(c.MODELOUTPUT_ANY_PATH)
    
    
 
    model_any_2 = md.anyDeepModel(engine=ResNet50, input_dims=(224, 224), batch_size=16, loss_fun= None, metrics_list= ['acc'],
                             checkpoint_path = c.MODELOUTPUT_ANY_PATH,
                             epochs = c.EPOCHS,
                             steps = c.STEPS,
                             learning_rate=1e-3, 
                             weights=c.pretrained_models["resnet_50"], verbose=2)
    
    model_any_2 = load_model(c.MODELOUTPUT_ANY_PATH)
    
    
    alpha_subtypes = 0.25 
    gamma_subtypes = 2
    ## TODO: CHANGE THE LOSS FUNCTION TO loss_fun = ['binary_crossentropy', 'binary_crossentropy'],metrics_list={"any_predictions":"binary_crossentropy","subtype_pred": } if the resutls with current loss is not good
    
    model_sub = md.MyDeepModel(engine=model_any_1,engine2=model_any_2, input_dims=(224, 224), batch_size=16, 
                             loss_fun=['binary_crossentropy', md.multilabel_focal_loss(alpha=alpha_subtypes, gamma=gamma_subtypes)], 
                             metrics_list={"any_predictions":"binary_crossentropy","subtype_pred": md.multilabel_focal_loss(alpha=0.25, gamma=2)},
                             checkpoint_path = c.MODELOUTPUT_ANY_SUB_PATH,
                             epochs = c.EPOCHS,
                             steps = c.STEPS,
                             learning_rate=1e-3, 
                             weights=c.pretrained_models["resnet_50"], verbose=2)
#    
##    history = model_any.fit(train_df, train_idx,valid_idx, c.TRAIN_PNG, filename_dic)
#    history = model_sub.fit(train_df, train_idx,valid_idx, c.TRAIN_PNG, filename_dic)
    model_sub.load(c.MODELOUTPUT_ANY_SUB_PATH)
    

    
    train_copy_df =train_df.copy() 
    print("Predict the labels of test_set")
    train_preds = model_sub.predict(test_df, range(test_df.shape[0]), c.TRAIN_PNG)
    train_preds = np.concatenate((train_preds[0],train_preds[1]),axis = 1)
    
    
#    test_copy_df =test_df.copy() 
#    print("Predict the labels of test_set")
#    test_preds = model_sub.predict(test_df, range(test_df.shape[0]), c.STAGE1_TEST_DIR)
#    test_preds = np.concatenate((test_preds[0],test_preds[1]),axis = 1)
#    
#    
#    test_df.iloc[:, :] = test_preds
##    
#    test_df = test_df.stack().reset_index()
#    
#    test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])
#    
#    test_df = test_df.drop(["Image", "Diagnosis"], axis=1)
#    
#    test_df.to_csv(c.HOME_DIR+'submission.csv', index=False)


