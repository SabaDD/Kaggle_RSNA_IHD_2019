# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
from keras import layers
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from tqdm import tqdm
from keras.callbacks import ReduceLROnPlateau

import Constants as c
import Data_handler as dh

import logging

  
#from keras import backend as K
#import tensorflow as tf

#logging.basicConfig(filename='error_files.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


types = ['epidural', 'intraparenchymal','intraventricular', 'subarachnoid', 'subdural', 'any']

train_df = pd.read_csv(c.HOME_DIR + 'stage_1_train.csv')
train_df = train_df.set_index(['ID'])
sub_df = pd.read_csv(c.HOME_DIR + 'stage_1_sample_submission.csv')
sub_df = sub_df.set_index(['ID'])

#train_df['filename'] = train_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")
#train_df['type'] = train_df['ID'].apply(lambda st: st.split('_')[2])
#sub_df['filename'] = sub_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")
#sub_df['type'] = sub_df['ID'].apply(lambda st: st.split('_')[2])

print(train_df.shape)
train_df.head()

#test_df = pd.DataFrame(sub_df.filename.unique(), columns=['filename'])
#print(test_df.shape)
#test_df.head()



########################################### randomly samples 400k images.
#np.random.seed(1749)
#sample_files = np.random.choice(os.listdir(BASE_PATH + TRAIN_DIR), 400000)
#sample_df = train_df[train_df.filename.apply(lambda x: x.replace('.png', '.dcm')).isin(sample_files)]

# SAMPLE ALL THE FILES!!
sample_files = os.listdir(c.STAGE1_TRAIN_DIR)
test_files = os.listdir(c.STAGE1_TEST_DIR)
#sample_df = train_df[train_df.filename.apply(lambda x: x.replace('.png', '.dcm')).isin(sample_files)]
#
#pivot_df = sample_df[['Label', 'filename', 'type']].drop_duplicates().pivot(
#    index='filename', columns='type', values='Label').reset_index()
#print(pivot_df.shape)
#pivot_df.head()

def window_image(img, window_center,window_width, intercept, slope, rescale=True):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    
    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)
    
    return img
    
def _get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def _get_windowing(data):
    intercept = data.RescaleIntercept if 'RescaleIntercept' in data else -1024
    slope= data.RescaleSlope if 'RescaleSlope' in data else 1
    
    dicom_fields = [127, 256, intercept, slope]
    
    return [_get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def _save_and_resize(filenames, load_dir):    
    
    for filename in tqdm(filenames):
        path = load_dir + filename
        
        ID = (filename.split('.'))[0]
#        print(ID)
        row_data = train_df.loc[ID+'_any', 'Label']
#        print(row_data)
        if (row_data.any() != 0):
            save_dir = c.HOME_DIR + 'stage_1_train_abnormal_png/'
        else:
            save_dir = c.HOME_DIR + 'stage_1_train_normal_png/'
#        save_dir = c.HOME_DIR + 'stage_1_train_normal_png/'
        new_path = save_dir + filename.replace('.dcm', '.png')
        try:
#            if not os.path.isfile(new_path):
            dcm = pydicom.dcmread(path, force= True)
            window_params = _get_windowing(dcm)
    
            img = dcm.pixel_array
            img = window_image(img, *window_params)
#                img = img*255
            resized = cv2.resize(img, (512, 512))
            wo_artifact = dh._brain_segmentation(resized, threshold = 0)
            cv2.imwrite(new_path, wo_artifact)
        except:
            logging.debug(filename)
            continue
        
def _save_and_resize_test(filenames, load_dir):    
    save_dir = c.HOME_DIR + 'stage_1_test_png/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in tqdm(filenames):
        path = load_dir + filename
        new_path = save_dir + filename.replace('.dcm', '.png')
        try:
#            if not os.path.isfile(new_path):
            dcm = pydicom.dcmread(path, force = True)
            window_params = _get_windowing(dcm)
    
            img = dcm.pixel_array
            img = window_image(img, *window_params)
#                img = img*255
            resized = cv2.resize(img, (512, 512))
            wo_artifact = dh._brain_segmentation(resized, threshold = 0)
            cv2.imwrite(new_path, wo_artifact)

        except:
            logging.debug(filename)
            continue
       #################################################################

def _new_save_train(filenames, load_dir):
    save_dir = c.HOME_DIR + 'stage_1_train_bsb_png/'

    for filename in tqdm(filenames):
        path = load_dir + filename
        new_path = save_dir + filename.replace('.dcm', '.png')
        try:
           
            img = dh._read(path)
            
            cv2.imwrite(new_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        except:
#            logging.debug(filename)
            continue
def _new_save_test(filenames, load_dir):
    save_dir = c.HOME_DIR + 'stage_1_test_bsb_png/'
    for filename in tqdm(filenames):
        path = load_dir + filename
        new_path = save_dir + filename.replace('.dcm', '.png')
        try:
           
            img = dh._read(path)

            cv2.imwrite(new_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        except:
#            logging.debug(filename)
            continue        
_new_save_train(filenames=sample_files, load_dir=c.STAGE1_TRAIN_DIR)
_new_save_test(filenames=test_files, load_dir=c.STAGE1_TEST_DIR)