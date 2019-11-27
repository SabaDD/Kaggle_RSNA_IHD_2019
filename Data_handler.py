#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:00:08 2019

"""

import numpy as np
import cv2 
import math
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import os

import pydicom
import keras



def _find_area(img):
    img= cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
    img.astype(np.uint8)
    
    
    ret, thresh = cv2.threshold(img,0,255,0)
    ret, markers = cv2.connectedComponents(thresh)
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)+1) if m!=0] 

    return max(marker_area)

def _brain_segmentation(img, threshold = 0 ):

    img= cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
    img.astype(np.uint8)
    
     
    
    ret, thresh = cv2.threshold(img,0,255,0)
    ret, markers = cv2.connectedComponents(thresh)
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)+1) if m!=0] 
    
    
    brain_out = img.copy()
    max_area = 0
    brain_mask = np.full(img.shape, False)
    
    if len(marker_area) > 0: 
        largest_component = np.argmax(marker_area)+1                     
        max_area = max(marker_area)
        
        brain_mask = markers==largest_component
    
    
        #clear those pixels that don't correspond to the brain
        brain_out[brain_mask==False] = 0
    

    return brain_out,brain_mask,max_area

def _segment_with_mask(img, mask, max_area):
    img= cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
    img.astype(np.uint8)
    
    if max_area > 0:
        out = img.copy()
        out[mask==False] = 0
        
        return out
    else:
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


def _window_image(img, window_center, window_width, slope, intercept):
    img = (img * slope + intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img 
def _map_to_gradient_sig(grey_img):
    rainbow_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))
    rainbow_img[:, :, 0] = np.clip(4*grey_img - 2, 0, 1.0) * (grey_img > 0.01) * (grey_img <= 1.0)
    rainbow_img[:, :, 1] =  np.clip(4*grey_img * (grey_img <=0.75), 0,1) + np.clip((-4*grey_img + 4) * (grey_img > 0.75), 0, 1)
    rainbow_img[:, :, 2] = np.clip(-4*grey_img + 2, 0, 1.0) * (grey_img > 0.01) * (grey_img <= 1.0)
    return rainbow_img

def _sigmoid_window(img, window_center, window_width, U=1.0, eps=(1.0 / 255.0)):
    _, _, intercept, slope = _get_windowing(img)
    img = img.pixel_array * slope + intercept
    ue = np.log((U / eps) - 1.0)
    W = (2 / window_width) * ue
    b = ((-2 * window_center) / window_width) * ue
    z = W * img + b
    img = U / (1 + np.power(np.e, -1.0 * z))
    #normalize
    
    if (not img.size) or (img.max() == img.min()):
        return np.zeros((512,512))
    
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def _sigmoid_bsb_window(img):
    brain_img = _sigmoid_window(img, 40, 80)
    brain_img,mask,max_area = _brain_segmentation(brain_img)
    
    subdural_img = _sigmoid_window(img, 80, 200)
    subdural_img = _segment_with_mask(subdural_img, mask, max_area)
    
    bone_img = _sigmoid_window(img, 600, 2000)
    bone_img = _segment_with_mask(bone_img, mask, max_area)
    
    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    return bsb_img, max_area

def _sigmoid_rainbow_bsb_window(img):
    
    brain_img = _sigmoid_window(img, 40, 80)
    brain_img,mask,max_area = _brain_segmentation(brain_img)
    
    subdural_img = _sigmoid_window(img, 80, 200)
    subdural_img = _segment_with_mask(subdural_img, mask, max_area)
    
    bone_img = _sigmoid_window(img, 600, 2000)
    bone_img = _segment_with_mask(bone_img, mask, max_area)
    
    combo = (brain_img*0.35 + subdural_img*0.5 + bone_img*0.15)
    combo_norm = (combo - np.min(combo)) / (np.max(combo) - np.min(combo))
    return _map_to_gradient_sig(combo_norm), max_area

def _normalize(img):
    if img.max() == img.min():
        return np.zeros(img.shape)
    return 2 * (img - img.min())/(img.max() - img.min()) - 1
def _below_threshold(area):
    if area < 20000:
        return True
    else:
        return False


def _read(path, desired_size=(224, 224)):
    """Will be used in DataGenerator"""
    
    dcm = pydicom.dcmread(path)
    area = 0
    try:
        img, area= _sigmoid_bsb_window(dcm)
#        img, area = _sigmoid_rainbow_bsb_window(dcm)
        img= cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)

    except Exception as e :
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        img = np.zeros((*desired_size,3))
    
    img = cv2.resize(img, desired_size, interpolation=cv2.INTER_LINEAR)
    
    if(_below_threshold(area)):
        img = np.zeros((*desired_size,3))
        
    return img, area

def _read_png(path, desired_size= (224,224)):
    

    if not os.path.isfile(path):
#        print(path)
        return np.zeros((*desired_size,3))
    img = cv2.imread(path)
    if img is None:
#       raise Exception("Image is NONE") 
        return np.zeros((*desired_size,3))
    img= cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
    img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, desired_size, interpolation=cv2.INTER_LINEAR)
    
    return img
#_read_png(c.TRAIN_PNG+'ID_0a4b51e10.png')

def read_testset(filename):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    
    return df

def read_trainset(filename):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    
    duplicates_to_remove = [
        1598538, 1598539, 1598540, 1598541, 1598542, 1598543,
        312468,  312469,  312470,  312471,  312472,  312473,
        2708700, 2708701, 2708702, 2708703, 2708704, 2708705,
        3032994, 3032995, 3032996, 3032997, 3032998, 3032999
    ]
    
    df = df.drop(index=duplicates_to_remove)
    df = df.reset_index(drop=True)
    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    
    return df

class AnySubTrainDataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels,steps,over_sample, batch_size, img_size, 
                 img_dir,file_to_zero, *args, **kwargs):

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.over_sample = over_sample
        self.img_size = img_size
        self.img_dir = img_dir
        self.file_to_zero = file_to_zero
        self.on_epoch_end()

    def oversample(self):
        
        zero_ids = list(self.labels.loc[self.labels[('Label','any')] == 0].index.values)
    
        one_ids = list(self.labels.loc[self.labels[('Label','any')] == 1].index.values)
        
        self.list_IDs = zero_ids + one_ids + one_ids + one_ids
     
        np.random.shuffle(self.list_IDs)

    def __len__(self):
        return int(math.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        X, Y_sub, Y_any = self.__data_generation(list_IDs_temp)
        return X, [Y_any, Y_sub]
       
    def on_epoch_end(self):
        if self.over_sample:
            self.oversample()
        self.indices = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        if not os.path.isdir(self.img_dir):
            raise Exception("Cannot find a folder for train set!!")  
            
        X = np.empty((self.batch_size, *self.img_size, 3))
        Y_sub = np.empty((self.batch_size, 5), dtype=np.float32)
        Y_any = np.empty((self.batch_size, 1), dtype=np.float32)
        
        for i, ID in enumerate(list_IDs_temp):
#            temp , _ = _read(self.img_dir+ID+'.dcm', self.img_size)
            if ID+'.png' not in self.file_to_zero.keys() :
                X[i,] = _read_png(self.img_dir+ID+'.png', self.img_size)
                 
                if(np.mean(X[i,]) == 0):
    #             
                   Y_sub[i,] = np.zeros(5)
                   Y_any[i,] = 0
                else:
                    Y_sub[i,] = self.labels.drop(('Label','any'),axis = 1).loc[ID].values
                    Y_any[i,] = self.labels.loc[ID, ('Label','any')]
            else:
               X[i,] = np.zeros((*self.img_size,3)) 
               Y_sub[i,] = np.zeros(5)
               Y_any[i,] = 0
               
        return X, Y_sub, Y_any
        
    


class AnyTrainDataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels,steps,over_sample, batch_size, img_size, 
                 img_dir,file_to_zero, *args, **kwargs):

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.over_sample = over_sample
        self.img_size = img_size
        self.img_dir = img_dir
        self.file_to_zero = file_to_zero
        self.on_epoch_end()
        


    def oversample(self):
        
        zero_ids = list(self.labels.loc[self.labels[('Label','any')] == 0].index.values)
    
        one_ids = list(self.labels.loc[self.labels[('Label','any')] == 1].index.values)
        
        self.list_IDs = zero_ids + one_ids + one_ids
     
        np.random.shuffle(self.list_IDs)
        

    
    def __len__(self):
        return int(math.ceil(len(self.list_IDs) / self.batch_size))
        

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        X, Y_any = self.__data_generation(list_IDs_temp)
        return X, Y_any
    
    def on_epoch_end(self):
        if self.over_sample:
            self.oversample()
        self.indices = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indices)
       

    def __data_generation(self, list_IDs_temp):
        if not os.path.isdir(self.img_dir):
            raise Exception("Cannot find a folder for train set!!")  
        
        X = np.empty((self.batch_size, *self.img_size, 3))
        Y_any = np.empty((self.batch_size, 1), dtype=np.float32)
        
        for i, ID in enumerate(list_IDs_temp):
#            temp , _ = _read(self.img_dir+ID+'.dcm', self.img_size)
            if ID+'.png' not in self.file_to_zero.keys() :
                X[i,] = _read_png(self.img_dir+ID+'.png', self.img_size)
                
                if(np.mean(X[i,]) == 0):
                   Y_any[i,] = 0
                else:
    
                    try:
                        Y_any[i,] = self.labels.loc[ID, ('Label','any')]
                    except:
                        print(ID)
            else:
                X[i,] = np.zeros((*self.img_size,3)) 
                Y_any[i,] = 0
                 
#        print(Y_any)
#        print(len(Y_any))
        return X, Y_any
    
    
class TestDataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, batch_size, img_size, 
                 img_dir, *args, **kwargs):

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        return int(math.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        X = self.__data_generation(list_IDs_temp)
        return X

    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        
        X = np.empty((self.batch_size, *self.img_size, 3))
        
        for i, ID in enumerate(list_IDs_temp):
            if not os.path.isdir(self.img_dir):
                raise Exception("Cannot find a folder for test set!!")  
                
#            X[i,],_ = _read(self.img_dir+ID+'.dcm', self.img_size)
            X[i,] = _read_png(self.img_dir+ID+'.png', self.img_size)
        return X





