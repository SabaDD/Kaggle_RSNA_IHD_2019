# -*- coding: utf-8 -*-

import numpy as np
import pydicom


def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    intercept = data.RescaleIntercept if 'RescaleIntercept' in data else -1024
    slope= data.RescaleSlope if 'RescaleSlope' in data else 1
    
    dicom_fields = [127, 256, intercept, slope]
    
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
#%%
def metadata_window(img, print_ranges=True):
    # Get data from dcm
    window_center, window_width, intercept, slope = get_windowing(img)
    img = img.pixel_array
    
    # Window based on dcm metadata
    img = img * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    if print_ranges:
        print(img_min, img_max)
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    
    # Normalize
    img = (img - img_min) / (img_max - img_min)
    return img
#%%
def brain_window(img):
    window_min = 0
    window_max = 80
    _, _, intercept, slope = get_windowing(img)
    img = img.pixel_array
    img = img * slope + intercept
    img[img < window_min] = window_min
    img[img > window_max] = window_max
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def all_channels_window(img):
    grey_img = brain_window(img) * 3.0
    all_chan_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))
    all_chan_img[:, :, 2] = np.clip(grey_img, 0.0, 1.0)
    all_chan_img[:, :, 0] = np.clip(grey_img - 1.0, 0.0, 1.0)
    all_chan_img[:, :, 1] = np.clip(grey_img - 2.0, 0.0, 1.0)
    return all_chan_img

#%%
def map_to_gradient(grey_img):
    rainbow_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))
    rainbow_img[:, :, 0] = np.clip(4 * grey_img - 2, 0, 1.0) * (grey_img > 0) * (grey_img <= 1.0)
    rainbow_img[:, :, 1] =  np.clip(4 * grey_img * (grey_img <=0.75), 0,1) + np.clip((-4*grey_img + 4) * (grey_img > 0.75), 0, 1)
    rainbow_img[:, :, 2] = np.clip(-4 * grey_img + 2, 0, 1.0) * (grey_img > 0) * (grey_img <= 1.0)
    return rainbow_img

def rainbow_window(img):
    grey_img = brain_window(img)
    return map_to_gradient(grey_img)

#%%
def window_image(img, window_center, window_width):
    _, _, intercept, slope = get_windowing(img)
    img = img.pixel_array * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def bsb_window(img):
    brain_img = window_image(img, 40, 80)
    subdural_img = window_image(img, 80, 200)
    bone_img = window_image(img, 600, 2000)
    
    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    return bsb_img 

#%%
def rainbow_bsb_window(img):
    brain_img = window_image(img, 40, 80)
    subdural_img = window_image(img, 80, 200)
    bone_img = window_image(img, 600, 2000)
    combo = (brain_img*0.3 + subdural_img*0.5 + bone_img*0.2)
    return map_to_gradient(combo)

#%%
def sigmoid_window(img, window_center, window_width, U=1.0, eps=(1.0 / 255.0)):
    _, _, intercept, slope = get_windowing(img)
    img = img.pixel_array * slope + intercept
    ue = np.log((U / eps) - 1.0)
    W = (2 / window_width) * ue
    b = ((-2 * window_center) / window_width) * ue
    z = W * img + b
    img = U / (1 + np.power(np.e, -1.0 * z))
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def sigmoid_brain_window(img):
    return sigmoid_window(img, 40, 80)
#%%
def sigmoid_bsb_window(img):
    brain_img = sigmoid_window(img, 40, 80)
    subdural_img = sigmoid_window(img, 80, 200)
    bone_img = sigmoid_window(img, 600, 2000)
    
    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    return bsb_img

#%%
def map_to_gradient_sig(grey_img):
    rainbow_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))
    rainbow_img[:, :, 0] = np.clip(4*grey_img - 2, 0, 1.0) * (grey_img > 0.01) * (grey_img <= 1.0)
    rainbow_img[:, :, 1] =  np.clip(4*grey_img * (grey_img <=0.75), 0,1) + np.clip((-4*grey_img + 4) * (grey_img > 0.75), 0, 1)
    rainbow_img[:, :, 2] = np.clip(-4*grey_img + 2, 0, 1.0) * (grey_img > 0.01) * (grey_img <= 1.0)
    return rainbow_img

def sigmoid_rainbow_bsb_window(img):
    brain_img = sigmoid_window(img, 40, 80)
    subdural_img = sigmoid_window(img, 80, 200)
    bone_img = sigmoid_window(img, 600, 2000)
    combo = (brain_img*0.35 + subdural_img*0.5 + bone_img*0.15)
    combo_norm = (combo - np.min(combo)) / (np.max(combo) - np.min(combo))
    return map_to_gradient_sig(combo_norm)
    