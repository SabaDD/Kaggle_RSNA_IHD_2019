# -*- coding: utf-8 -*-

import numpy as np
import math


import keras
from keras.layers import Dropout, GlobalAveragePooling2D, Activation, concatenate, Dense, Input, Multiply, Lambda, InputLayer, multiply, Reshape
from keras.regularizers import l2
from keras.initializers import he_normal
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

import tensorflow as tf
from keras import backend as K

import Data_handler as dh




def weighted_log_loss_metric(trues, preds): # add 0.2
    class_weights = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
    # higher epsilon than competition metric
    epsilon = 1e-7
    
    preds = np.clip(preds, epsilon, 1-epsilon)
    loss_subtypes = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)
    loss_weighted = np.average(loss_subtypes, axis=1, weights=class_weights)

    return - loss_weighted.mean()

def np_multilabel_loss(y_true, y_pred, class_weights=None):
    y_pred = np.where(y_pred > 1-(1e-07), 1-1e-07, y_pred)
    y_pred = np.where(y_pred < 1e-07, 1e-07, y_pred)
    single_class_cross_entropies = - np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred), axis=0)
    
    print(single_class_cross_entropies)
    if class_weights is None:
        loss = np.mean(single_class_cross_entropies)
    else:
        loss = np.sum(class_weights*single_class_cross_entropies)
    return loss

def get_raw_xentropies(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
    xentropies = y_true * tf.log(y_pred) + (1-y_true) * tf.log(1-y_pred)
    return -xentropies

# multilabel focal loss equals multilabel loss in case of alpha=0.5 and gamma=0 
def multilabel_focal_loss(class_weights=None, alpha=0.5, gamma=2):
    def mutlilabel_focal_loss_inner(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        xentropies = get_raw_xentropies(y_true, y_pred)

        # compute pred_t:
        y_t = tf.where(tf.equal(y_true,1), y_pred, 1.-y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha * tf.ones_like(y_true), (1-alpha) * tf.ones_like(y_true))

        # compute focal loss contributions
        focal_loss_contributions =  tf.multiply(tf.multiply(tf.pow(1-y_t, gamma), xentropies), alpha_t) 

        # our focal loss contributions have shape (n_samples, s_classes), we need to reduce with mean over samples:
        focal_loss_per_class = tf.reduce_mean(focal_loss_contributions, axis=0)

        # compute the overall loss if class weights are None (equally weighted):
        if class_weights is None:
            focal_loss_result = tf.reduce_mean(focal_loss_per_class)
        else:
            # weight the single class losses and compute the overall loss
            weights = tf.constant(class_weights, dtype=tf.float32)
            focal_loss_result = tf.reduce_sum(tf.multiply(weights, focal_loss_per_class))
            
        return focal_loss_result
    return mutlilabel_focal_loss_inner




def _initial_layer(input_dims):
    inputs = keras.layers.Input(input_dims)
    
    x = keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), name="initial_conv2d")(inputs)
    x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='initial_bn')(x)
    x = keras.layers.Activation('relu', name='initial_relu')(x)
    
    return keras.models.Model(inputs, x)

def _weighted_log_loss(y_true, y_pred):
    
    class_weights_any = 1.8
    class_weights_sub = np.array([0.9, 0.9, 0.9, 0.9, 0.9]) # label "any" has twice the weight of the others
    
    y_pred_any = keras.backend.clip(y_pred[0], keras.backend.epsilon(), 1.0-keras.backend.epsilon())
    y_pred_sub = keras.backend.clip(y_pred[1], keras.backend.epsilon(), 1.0-keras.backend.epsilon())

    out_sub = -(         y_true[1:]  * keras.backend.log(      y_pred_sub) * class_weights_sub
            + (1.0 - y_true[1:]) * keras.backend.log(1.0 - y_pred_sub) * class_weights_sub)
    
    out_any = -(         y_true[0]  * keras.backend.log(      y_pred_any) * class_weights_any
            + (1.0 - y_true[0]) * keras.backend.log(1.0 - y_pred_any) * class_weights_any)
    return out_any , keras.backend.mean(out_sub, axis=-1)

def muMultiply(x):
    return x[0]*x[1]





class MyDeepModel:
    
    def __init__(self, engine,engine2, input_dims, batch_size,loss_fun, metrics_list,checkpoint_path,epochs,steps, learning_rate=1e-3, 
                  weights="imagenet", verbose=1):
        
        self.engine = engine
        self.engine2 = engine2
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.loss_fun = loss_fun
        self.metrics_list = metrics_list
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.steps = steps
        self.learning_rate = learning_rate
        self.weights = weights
        self.verbose = verbose
        self._build()
        self.checkpoint = ModelCheckpoint(filepath=self.checkpoint_path,
                                          mode="min",
                                          verbose=1,
                                          save_best_only=True,
                                          save_weights_only=True,
                                          period=1)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.5,
                                           patience=2,
                                           min_lr=1e-8,
                                           mode="min")
        self.e_stopping = EarlyStopping(monitor="val_loss",
                                        min_delta=0.01,
                                        patience=5,
                                        mode="min")

    
    def _build(self):
         inputs  = Input(shape = (*self.input_dims, 3))
#         engine = self.engine(include_top= False, weights = self.weights)
         engine = self.engine
         for l in engine.layers:
             if 'dense' in l.name or 'any_predictions' in l.name:
                 l.trainable = True
             else:
                 l.trainable = False
         
         any_pred = Model(engine.input, engine.output)(inputs)

         
         round_pred = Lambda(lambda x: K.round(x))(any_pred)
         
         cond = K.switch(K.equal(round_pred, 1), inputs , K.zeros(tf.shape(inputs)))
#         result = Lambda(lambda x: x[0]*x[1])([inputs,cond])

         result = Input(tensor = cond, name = 'new_image') 

#         resnext = self.engine2(input_shape = (224, 224, 3),include_top = False, pooling = 'avg', depth=29, cardinality=8, width=64)
         engine2 = self.engine2
#         engine2 = self.engine2(include_top= False, weights = self.weights)
#         for l in engine2.layers:
#             l.trainable = False
         layer = None
         for l in engine2.layers:
             if 'dropout' in l.name:
                 layer = l
         y = Model(engine2.input, layer.output)(result)
        
         sub_pred = Dense(5, name = 'subtype_pred', kernel_initializer = he_normal(seed =12),
                          kernel_regularizer = l2(0.1),
                          bias_regularizer = l2(0.1),
                          activation='sigmoid')(y)
         
         self.model = Model(inputs = [inputs,result], outputs =  [any_pred,sub_pred] )
#         self.model = Model(inputs = inputs, outputs = any_pred)
#         self.model.compile(loss =['binary_crossentropy', multilabel_focal_loss(alpha=0.3, gamma=2)],loss_weights = [1., 0.], optimizer = Adam(0.001))
         self.model.compile(loss =self.loss_fun,
                             loss_weights = [1., 0.],
                             metrics=self.metrics_list,
                             optimizer = Adam(self.learning_rate))
         self.model.summary()
         
    
    
    def fit(self, df, train_idx,valid_idx, img_dir,file_to_delete):
        self.model.fit_generator(
            dh.AnySubTrainDataGenerator(
                df.iloc[train_idx].index, 
                df.iloc[train_idx],
                self.steps,
                True,
                self.batch_size, 
                self.input_dims, 
                img_dir,
                file_to_delete
            ),
            validation_data = dh.AnySubTrainDataGenerator(
                df.iloc[valid_idx].index, 
                df.iloc[valid_idx], 
                self.steps,
                False,
                self.batch_size, 
                self.input_dims, 
                img_dir,
                file_to_delete
            ),
            epochs = self.epochs,
            verbose=self.verbose,
            use_multiprocessing=False,
            workers=8,
            callbacks=[self.checkpoint, self.reduce_lr, self.e_stopping]
        )
    
    
    def predict(self, df, test_idx, img_dir):
        predictions = self.model.predict_generator(
            dh.TestDataGenerator(
                df.iloc[test_idx].index, 
                None, 
                1, 
                self.input_dims, 
                img_dir
            ),
            verbose=self.verbose,
            use_multiprocessing=False,
            workers=8
        )
        return predictions[:df.iloc[test_idx].shape[0]]
    
    def save(self, path):
        self.model.save_weights(path)
    
    def load(self, path):
        self.model.load_weights(path)
        
        
class anyDeepModel(MyDeepModel):
    
    def __init__(self, engine, input_dims, batch_size,loss_fun, metrics_list, checkpoint_path ,epochs,steps, learning_rate=1e-3, 
                  weights="imagenet", verbose=1):
        
        MyDeepModel.__init__(self,
                             engine = engine,
                             engine2 = None,
                             input_dims = input_dims,
                             batch_size = batch_size,
                             loss_fun = loss_fun,
                             metrics_list = metrics_list,
                             checkpoint_path = checkpoint_path,
                             epochs = epochs,
                             steps = steps,
                             learning_rate= learning_rate,
                             weights = weights,
                             verbose = verbose)
                                  
      
    def _build(self):
         inputs  = Input(shape = (*self.input_dims, 3))
         engine = self.engine(include_top= False, weights = self.weights)
#         for l in engine.layers:
#             l.trainable = False
         x = Model(engine.input, engine.output)(inputs)
         x = GlobalAveragePooling2D()(x)
         x = Dropout(0.5)(x)
         any_logits = Dense(1, kernel_initializer = he_normal(seed=11),
                            kernel_regularizer = l2(0.1),
                            bias_regularizer = l2(0.1))(x)
         
         any_pred = Activation('sigmoid', name = 'any_predictions')(any_logits)
         
         
         self.model = Model(inputs = [inputs], outputs =  [any_pred] )
         self.model.compile(loss ='binary_crossentropy',optimizer = Adam(self.learning_rate),metrics = self.metrics_list)
#         self.model.summary()
         
         
    def fit(self, df, train_idx,valid_idx, img_dir,file_to_delete):
        self.model.fit_generator(
            dh.AnyTrainDataGenerator(
                df.iloc[train_idx].index, 
                df.iloc[train_idx], 
                self.steps,
                True,
                self.batch_size, 
                self.input_dims, 
                img_dir,
                file_to_delete
            ),
            validation_data = dh.AnyTrainDataGenerator(
                df.iloc[valid_idx].index, 
                df.iloc[valid_idx], 
                self.steps,
                False,
                self.batch_size, 
                self.input_dims, 
                img_dir,
                file_to_delete
            ),
            epochs = self.epochs,
            verbose=self.verbose,
            use_multiprocessing=False,
            workers=8,
            callbacks=[self.checkpoint, self.reduce_lr, self.e_stopping]
        )