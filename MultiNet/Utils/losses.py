
#description     :Have functions to get optimizer and loss
#usage           :imported in other files
#python_version  :3.5.4
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.models import model_from_json
import os
import itertools
import numpy as np
import pandas as pd
import glob
import math
from numpy import genfromtxt
from keras import optimizers
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn import preprocessing
import time
import keras as keras
from math import sqrt
from sklearn.utils import shuffle
from sklearn import manifold
from keras import backend as K
from sklearn.metrics import mean_squared_error,median_absolute_error
from keras.models import load_model
import timeit
from sklearn.metrics import jaccard_score, classification_report, confusion_matrix
from  skimage.metrics import structural_similarity as ssim
import skimage
# print(skimage.__version__)
from keras import losses
from sklearn.preprocessing import label_binarize


class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
    
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
    
        return K.mean(K.square(model(y_true) - model(y_pred)))

def FuzzyJaccard_distance_loss(y_true, y_pred,  n_channels=1):
 	jac = 0
 	for c in range(n_channels):
            true_batch_c = y_true[:, :, :, c]
            pred_batch_c = y_pred[:, :, :, c]
            intersect_batch = (K.minimum(true_batch_c, pred_batch_c))
            union_batch = (K.maximum(true_batch_c, pred_batch_c))
            intersect = K.sum(K.sum(intersect_batch, axis=-1), axis=-1)
            union = K.sum(K.sum(union_batch, axis=-1), axis=-1)
            j = intersect / union
            jac += K.mean(j)
 	union= jac / n_channels
 	union= (-K.log(K.clip(union, K.epsilon(), None) )) #**0.3
 	return union # (1- union )


def dst_transform(y_true, y_pred):  
    """
    Compute the distance transform of y_true, y_pred 
    Then calculate the error between the distance  (L1 or L2) 
    Usage:
    loss = dst_transform(weights)
    """

    loss=0
    size= K.int_shape(y_true)[0]
    if (size is None):
        size= 1
    for i in range(size):
#         print("y_true {.shape}".format(y_true))

        true_batch_c = K.tf.squeeze(y_true[i, :, :,:])
        pred_batch_c = K.tf.squeeze(y_pred[i, :, :,:])
#         print("true_batch_c {.shape}".format(true_batch_c))

        # ###### dist_transform for y_true#####
        y_true_mask_f = K.cast(K.greater(true_batch_c,(K.mean(true_batch_c)+K.std(true_batch_c))), dtype='float32')
#         print("y_true_mask_f {.shape}".format(y_true_mask_f))

        ones_true = K.tf.where(K.tf.greater_equal(y_true_mask_f,1))
#         print("ones_true {.shape}".format(ones_true))
        zeros_true = K.tf.where(K.tf.equal(y_true_mask_f,0))
#         print("zeros_true {.shape}".format(zeros_true))

        X_true = K.cast(ones_true,dtype="float32")
        Y_true =  K.cast(zeros_true,dtype="float32")
        is_empty_gt = K.tf.equal(K.tf.size(X_true), 0)
        a_true = -2*K.dot( Y_true, K.transpose(X_true))
        b_true  = K.sum(K.square(X_true), axis=1) 
        c_true  =K.tf.expand_dims(K.sum(K.square(Y_true), axis=1), 1)
        sumall_true =a_true +b_true +c_true 
#     return sumall_true#losss/batch_size #K.mean(finalChamferDistanceSum)

        dists_true = K.sqrt(K.min(sumall_true,axis=-1)) # min dist of each zero pixel to one pixel
#     print("dists {.shape}".format(dists_true))
        dist_trans_true= K.zeros ((128,128),dtype='float32')
        shape= (128,128)
        delta_true = K.tf.SparseTensor(zeros_true, dists_true, shape)
        result_t = dist_trans_true + K.tf.sparse.to_dense(delta_true)
        result_true = K.tf.cond(is_empty_gt, lambda: dist_trans_true, lambda: result_t/128)
    
# ###### dist_transform for y_pred#####
        y_pred_mask_f = K.cast(K.greater(pred_batch_c,(K.mean(pred_batch_c)+K.std(pred_batch_c))), dtype='float32')
        ones_pred= K.tf.where(K.tf.greater_equal(y_pred_mask_f,1))
#     print("ones_true {.shape}".format(ones_true))
        zeros_pred= K.tf.where(K.tf.equal(y_pred_mask_f,0))
#     print("zeros_true {.shape}".format(zeros_true))

        X_pred = K.cast(ones_pred,dtype="float32")
        Y_pred =  K.cast(zeros_pred,dtype="float32")
        is_empty_pre = K.tf.equal(K.tf.size(X_pred), 0)
    
        a_pred = -2*K.dot( Y_pred, K.transpose(X_pred))
        b_pred  = K.sum(K.square(X_pred), axis=1) 
        c_pred  =K.tf.expand_dims(K.sum(K.square(Y_pred), axis=1), 1)
        sumall_pred =a_pred +b_pred +c_pred
        dists_pred = K.sqrt(K.min(sumall_pred,axis=-1)) # min dist of each zero pixel to one pixel
#     print("dists {.shape}".format(dists_true))
        dist_trans_pred= K.zeros ((128,128),dtype='float32')
        shape= (128,128)
        delta_pred = K.tf.SparseTensor(zeros_pred, dists_pred, shape)
        result_p = dist_trans_pred + K.tf.sparse.to_dense(delta_pred)
        result_pred = K.tf.cond(is_empty_pre, lambda: dist_trans_pred, lambda: result_p/128)
    
        L1_Distance= K.mean(K.sum(K.abs(result_true - result_pred), axis=-1), axis=-1)/(128)      #, axis=-1)/(128*128*2) #K.sqrt(K.sum(K.square(result_true - result_pred), axis=-1))

        loss += L1_Distance
    return loss/ size


def loss_r(alpha, Beta,batch_size):
    def custom_loss_func_r(y_true, y_pred):
        return custom_loss_r(y_true, y_pred, alpha, Beta,batch_size)
    return custom_loss_func_r

def custom_loss_r(y_true, y_pred, alpha, Beta,batch_size):  
    loss =losses.mean_squared_error(y_true, y_pred)
    loss+= alpha *FuzzyJaccard_distance_loss(y_true, y_pred)
    loss += Beta* dst_transform(y_true, y_pred)

    return  loss


