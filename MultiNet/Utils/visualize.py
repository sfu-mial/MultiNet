
#description     :Have functions to get optimizer and loss
#usage           :imported in other files
#python_version  :3.5.4
import tensorflow as tf
from Utils.Utils_models import *
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
from keras import losses
from sklearn.preprocessing import label_binarize

global Tmp_ssimlist
Tmp_ssimlist = 0

def plot_generated_images(epoch, dir,generated_image_1 , x_train, GT_label,val =True, examples=120, dim=(1, 2), figsize=(10, 5)):
    fg_color = 'black'
    bg_color =  'white'
    DistanceROI = []
    mselist=[]
    psnrlist=[]
    ssimlist=[]
    Dicelist= []
    FJaccard=[]
    vmin=0
    vmax=25
    scale = np.array(100)  
    # PD_label=[]
    # GT_label=[]
    global Tmp_ssimlist
    dirfile= dir+ '/test_generated_image_'
    if val :
        r= examples
    else: 
        r= len(x_train)
    for index in range(r):
            ## plot GT
            fig=plt.figure(figsize=figsize)
            ax1=plt.subplot(dim[0], dim[1], 1)
            ax1.set_title('GT', color=fg_color)
            imgn = np.flipud(x_train[index])/scale 
            im1 = ax1.imshow(imgn.reshape(128, 128))  
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax1.axis('off')
            fig.colorbar(im1, cax=cax, orientation='vertical')

            ## plot Rec1
            ax2=plt.subplot(dim[0], dim[1], 2)
            imgnr = np.flipud(generated_image_1[index])/scale 
            ax2.set_title('Recons_f1', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax2.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ## plot Rec2
            ax3=plt.subplot(dim[0], dim[1], 3)
            imgnr = np.flipud(generated_image_2[index])/scale 
            ax3.set_title('Recons_f2', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax3.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ## plot Rec3
            ax4=plt.subplot(dim[0], dim[1], 4)
            imgnr = np.flipud(generated_image_3[index])/scale 
            ax4.set_title('Recons_f3', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax4.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ## plot Rec4
            ax5=plt.subplot(dim[0], dim[1], 5)
            imgnr = np.flipud(generated_image_4[index])/scale 
            ax5.set_title('Recons_f4', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))
            divider = make_axes_locatable(ax5)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax5.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            ## plot Rec_fusion
            ax6=plt.subplot(dim[0], dim[1], 6)
            imgnr = np.flipud(generated_image_f[index])/scale 
            ax6.set_title('Recons_all', color=fg_color)
            im2=plt.imshow(imgnr.reshape(128, 128))
            divider = make_axes_locatable(ax6)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax6.axis('off')
            fig.colorbar(im2, cax=cax, orientation='vertical')
            plt.tight_layout(pad=0.01)
            plt.savefig(dirfile+ '-' +str(index)+'.png' )
                
            ## compute metrics
            v=calculateDistance (generated_image_f[index],x_train[index])#
            DistanceROI.append(v)
            p=psnr(generated_image_f[index],x_train[index])
            psnrlist.append(p)
            ss_im = ssim(x_train[index].reshape(128, 128), generated_image_f[index].reshape(128, 128))
            ssimlist.append(ss_im)
            fjacc= FuzzyJaccard(x_train[index],generated_image_f[index])
            FJaccard.append(fjacc)
            plt.close("all")
 
    FJ_mean= np.mean(FJaccard)
    FJ_std= np.std(FJaccard)
    DistanceROI_mean= np.mean(DistanceROI)
    DistanceROI_std= np.std(DistanceROI)
    psnr_mean=np.mean(psnrlist)
    psnr_std=np.std(psnrlist)
    ssim_mean=np.mean(ssimlist)
    ssim_std=np.std(ssimlist)






