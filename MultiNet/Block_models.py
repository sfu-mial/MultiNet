# Module blocks for building Multi-freq model

from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add
import keras  as keras
from keras.layers import *
from keras.layers import GlobalAveragePooling2D, Reshape, multiply, Permute
from keras import backend as K
from keras.utils.vis_utils import plot_model
import tensorflow as tf


def squeeze_excite_channel_wise_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='glorot_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='glorot_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def squeeze_excite_spacial_wise_block(input):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    se = Conv2D(filters = 1, kernel_size = 1, strides = 1, padding = "same")(init)
    se = Activation('sigmoid')(se) 

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x
    
def res_block_gen(model, kernal_size, filters, strides):

    rec = model

    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, kernel_initializer='glorot_normal',padding = "same")(model)
    model = Activation('tanh')(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, kernel_initializer='glorot_normal',padding = "same")(model)
    model = Activation('tanh')(model)
    ## squeeze and excite spacial wise block
    model = squeeze_excite_spacial_wise_block(model) # x2
    ## Concurrent Spatial and Channel Squeeze & Excitation model = add([x1, x2])
    model = add([rec, model])

    return model

def reconst_block(input, filters, initializers, shape):
    ## Create an initial image estimate''' 
    model= Dense( 128*128, activation = 'relu', kernel_initializer=initializers)(input)
    model = keras.layers.Reshape(shape)(model)
    ## Shallow convolion over image ''' 
    model = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same")(model)
    model = Activation('relu')(model) 
    ## Deep attention block ''' 
    for index in range(2): #16
        model = res_block_gen(model, 3, 32, 1)
    for index in range(2): #16
        model = res_block_gen(model, 5, 32, 1)
    model= Conv2D(filters = 32, kernel_size = 7, strides = 1, padding = "same",kernel_regularizer=tf.keras.regularizers.L2(0.001))(model)
    model = Activation('relu')(model)
    return model


