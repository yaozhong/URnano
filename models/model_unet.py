# Copyright 2019 
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

from __future__ import division

from keras import Input, models, layers, regularizers
from keras.optimizers import RMSprop,SGD, Adam
from keras import callbacks, losses
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import  Conv2DTranspose, Lambda, Cropping1D, CuDNNGRU
from keras_contrib.layers import CRF

from losses import *


# using the extanding and call the 2dTranspose function.  not work right the extra dimentional is kept
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

#################################################
## loss functions used for training Unet
#################################################a

def create_sparse(ten):
    ten = K.cast(ten,K.floatx())

    n = ten.shape[0]
    ind, values = [], []
    max_len = 0
    
    for xi in K.tf.range(n):
        for yi in K.tf.range(len(ten[xi])):
            ind.append([xi, yi])
            values.append(ten[xi, yi])
        if len(ten[xi]) > max_len:
            max_len = len(ten[xi])

    shape = [n, max_len]

    return K.tf.SparseTensorValue(ind, values, shape)


def getCropShape(target, refer):

    cw = (refer.get_shape()[1] -target.get_shape()[1]).value
    assert (cw >= 0)

    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)

    return (cw1, cw2)


def getCropShape_adj(target, refer, adj):

    cw = -(target.get_shape()[1] - refer.get_shape()[1]).value + adj
    print cw

    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)

    return (cw1, cw2)

"""
# baisc U-net module
"""
def UNet_module(rd_input, kernels, conv_window_len, maxpooling_len, stride=1, BN=True, DropoutRate=0.2):

            initializer = 'glorot_uniform'

            ##################### Conv1 #########################      
            conv1 = layers.Conv1D(kernels[0], conv_window_len, strides= stride , padding='same', \
                kernel_initializer=initializer)(rd_input)
            if BN: conv1 = layers.BatchNormalization()(conv1)
            conv1 = layers.Activation('relu')(conv1)
            
            conv1 = layers.Conv1D(kernels[0], conv_window_len, strides= stride, padding='same', \
                kernel_initializer=initializer)(conv1)
            if BN: conv1 = layers.BatchNormalization()(conv1) 
            conv1 = layers.Activation('relu')(conv1)

            pool1 = layers.MaxPooling1D(maxpooling_len[0])(conv1)
        
            ##################### Conv2 ##########################
            conv2 = layers.Conv1D(kernels[1], 3, strides= stride, padding='same',\
                kernel_initializer=initializer)(pool1)
            if BN: conv2 = layers.BatchNormalization()(conv2) 
            conv2 = layers.Activation('relu')(conv2)
            
            conv2 = layers.Conv1D(kernels[1], 3, strides= stride, padding='same',\
                kernel_initializer=initializer)(conv2)
            if BN: conv2 = layers.BatchNormalization()(conv2) 
            conv2 = layers.Activation('relu')(conv2)

            pool2 = layers.MaxPooling1D(maxpooling_len[1])(conv2)

            ##################### conv3 ###########################
            conv3 = layers.Conv1D(kernels[2], 3, strides= stride,  padding='same',\
                kernel_initializer=initializer)(pool2)
            if BN: conv3 = layers.BatchNormalization()(conv3) 
            conv3 = layers.Activation('relu')(conv3)
            
            conv3 = layers.Conv1D(kernels[2], 3, strides= stride, padding='same',\
                kernel_initializer=initializer)(conv3)
            if BN: conv3 = layers.BatchNormalization()(conv3) 
            conv3 = layers.Activation('relu')(conv3)

            if DropoutRate > 0:
                drop3 = layers.Dropout(DropoutRate)(conv3)
            else:
                drop3 = conv3

            pool3 = layers.MaxPooling1D(maxpooling_len[2])(drop3)


            ####################  conv4 (U bottle) #####################
            conv4 = layers.Conv1D(kernels[3], 3, strides= 1, padding='same',\
                kernel_initializer=initializer)(pool3)
            if BN: conv4 = layers.BatchNormalization()(conv4) 
            conv4 = layers.Activation('relu')(conv4)
            
            conv4 = layers.Conv1D(kernels[3], 3, strides= 1, padding='same',\
                kernel_initializer=initializer)(conv4)
            if BN: conv4 = layers.BatchNormalization()(conv4) 
            conv4 = layers.Activation('relu')(conv4)

            if DropoutRate > 0:
                drop4 = layers.Dropout(DropoutRate)(conv4)
            else:
                drop4 = conv4

            ################### upSampling, upConv5 ##########################
            # up5 = layers.UpSampling1D(maxpooling_len[2])(drop4)
            up5 = Conv1DTranspose(drop4, kernels[2] , 3, strides=maxpooling_len[2], padding='same')

            merge5 = layers.Concatenate(-1)([drop3, up5])

            conv5 = layers.Conv1D(kernels[2], 3, padding='same', \
                kernel_initializer=initializer)(merge5)
            if BN: conv5 = layers.BatchNormalization()(conv5) 
            conv5 = layers.Activation('relu')(conv5)
            
            conv5 = layers.Conv1D(kernels[2], 3, padding='same', \
                kernel_initializer=initializer)(conv5)
            if BN: conv5 = layers.BatchNormalization()(conv5) 
            conv5 = layers.Activation('relu')(conv5)


            ################### upConv 6 ##############################
            #up6 = layers.UpSampling1D(maxpooling_len[1])(conv5)
            up6 = Conv1DTranspose(conv5, kernels[1] , 3, strides=maxpooling_len[1], padding='same')
            
            merge6 = layers.Concatenate(-1)([conv2, up6])
        
            conv6 = layers.Conv1D(kernels[1], 3, padding='same', \
                kernel_initializer=initializer)(merge6)
            if BN: conv6 = layers.BatchNormalization()(conv6) 
            conv6 = layers.Activation('relu')(conv6)
            
            conv6 = layers.Conv1D(kernels[1], 3, padding='same',\
                kernel_initializer=initializer)(conv6)
            if BN: conv6 = layers.BatchNormalization()(conv6) 
            conv6 = layers.Activation('relu')(conv6)


            ################### upConv 7 #########################
            #up7 = layers.UpSampling1D(maxpooling_len[0])(conv6)
            up7 = Conv1DTranspose(conv6, kernels[0] , 3, strides=maxpooling_len[0], padding='same')
            
            merge7 = layers.Concatenate(-1)([conv1, up7])

            conv7 = layers.Conv1D(kernels[0], 3, padding='same',\
                kernel_initializer=initializer)(merge7)
            if BN: conv7 = layers.BatchNormalization()(conv7) 
            conv7 = layers.Activation('relu')(conv7)
            
            conv7 = layers.Conv1D(kernels[0], 3, padding='same', \
                kernel_initializer=initializer)(conv7)
            if BN: conv7 = layers.BatchNormalization()(conv7) 
            conv7 = layers.Activation('relu')(conv7)

            ################## final output ###################### 
            conv8 = layers.Conv1D(kernels[0], 3, padding='same', \
                kernel_initializer=initializer)(conv7)
            if BN: conv8 = layers.BatchNormalization()(conv8) 
            conv8 = layers.Activation('relu')(conv8)
            
            if DropoutRate > 0:
                conv8 = layers.Dropout(DropoutRate)(conv8)

            return conv8


def UNet_module_test(rd_input, kernels, conv_window_len, maxpooling_len, stride=1, BN=True, DropoutRate=0.2):

            initializer = 'glorot_uniform'

            ##################### Conv1 #########################      
            conv1_0 = layers.SeparableConv1D(int(kernels[0]/4), 1, strides= stride , padding='same', \
                kernel_initializer=initializer)(rd_input)
            if BN: conv1_0 = layers.BatchNormalization()(conv1_0)
            conv1_0 = layers.Activation('relu')(conv1_0)

            conv1_1 = layers.SeparableConv1D(int(kernels[0]/4), 3, strides= stride , padding='same', \
                kernel_initializer=initializer)(rd_input)
            if BN: conv1_1 = layers.BatchNormalization()(conv1_1)
            conv1_1 = layers.Activation('relu')(conv1_1)
            
            conv1_2 = layers.SeparableConv1D(int(kernels[0]/4), 7, strides= stride, padding='same', \
                kernel_initializer=initializer)(rd_input)
            if BN: conv1_2 = layers.BatchNormalization()(conv1_2) 
            conv1_2 = layers.Activation('relu')(conv1_2)

            conv1_3 = layers.SeparableConv1D(int(kernels[0]/4), 11, strides= stride, padding='same', \
                kernel_initializer=initializer)(rd_input)
            if BN: conv1_3 = layers.BatchNormalization()(conv1_3) 
            conv1_3 = layers.Activation('relu')(conv1_3)

            conv1_4 = layers.SeparableConv1D(int(kernels[0]/4), 15, strides= stride, padding='same', \
                kernel_initializer=initializer)(rd_input)
            if BN: conv1_4 = layers.BatchNormalization()(conv1_4) 
            conv1_4 = layers.Activation('relu')(conv1_4)

            conv1 = layers.Concatenate(-1)([conv1_0, conv1_1,conv1_2,conv1_3,conv1_4])
            pool1 = layers.MaxPooling1D(maxpooling_len[0])(conv1)
        
            ##################### Conv2 ##########################
            conv2 = layers.Conv1D(kernels[1], 3, strides= stride, padding='same',\
                kernel_initializer=initializer)(pool1)
            if BN: conv2 = layers.BatchNormalization()(conv2) 
            conv2 = layers.Activation('relu')(conv2)
            
            conv2 = layers.Conv1D(kernels[1], 3, strides= stride, padding='same',\
                kernel_initializer=initializer)(conv2)
            if BN: conv2 = layers.BatchNormalization()(conv2) 
            conv2 = layers.Activation('relu')(conv2)

            pool2 = layers.MaxPooling1D(maxpooling_len[1])(conv2)

            ##################### conv3 ###########################
            conv3 = layers.Conv1D(kernels[2], 3, strides= stride,  padding='same',\
                kernel_initializer=initializer)(pool2)
            if BN: conv3 = layers.BatchNormalization()(conv3) 
            conv3 = layers.Activation('relu')(conv3)
            
            conv3 = layers.Conv1D(kernels[2], 3, strides= stride, padding='same',\
                kernel_initializer=initializer)(conv3)
            if BN: conv3 = layers.BatchNormalization()(conv3) 
            conv3 = layers.Activation('relu')(conv3)

            if DropoutRate > 0:
                drop3 = layers.Dropout(DropoutRate)(conv3)
            else:
                drop3 = conv3

            pool3 = layers.MaxPooling1D(maxpooling_len[2])(drop3)


            ####################  conv4 (U bottle) #####################
            conv4 = layers.Conv1D(kernels[3], 3, strides= 1, padding='same',\
                kernel_initializer=initializer)(pool3)
            if BN: conv4 = layers.BatchNormalization()(conv4) 
            conv4 = layers.Activation('relu')(conv4)
            
            conv4 = layers.Conv1D(kernels[3], 3, strides= 1, padding='same',\
                kernel_initializer=initializer)(conv4)
            if BN: conv4 = layers.BatchNormalization()(conv4) 
            conv4 = layers.Activation('relu')(conv4)

            if DropoutRate > 0:
                drop4 = layers.Dropout(DropoutRate)(conv4)
            else:
                drop4 = conv4

            ################### upSampling, upConv5 ##########################
            # up5 = layers.UpSampling1D(maxpooling_len[2])(drop4)
            up5 = Conv1DTranspose(drop4, kernels[2] , 3, strides=maxpooling_len[2], padding='same')

            merge5 = layers.Concatenate(-1)([drop3, up5])

            conv5 = layers.Conv1D(kernels[2], 3, padding='same', \
                kernel_initializer=initializer)(merge5)
            if BN: conv5 = layers.BatchNormalization()(conv5) 
            conv5 = layers.Activation('relu')(conv5)
            
            conv5 = layers.Conv1D(kernels[2], 3, padding='same', \
                kernel_initializer=initializer)(conv5)
            if BN: conv5 = layers.BatchNormalization()(conv5) 
            conv5 = layers.Activation('relu')(conv5)


            ################### upConv 6 ##############################
            #up6 = layers.UpSampling1D(maxpooling_len[1])(conv5)
            up6 = Conv1DTranspose(conv5, kernels[1] , 3, strides=maxpooling_len[1], padding='same')
            
            merge6 = layers.Concatenate(-1)([conv2, up6])
        
            conv6 = layers.Conv1D(kernels[1], 3, padding='same', \
                kernel_initializer=initializer)(merge6)
            if BN: conv6 = layers.BatchNormalization()(conv6) 
            conv6 = layers.Activation('relu')(conv6)
            
            conv6 = layers.Conv1D(kernels[1], 3, padding='same',\
                kernel_initializer=initializer)(conv6)
            if BN: conv6 = layers.BatchNormalization()(conv6) 
            conv6 = layers.Activation('relu')(conv6)


            ################### upConv 7 #########################
            #up7 = layers.UpSampling1D(maxpooling_len[0])(conv6)
            up7 = Conv1DTranspose(conv6, kernels[0] , 3, strides=maxpooling_len[0], padding='same')
            
            merge7 = layers.Concatenate(-1)([conv1, up7])

            conv7 = layers.Conv1D(kernels[0], 3, padding='same',\
                kernel_initializer=initializer)(merge7)
            if BN: conv7 = layers.BatchNormalization()(conv7) 
            conv7 = layers.Activation('relu')(conv7)
            
            conv7 = layers.Conv1D(kernels[0], 3, padding='same', \
                kernel_initializer=initializer)(conv7)
            if BN: conv7 = layers.BatchNormalization()(conv7) 
            conv7 = layers.Activation('relu')(conv7)

            ################## final output ###################### 
            conv8 = layers.Conv1D(kernels[0], 3, padding='same', \
                kernel_initializer=initializer)(conv7)
            if BN: conv8 = layers.BatchNormalization()(conv8) 
            conv8 = layers.Activation('relu')(conv8)
            
            if DropoutRate > 0:
                conv8 = layers.Dropout(DropoutRate)(conv8)

            return conv8


def UNet_only(rd_input, kernels, conv_window_len, maxpooling_len,stride, BN=True, DropoutRate=0.2):

            #kernels =[8, 16, 32, 64] # kernels =[64, 128, 256, 512]
            #stride=1

            unet_module_output = UNet_module(rd_input, kernels, conv_window_len, maxpooling_len, stride, BN, DropoutRate)
            
            conv9 = layers.Conv1D(8, 1, activation='softmax')(unet_module_output)
            model = models.Model(rd_input, conv9)

            return model


def UNet_GRU3(rd_input, kernels, conv_window_len, maxpooling_len, stride, BN=True, DropoutRate=0.2):

            unet_module_output = UNet_module(rd_input, kernels, conv_window_len, maxpooling_len, stride, BN, DropoutRate)
    
            output = layers.Bidirectional(layers.CuDNNGRU(32, return_sequences=True))(unet_module_output)
            output = layers.Bidirectional(layers.CuDNNGRU(32, return_sequences=True))(output)
            output = layers.Bidirectional(layers.CuDNNGRU(32, return_sequences=True))(output)
            output = layers.TimeDistributed(layers.Dense(8, activation="softmax"))(output)

            model = models.Model(rd_input, output)

            return model


def UNet_GRU2(rd_input, kernels, conv_window_len, maxpooling_len, stride, BN=True, DropoutRate=0.2):

            #stride=1
            #kernels =[64, 128, 256, 512]

            unet_module_output = UNet_module(rd_input, kernels, conv_window_len, maxpooling_len, stride, BN, DropoutRate)
    
            output = layers.Bidirectional(layers.CuDNNGRU(32, return_sequences=True))(unet_module_output)
            output = layers.Bidirectional(layers.CuDNNGRU(32, return_sequences=True))(output)
            output = layers.TimeDistributed(layers.Dense(8, activation="softmax"))(output)

            model = models.Model(rd_input, output)

            return model

def UNet_GRU1(rd_input, kernels, conv_window_len, maxpooling_len, stride, BN=True, DropoutRate=0.2):

            #stride=1
            #kernels =[64, 128, 256, 512]
            unet_module_output = UNet_module(rd_input, kernels, conv_window_len, maxpooling_len, stride, BN, DropoutRate)
    
            output = layers.Bidirectional(layers.CuDNNGRU(32, return_sequences=True))(unet_module_output)
            output = layers.TimeDistributed(layers.Dense(8, activation="softmax"))(output)

            model = models.Model(rd_input, output)

            return model



### testing the performance of the last 3GRU layers only
def GRU3_solo(rd_input, kernels, conv_window_len, maxpooling_len, stride, BN=True, DropoutRate=0.2):

    initializer = 'glorot_uniform'
    conv1 = layers.Conv1D(kernels[0], 1, strides= stride , padding='same', \
        kernel_initializer=initializer)(rd_input)

    output = layers.Bidirectional(layers.CuDNNGRU(32, return_sequences=True))(conv1)
    output = layers.Bidirectional(layers.CuDNNGRU(32, return_sequences=True))(output)
    output = layers.Bidirectional(layers.CuDNNGRU(32, return_sequences=True))(output)
    output = layers.TimeDistributed(layers.Dense(8, activation="softmax"))(output)

    model = models.Model(rd_input, output)
    return model

def URNet(rd_input, kernels, conv_window_len, maxpooling_len, stride, BN=True, DropoutRate=0.2):
            
            initializer = 'glorot_uniform' #
            #stride=1
            #kernels =[64, 128, 256, 512]
            #kernels =[8, 16, 32, 64]
       
            ##################### Conv1 #########################      
            conv1 = layers.Conv1D(kernels[0], conv_window_len, strides= stride , padding='same', \
                kernel_initializer=initializer)(rd_input)
            if BN: conv1 = layers.BatchNormalization()(conv1)
            conv1 = layers.Activation('relu')(conv1)
            
            conv1 = layers.Conv1D(kernels[0], conv_window_len, strides= stride, padding='same', \
                kernel_initializer=initializer)(conv1)
            if BN: conv1 = layers.BatchNormalization()(conv1) 
            conv1 = layers.Activation('relu')(conv1)

            rnn1 = layers.CuDNNGRU(kernels[0], return_sequences=True)(conv1)
            # use it for the next stage
            pool1 = layers.MaxPooling1D(maxpooling_len[0])(rnn1)
        
            ##################### Conv2 ##########################
            conv2 = layers.Conv1D(kernels[1], 3, strides= stride, padding='same',\
                kernel_initializer=initializer)(pool1)
            if BN: conv2 = layers.BatchNormalization()(conv2) 
            conv2 = layers.Activation('relu')(conv2)
            
            conv2 = layers.Conv1D(kernels[1], 3, strides= stride, padding='same',\
                kernel_initializer=initializer)(conv2)
            if BN: conv2 = layers.BatchNormalization()(conv2) 
            conv2 = layers.Activation('relu')(conv2)

            rnn2 = layers.CuDNNGRU(kernels[1], return_sequences=True)(conv2)
            #use it for the next stage
            pool2 = layers.MaxPooling1D(maxpooling_len[1])(rnn2)

            ##################### conv3 ###########################
            conv3 = layers.Conv1D(kernels[2], 3, strides= stride,  padding='same',\
                kernel_initializer=initializer)(pool2)
            if BN: conv3 = layers.BatchNormalization()(conv3) 
            conv3 = layers.Activation('relu')(conv3)
            
            conv3 = layers.Conv1D(kernels[2], 3, strides= stride, padding='same',\
                kernel_initializer=initializer)(conv3)
            if BN: conv3 = layers.BatchNormalization()(conv3) 
            conv3 = layers.Activation('relu')(conv3)

            if DropoutRate > 0:
                drop3 = layers.Dropout(DropoutRate)(conv3)
            else:
                drop3 = conv3

            rnn3 = layers.CuDNNGRU(kernels[2], return_sequences=True)(drop3)
            # use it for the next stage
            pool3 = layers.MaxPooling1D(maxpooling_len[2])(rnn3)


            ####################  conv4 (U bottle) #####################
            conv4 = layers.Conv1D(kernels[3], 3, strides= 1, padding='same',\
                kernel_initializer=initializer)(pool3)
            if BN: conv4 = layers.BatchNormalization()(conv4) 
            conv4 = layers.Activation('relu')(conv4)
            
            conv4 = layers.Conv1D(kernels[3], 3, strides= 1, padding='same',\
                kernel_initializer=initializer)(conv4)
            if BN: conv4 = layers.BatchNormalization()(conv4) 
            conv4 = layers.Activation('relu')(conv4)

            if DropoutRate > 0:
                drop4 = layers.Dropout(DropoutRate)(conv4)
            else:
                drop4 = conv4

            ################### upSampling, upConv5 ##########################
            # up5 = layers.UpSampling1D(maxpooling_len[2])(drop4)
            up5 = Conv1DTranspose(drop4, kernels[2] , 3, strides=maxpooling_len[2], padding='same')

            merge5 = layers.Concatenate(-1)([drop3, up5])
            merge5 = layers.Concatenate(-1)([merge5, rnn3])

            conv5 = layers.Conv1D(kernels[2], 3, padding='same', \
                kernel_initializer=initializer)(merge5)
            if BN: conv5 = layers.BatchNormalization()(conv5) 
            conv5 = layers.Activation('relu')(conv5)
            
            conv5 = layers.Conv1D(kernels[2], 3, padding='same', \
                kernel_initializer=initializer)(conv5)
            if BN: conv5 = layers.BatchNormalization()(conv5) 
            conv5 = layers.Activation('relu')(conv5)


            ################### upConv 6 ##############################
            #up6 = layers.UpSampling1D(maxpooling_len[1])(conv5)
            up6 = Conv1DTranspose(conv5, kernels[1] , 3, strides=maxpooling_len[1], padding='same')
            

            merge6 = layers.Concatenate(-1)([conv2, up6])
            merge6 = layers.Concatenate(-1)([merge6, rnn2])
        
            conv6 = layers.Conv1D(kernels[1], 3, padding='same', \
                kernel_initializer=initializer)(merge6)
            if BN: conv6 = layers.BatchNormalization()(conv6) 
            conv6 = layers.Activation('relu')(conv6)
            
            conv6 = layers.Conv1D(kernels[1], 3, padding='same',\
                kernel_initializer=initializer)(conv6)
            if BN: conv6 = layers.BatchNormalization()(conv6) 
            conv6 = layers.Activation('relu')(conv6)


            ################### upConv 7 #########################
            #up7 = layers.UpSampling1D(maxpooling_len[0])(conv6)
            up7 = Conv1DTranspose(conv6, kernels[0] , 3, strides=maxpooling_len[0], padding='same')
            
            merge7 = layers.Concatenate(-1)([conv1, up7])
            merge7 = layers.Concatenate(-1)([merge7, rnn1])

            conv7 = layers.Conv1D(kernels[0], 3, padding='same',\
                kernel_initializer=initializer)(merge7)
            if BN: conv7 = layers.BatchNormalization()(conv7) 
            conv7 = layers.Activation('relu')(conv7)
            
            conv7 = layers.Conv1D(kernels[0], 3, padding='same', \
                kernel_initializer=initializer)(conv7)
            if BN: conv7 = layers.BatchNormalization()(conv7) 
            conv7 = layers.Activation('relu')(conv7)

            ################## final output ###################### 
            conv8 = layers.Conv1D(kernels[0], 3, padding='same', \
                kernel_initializer=initializer)(conv7)
            if BN: conv8 = layers.BatchNormalization()(conv8) 
            conv8 = layers.Activation('relu')(conv8)
            
            if DropoutRate > 0:
                conv8 = layers.Dropout(DropoutRate)(conv8)

            output = conv8
            output = layers.Bidirectional(layers.CuDNNGRU(64, return_sequences=True))(conv8)
            output = layers.Bidirectional(layers.CuDNNGRU(64, return_sequences=True))(output)
            output = layers.Bidirectional(layers.CuDNNGRU(64, return_sequences=True))(output)
            output = layers.TimeDistributed(layers.Dense(8, activation="softmax"))(output)
                
            #conv9 = layers.Conv1D(8, 1, activation='softmax')(conv8)
            model = models.Model(rd_input, output)

            return model
