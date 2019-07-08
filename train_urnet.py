# Copyright 2019 
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import argparse, sys, os, time, json

import datetime
import tensorflow as tf

from data_input import *
from utility import *
from six.moves import range

from models.model_unet import *
from keras.utils import plot_model

import h5py
import numpy as np

from unet_hyperOpt import *

CB = [callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="auto", restore_best_weights=True)]

def train(args, modelSavePath="../experiment/model/", curveMonitor=True):

    lossType = args.loss

    print("\n"+"-"*50)
    print("Start training nanopore base-calling model")
    print("-"*50)

    print("@ Loading Data ... ")
    trainX, seqLen, label, label_vec, label_seg, label_raw, label_vec_new = loading_data((args.data_dir, args.train_cache), args.cacheFile)
    label_seg = np.array(label_seg)

    if args.fSignal > 0:
        print("@ Condudct filtering of outline singnals ... ")
        Idx = signalFiltering(trainX, args.fSignal)
        trainX, label_vec_new = trainX[Idx], label_vec_new[Idx]
        label_seg = label_seg[Idx]

    if args.norm != "":
        print("@ Perform global data normalization ... ")
        meanX, stdX = np.mean(trainX), np.std(trainX)
        trainX = (trainX - meanX)/stdX
        print("*** Data statistics:mean=%f, std=%f" %(meanX, stdX))

        # save the statistics of training for the testing. 
        stat_dict={"m":meanX, "s":stdX}
        pickle_out = open(args.norm, "wb")
        pickle.dump(stat_dict, pickle_out)
        pickle_out.close()
    else:
        # performa local data normalization
        print("@ Perform local data normalization ... ")
        trainX = independ_sample_norm(trainX)

    print("@ Data reshaping ...")
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1).astype("float32")
    trainY = to_categorical(label_vec_new, num_classes=8)

    # signal augmenation perofmrance here, currently simply replace the original input to avoid 2d processing. 
    ## solo usage will reduce performance. 
    if(args.inputAug_winLen > 0):
        trainX = signalAug(trainX, args.inputAug_winLen)

    print(">>Loaded data shape:")
    print(trainX.shape) # fixed length
    print(trainY.shape) # could be padding with extra 0 in the end part. 

    ###############################################
    # loading model parameters
    ###############################################
    if args.model_param != "":
        params = load_modelParam(args.model_param)
    else:
        # save the number of hyper-parameters
        model_param_file = modelSavePath + "/hpt.11.URnet.model.parameters.json"
        if not os.path.exists(model_param_file):
            print("@ Start hyperParameters training...\n %s" %(model_param_file))
            tmpData = (trainX, trainY)
            do_hyperOpt(tmpData, 20, model_param_file)
            params = load_modelParam(model_param_file)

    print("@ Loaded model parameters are :")
    print(params)
    model_name = get_unet_model_name(params, args)

    signals = Input(name='input', shape=[trainX.shape[1], 1], dtype=np.float32)

    if args.lstm == 0:
        print("** START module of UNet")
        model = UNet_only(signals, params["kernel_size"], params["conv_window_len"],  params["maxpooling_len"],  \
            params["stride"], True, params["DropoutRate"])  

    elif args.lstm == 3:
        print("** START module of UNet-GRU3")
        model = UNet_GRU3(signals, params["kernel_size"], params["conv_window_len"],  params["maxpooling_len"],  \
            params["stride"], True, params["DropoutRate"])
        #plot_model(model, to_file='../experiment/devLOG/figures/UNet_GRU3_model.png')

    elif args.lstm == 10:
        print("** START conv1-GRU3-solo ...")
        model = GRU3_solo(signals, params["kernel_size"], params["conv_window_len"],  params["maxpooling_len"],  \
            params["stride"], True, params["DropoutRate"])


    elif args.lstm == 12:
        print("** START UR-net ...")
        model = UNet_LSTM_MIX(signals, params["kernel_size"], params["conv_window_len"],  params["maxpooling_len"],  \
            params["stride"], True, params["DropoutRate"])
        #plot_model(model, to_file='../experiment/devLOG/figures/MIX_UNet_RNN_model.png')


    # different loss function, loading ...
    if lossType == "dice_loss":       
        print("Training using dice_loss ...")
        model.compile(optimizer=Adam(params["lr"]), loss = dice_coef_loss , metrics=[ dice_coef_loss, metrics.categorical_accuracy ])
    elif lossType == "categorical_loss":
        model.compile(optimizer=Adam(params["lr"]), loss = 'categorical_crossentropy' , metrics=[metrics.categorical_accuracy])
    elif lossType == "bce_dice_loss":
        model.compile(optimizer=Adam(params["lr"]), loss = bce_dice_loss , metrics=[ bce_dice_loss, metrics.categorical_accuracy ])
    ## add the 20190607  
    elif lossType == "categorical_focal_loss":
        model.compile(optimizer=Adam(params["lr"]), loss = [categorical_focal_loss(alpha=.25, gamma=2)] ,\
            metrics=[ metrics.categorical_accuracy ])
    elif lossType == "ce_dice_loss":
        model.compile(optimizer=Adam(params["lr"]), loss = ce_dice_loss , metrics=[ metrics.categorical_accuracy ])
    
    vsplit = 0
    print("@ data valdiating [%f]" %(vsplit))

    # continut the training
    if args.contrain >  0:
        del model
        model_name = get_unet_model_name(params, args)
        print("@ Contintue training the model %s" %(model_name))
        model = models.load_model(modelSavePath+ "/weights/" + model_name+ ".h5", \
            custom_objects={'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef, 'bce_dice_loss': bce_dice_loss, \
            'categorical_focal_loss_fixed':categorical_focal_loss(gamma=2., alpha=.25), \
            'ce_dice_loss': ce_dice_loss})

        history = model.fit(trainX, trainY, epochs=args.contrain, batch_size=params["batchSize"], verbose=1, callbacks=CB, validation_split=vsplit)
    else:
        history = model.fit(trainX, trainY, epochs=params["epoch"], batch_size=params["batchSize"], verbose=1, callbacks=CB, validation_split=vsplit)
    
    print("@ Saving model ...")  
    model.save(modelSavePath + "/weights/" + model_name + ("_cont-" + str(args.contrain)  if args.contrain > 0 else "") + ".h5")
    
    timeTag = datetime.datetime.now()

    if curveMonitor == True and vsplit > 0:

        fig=plt.figure()
        figureSavePath="../experiment/devLOG/train_curve/"+ timeTag.strftime("%Y%m%d")  + "-" + model_name + ".png"
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Training curve of the whole training set " + lossType)
        plt.savefig(figureSavePath)
        plt.close("all")


def run(args):
    global FLAGS

    FLAGS = args
    FLAGS.data_dir = FLAGS.data_dir + os.path.sep
    FLAGS.log_dir = FLAGS.log_dir + os.path.sep

    ## training the model
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training model with tfrecord file')

    parser.add_argument('-g', '--gpu', default='0', help="Assigned GPU for running the code")

    parser.add_argument('-i', '--data_dir', default="/data/workspace/nanopore/data/chiron_data/train/" ,required= False,
                        help="Directory that store the tfrecord files.")

    #parser.add_argument('-o', '--log_dir', default="model/", required = False  ,
    #                    help="log directory that store the training model.")

    #parser.add_argument('-m', '--model_name', default="devTest", required = False,
    #                    help='model_name')

    parser.add_argument('-v', '--validation', default = None, 
                        help="validation tfrecord file, default is None, which conduct no validation")

    parser.add_argument('-f', '--tfrecord', default="train.tfrecords",
                        help='tfrecord file')

    parser.add_argument('--train_cache', default=None, help="Cache file for training dataset.")

    parser.add_argument('--valid_cache', default=None, help="Cache file for validation dataset.")

    parser.add_argument('-s', '--segment_len', type=int, default=300,
                        help='the length of sequence')

    parser.add_argument('-b', '--batch_size', type=int, default=400,
                        help='Batch size')
    
    parser.add_argument('-t', '--step_rate', type=float, default=1e-2,
                        help='Step rate')

    parser.add_argument('-x', '--max_steps', type=int, default=10000,
                        help='Maximum step')

    parser.add_argument('-n', '--segments_num', type = int, default = 20000,
                        help='Maximum number of segments read into the training queue, default(None) read all segments.')

    #parser.add_argument('--configure', default = None,
    #                    help="Model structure configure json file.")

    parser.add_argument('-k', '--k_mer', default=1, help='Output k-mer size')

    parser.add_argument('--retrain', dest='retrain', action='store_true',
                        help='Set retrain to true')

    parser.add_argument('--read_cache',dest='read_cache',action='store_true',
                        help="Read from cached hdf5 file.")

    parser.add_argument('-l', '--loss', default="", required=True, help="loss function used to learn the segmentation model.")
    parser.add_argument('-cf', '--cacheFile', default="", required=True, help="Assigned cache files.")
    parser.add_argument('-mp', '--model_param', default="", required=True, help="loss function used to learn the segmentation model.")
    parser.add_argument('-lstm', '--lstm', default=0, type=int, help="loss function used to learn the segmentation model.")
    parser.add_argument('-fSignal', '--fSignal', default=0, type=int,  help="Extrem signals of data to determine whethe kept.")
    parser.add_argument('-norm', '--norm', default="",type=str,  help="Training data statistics of saved file")
    parser.add_argument('-tag', '--tag', default="",type=str,  help="Model tag information.")
    parser.add_argument('-iaw', '--inputAug_winLen', default=0,type=int,  help="input Signal augmentation with the windowScreen variance detection.")

    # loading model for initailization
    parser.add_argument('-cont', '--contrain', default=0,type=int,  help="Loading already training model to contintue the training process.")


    parser.set_defaults(retrain=False)
    args = parser.parse_args(sys.argv[1:])
    
    if args.train_cache is None:
        args.train_cache = args.data_dir + '/train_cache_gplabel.hdf5'
    if (args.valid_cache is None) and (args.validation is not None):
        args.valid_cache = args.data_dir + '/valid_cache_gplabel.hdf5'
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    train(args)

