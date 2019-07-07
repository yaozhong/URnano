#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# 20190116
# UNet for processing the same amount of data as chiron, revised by Yao-zhong@imsut
# 20190220 re-crafted U-net training
# 20160615 change to the 11 coding system

from __future__ import absolute_import, division
from __future__ import print_function
import argparse, sys, os, time, json

from distutils.dir_util import copy_tree

import tensorflow as tf

from six.moves import range

from keras import Input, models, layers, regularizers, metrics
from keras import callbacks, losses
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Lambda

from models.model_unet import *
from models.model_cnn import *
from unet_hyperOpt import *

from utility import *
from data_input_new import *

import h5py
import numpy as np
from scipy.stats import pearsonr

##############################################################
def get_pred_ed_list(X, Y, model, plotFigurePath=""):

    preds = model.predict(X, verbose=0)
    golds = Y
    eds, prs = [], []

    for i in range(X.shape[0]):

        logits = np.argmax(preds[i], -1)
        # nc+count converage.
        pred_2d, pred_seg = seqCompact(logits) 
        # A_n to the original label of [A]*n
        pred = getOrigin_from_label2Dseq_11(pred_2d)
        pred = validPredLabel(pred)

        label_2d, label_seg = seqCompact(golds[i])
        gold = getOrigin_from_label2Dseq_11(label_2d)
        gold = validPredLabel(gold)

        # 20190613 revision the evaluation of the last one
        ## original
        ##ed =editDistance(pred, gold)
        ed =editDistance(pred[:-1], gold[:-1])
        eds.append(ed/(len(gold)-1))   # correct the bug 20190624

        # cacluate the perason correlation, equal length is required

        # if len(pred_seg) > len(label_seg):
        #     pr, _ = pearsonr(pred_seg[:-1], np.pad(label_seg[:-1], (0,len(pred_seg)-len(label_seg)),'constant'))
        # else:
        #     pr, _ = pearsonr(np.pad(pred_seg[:-1],(0, len(label_seg)-len(pred_seg)),'constant'), label_seg[:-1])

        # ## perason score need to be rechecked        
        # if(not np.isnan(pr)):
        #     prs.append(pr)
        # else:
        #     print(X[i])
        #     print(pred_seg)
        #     print(label_seg)

        ############################ IOU calucation for segments ########################
        ## processing the non-even issues
        # pred_split = [0] + np.cumsum(pred_seg[:-1])
        # gold_split = [0] + np.cumsum(gold_seg[:-1])



        # cacluate the dice score with labels.
        if plotFigurePath != "":
            print("-------[gold]---------")
            print(gold)
            print("-------[pred]---------")
            print(pred)
            print("Pearson correlation of segment=%f\n" %(pr))

            # save the read singal files for Arda to generate sequences.
            saveName = plotFigurePath +"/_seq_" +"" +"".join(gold) +".signal"
            with open(saveName, 'w') as fs:
                 np.savetxt(fs, X[i])
            
            vis_prediction(X[i], label_seg, nc_from2dList_11(label_2d), pred_seg, nc_from2dList_11(pred_2d), plotFigurePath +"/" +"".join(gold), True)
            
    return eds, prs


#######################################################################
def test_unet(args, modelSavePath="../experiment/model/",norm=False):

    if(args.plot_figure !="" and not os.path.exists(args.plot_figure)):
        os.mkdir(args.plot_figure)
	
    lossType = args.loss
    fileConfig = (args.data_dir, args.test_cache)

    # loading from cache data
    print("\n------START Testing of UNet model----------")
    print("@ Loading data ...")
    X, seqLen, label, label_vec, label_seg, label_raw, label_vec_new = loading_data(fileConfig, args.cacheFile)

    """ not use in the test case, for the filtering, but can give warning."""
    
    if args.fSignal > 0:
        print("@ Condudct filtering of outline singnals ... ")
        Idx = signalFiltering(X, args.fSignal)
        X, label_vec_new = X[Idx], label_vec_new[Idx]
    

    if args.norm != "":
        print("@ Perform data normalization ... ")
        print("- Loading form %s" %(args.norm))

        pickle_in = open(args.norm,"rb")
        stat_dict = pickle.load(pickle_in)
        print("- Training Data statistics m=%f, s=%f" %(stat_dict["m"], stat_dict["s"]))
        print("- [Ref]: test data itself statistics (not used in normalization) m=%f, s=%f" %(np.mean(X), np.std(X)))     
        X = (X - stat_dict["m"])/(stat_dict["s"])
    else:
        print("@ **$$** Perform local normalization for the signal ...")
        X = independ_sample_norm(X)


    label_seq, noused_count = seqCompact(label_vec_new[0])
    gold = getOrigin_from_label2Dseq_11(label_seq)
    gold = validPredLabel(gold)
    
    X = X.reshape(X.shape[0], X.shape[1], 1).astype("float32")
    Y = to_categorical(label_vec_new, num_classes=8)

    print("@ Loaded data scales are:")
    print(X.shape) # fixed length
    print(Y.shape) # could be padding with extra 0 in the end part. 


    if args.model_param != "":
        params = load_modelParam(args.model_param)
    else:
        print("! Unable to load the model parameters, pls check!")
        exit()

    # basic model parameters
    model_name = get_unet_model_name(params, args)
            
    model = models.load_model(modelSavePath+ "/weights/" + model_name+ ("_cont-" + str(args.contrain)  if args.contrain > 0 else "") +".h5", \
        custom_objects={'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef, 'bce_dice_loss': bce_dice_loss, \
        'categorical_focal_loss_fixed':categorical_focal_loss(gamma=2., alpha=.25), \
        'ce_dice_loss': ce_dice_loss})

    pred_batch = 10000
    num_test = X.shape[0]
    epochs = int(num_test/pred_batch)
    eds, prs = [], []

    if epochs > 0:
        for i in range(epochs):
            dataX = X[i*pred_batch:(i+1)*pred_batch]
            goldY = label_vec_new[i*pred_batch:(i+1)*pred_batch]
            eds_tmp, prs_tmp = get_pred_ed_list(dataX, goldY, model, args.plot_figure)        
            eds.extend(eds_tmp)
            prs.extend(prs_tmp)            
            
    dataX = X[epochs*pred_batch:]
    goldY = label_vec_new[epochs*pred_batch:]

    eds_tmp, prs_tmp = get_pred_ed_list(dataX, goldY, model, args.plot_figure)
    eds.extend(eds_tmp)
    prs.extend(prs_tmp)

    print("** Averaged edit distance for the U-net model %d samples is: %f ±(%f)" %(len(eds), np.mean(eds), np.std(eds))) 
    print("** Averaged Pearson for the U-net model %d samples is: %f ±(%f)" %(len(prs), np.mean(prs), np.std(prs))) 


## functions to calcuate the confusion matrix
def test_confustion_matrix(args, modelSavePath="../experiment/model/", norm=False):

    lossType = args.loss
    fileConfig = (args.data_dir, args.test_cache)

    X, seqLen, label, label_vec, label_seg, label_raw, label_vec_new = loading_data(fileConfig, args.cacheFile)

    if args.norm != "":
        print("@ Perform data normalization ... ")
        print("- Loading form %s" %(args.norm))

        pickle_in = open(args.norm,"rb")
        stat_dict = pickle.load(pickle_in)
        print("- Training Data statistics m=%f, s=%f" %(stat_dict["m"], stat_dict["s"]))
        print("- [Ref]: test data itself statistics (not used in normalization) m=%f, s=%f" %(np.mean(X), np.std(X)))     
        X = (X - stat_dict["m"])/(stat_dict["s"])
    else:
        print("@ **$$** Perform local normalization for the signal ...")
        X = independ_sample_norm(X)

    label_seq, noused_count = seqCompact(label_vec_new[0])
    gold = getOrigin_from_label2Dseq_11(label_seq)
    gold = validPredLabel(gold)
    
    X = X.reshape(X.shape[0], X.shape[1], 1).astype("float32")
    Y = to_categorical(label_vec_new, num_classes=8)

    if norm == True:
        print("-------------------------------")
        meanX, stdX = np.mean(X), np.std(X)
        X = (X - meanX)/stdX
        print("*** Data statistics:mean=%f, std=%f" %(meanX, stdX))
        print("-------------------------------")

    print("@ Loaded data scales are:")
    print(X.shape) # fixed length
    print(Y.shape) # could be padding with extra 0 in the end part. 

    if args.model_param != "":
        params = load_modelParam(args.model_param)
    else:
        print("! Unable to load the model parameters, pls check!")
        exit()

    # basic model parameters
    model_name = get_unet_model_name(params, args)

    model = models.load_model(modelSavePath+ "/weights/" + model_name+ ("_cont-" + str(args.contrain) if args.contrain > 0 else "") + ".h5", \
        custom_objects={'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef, 'bce_dice_loss': bce_dice_loss, \
        'categorical_focal_loss_fixed':categorical_focal_loss(gamma=2., alpha=.25), \
        'ce_dice_loss': ce_dice_loss})
    
    pred_batch = 10000
    num_test = X.shape[0]
    epochs = int(num_test/pred_batch)
    pred_results = []

    if epochs > 0:
        for i in range(epochs):
            dataX = X[i*pred_batch:(i+1)*pred_batch] 
            preds = model.predict(dataX, verbose=0)  
            pred_results.extend([p for p in preds])

    dataX = X[epochs*pred_batch:]
    preds = model.predict(dataX, verbose=0)  
    pred_results.extend([p for p in preds])

    pred_results = np.argmax(np.array(pred_results), axis=-1)
    print("Shape is the ", pred_results.shape)

    for i in range(2):
        print(pred_results[i])
        print(label_vec_new[i])
        print("----------------------")
 
    print("@ Visualziation confusion matrix ...")
    plot_confusion_matrix(label_vec_new.flatten(), pred_results.flatten(),range(8), "UNet_" + ("LSTM-"+str(args.lstm)  if args.lstm else "") + "_" + lossType)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training model with tfrecord file')

    parser.add_argument('-g', '--gpu', default='0', help="Assigned GPU for running the code")

    parser.add_argument('-i', '--data_dir', default="/data/workspace/nanopore/data/chiron_data/forDev/test/" ,required= False,
                        help="Directory that store the tfrecord files.")
    parser.add_argument('-o', '--log_dir', default="model/", required = False  ,
                        help="log directory that store the training model.")
    parser.add_argument('-m', '--model_name', default="devTest", required = False,
                        help='model_name')
    parser.add_argument('-v', '--validation', default = None, 
                        help="validation tfrecord file, default is None, which conduct no validation")
    parser.add_argument('-f', '--tfrecord', default="train.tfrecords",
                        help='tfrecord file')
    parser.add_argument('--test_cache', default=None, help="Cache file for training dataset.")
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
    parser.add_argument('--configure', default = None,
                        help="Model structure configure json file.")
    parser.add_argument('-k', '--k_mer', default=1, help='Output k-mer size')
    parser.add_argument('--retrain', dest='retrain', action='store_true',
                        help='Set retrain to true')
    parser.add_argument('--read_cache',dest='read_cache',action='store_true',
                        help="Read from cached hdf5 file.")
    parser.add_argument('-l', '--loss', default="categorical_loss", help="loss function used to learn the segmentation model.")

    parser.add_argument('-cf', '--cacheFile', default="", required=True, help="Assigned cache files.")
    parser.add_argument('-mp', '--model_param', default="", required=True, help="loss function used to learn the segmentation model.")
    parser.add_argument('-pf', '--plot_figure', default="", type=str, help="plot figure and show verbose")
    parser.add_argument('-tm', '--test_mode', default="cfm", required=True, help="test model selection cfm/plt")
    parser.add_argument('-lstm', '--lstm', default=0,type=int,  help="loss function used to learn the segmentation model.")

    parser.add_argument('-norm', '--norm', default="",type=str,  help="Training data statistics of saved file for global normalziation ... ")
    parser.add_argument('-tag', '--tag', default="",type=str,  help="Model tag information.")

    parser.add_argument('-iaw', '--inputAug_winLen', default=0,type=int,  help="input Signal augmentation with the windowScreen variance detection.")

    parser.add_argument('-fSignal', '--fSignal', default=0, type=int,  help="Extrem signals of data to determine whethe kept.")

    parser.add_argument('-cont', '--contrain', default=0,type=int,  help="Loading already training model to contintue the training process.")


    parser.set_defaults(retrain=False)
    args = parser.parse_args(sys.argv[1:])
   
    if args.test_cache is None:
        args.test_cache = args.data_dir + '../cache/test_chiron_gplabel.hdf5'
   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    if(args.test_mode == "cfm"):
        test_confustion_matrix(args)
    
    if(args.test_mode == "plt"):
        test_unet(args)
    


