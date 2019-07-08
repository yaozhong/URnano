# Copyright 2019 
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#


from __future__ import division

import numpy as np

from keras import Input, models, layers, regularizers, metrics
from keras.optimizers import RMSprop, SGD, Adam
from keras import callbacks, losses
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Lambda
from keras.callbacks import ModelCheckpoint, TensorBoard
from scipy.stats import beta
import pickle
import json


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.utils.multiclass import unique_labels

# 2019-06-18 testing with additional data augmentation to process the speed invarance. 
# data augmentation for the original data that try to capture the variance of data
def input_aug_localVariance_single(x, win_len= 11):
    # note to reshape the data
    x_pad = np.pad(x, (0, win_len-1), 'constant')
    var_shift = [ np.std(x_pad[i:i+win_len,0]) for i in range(len(x))]
    return var_shift

# 2019-06-03 decomposite multi-polymer number into 1-2 interval represeantation
def number_decomposition(num):
    if num < 3:
        return [num]

    repeats = int(num/3)
    tail_num = num%3

    if tail_num == 0:
        comp = [1, 2]*repeats
    elif tail_num == 1:
        comp = [1, 2]*repeats
        comp.append(1)
    elif tail_num == 2:
        comp = [2, 1]*repeats
        comp.append(2)

    return comp


#############################
# Basic encoding system
############################
def base2ind(base, alphabet_n=5, base_n=1):
    """base to 1-hot vector,
    Input Args:
        base: current base,can be AGCT, or AGCTX for methylation.
        alphabet_n: can be 4 or 5, related to normal DNA or methylation call.
    x is hte di
    """
    if alphabet_n == 4:
        Alphabeta = ['A', 'C', 'G', 'T']
        alphabeta = ['a', 'c', 'g', 't']
    elif alphabet_n == 5:
        Alphabeta = ['A', 'C', 'G', 'T', 'X']
        alphabeta = ['a', 'c', 'g', 't', 'x']
    else:
        raise ValueError('Alphabet number should be 4 or 5.')
    if base.isdigit():
        return int(base) / 256
    if ord(base) < 97:
        return Alphabeta.index(base) + 1
    else:
        return alphabeta.index(base) + 1


def ind2base(indList):

    strList = []
    Alphabeta = ['A', 'C', 'G', 'T', 'X']
    
    for idx in indList:
        idx = idx - 1
        strList.append(Alphabeta[idx])

    return strList

def index2base_0(indList):

    strList = []
    Alphabeta = ['A', 'C', 'G', 'T']
    
    for idx in indList:
        strList.append(Alphabeta[idx])

    return strList

#######################################
# model encoding part
#######################################

def nc2newIdx(nc, count, max_len=12):

	Alphabeta=['1', '2', '3', '4', '5']
	
	idx = Alphabeta.index(nc)
	new_idx = max_len*idx + count - 1
	
	return new_idx

def newIdx2nc(new_idx, max_len=12):
    
    Alphabet = ['A', 'C', 'G', 'T', "X"]
    idx = int(new_idx/max_len)
    count = new_idx%max_len + 1
    
    return (Alphabet[idx], count)

def nc_from2dList(ls):

    ncs = []
    for l in ls:
        nc= newIdx2nc(l, 12)
        nc = "\n".join([str(x) for x in nc])
        ncs.append(nc)
    return ncs


def nc_from2dList_11(ls):

    ncs = []
    for l in ls:
        nc= newIdx2nc(l, 2)
        nc= nc[0]
        ncs.append(nc)
    return ncs

def seqCompact(seq):

    idx = 0
    nc_seq, nc_count = [],[]

    current = seq[idx]
    count, output_seq = 1,""

    for i in range(1, len(seq)):
        if seq[i] == current:
            count += 1
        else:
            #output_seq += current
            nc_seq.append(current)
            nc_count.append(count)
            count, current = 1, seq[i]


    # stop until the first unknown coding
    #if str(current) != '4':
    #output_seq += current
    nc_seq.append(current)
    nc_count.append(count)

    return (nc_seq, nc_count)

# Transform the basic labeling to 2D labeling
def label2D_Transform(label_raw, debug=False):
    new_labels = []
    nc_seq, nc_count = seqCompact(np.array(label_raw).astype(str))
    
    for idx in range(len(nc_count)):
        new_idx = nc2newIdx(nc_seq[idx], nc_count[idx])
        #new_labels.extend([new_idx]*nc_count[idx])
        new_labels.append(new_idx)
        
    return new_labels

## added 2019/06/15
## for 11  this function is equal to label2D_Transform_norecovery_11
def label2D_Transform_11(label_raw, debug=False):
    new_labels = []
    nc_seq, nc_count = seqCompact(np.array(label_raw).astype(str))

    for idx in range(len(nc_count)):
        # go through every position
        for i in range(nc_count[idx]):
            if i%2 == 0:
                new_idx = nc2newIdx(nc_seq[idx], 1, 2)
            else:
                new_idx = nc2newIdx(nc_seq[idx], 2, 2)

            new_labels.append(new_idx)
    return new_labels


# Basic label tranformation
def getOrigin_from_label2Dseq(label_seqs):

	final_labels = []
	for label in label_seqs:
		nc, count = newIdx2nc(label)
		final_labels.extend([nc]*count)

	return final_labels


def getOrigin_from_label2Dseq_11(label_seqs):

    final_labels = []
    for label in label_seqs:
        nc, count = newIdx2nc(label, 2)
        final_labels.append(nc)

    return final_labels

########################################
# Final result prediction
########################################
def editDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min(distances[i1], distances[i1+1], distances_[-1]))
        distances = distances_

    return distances[-1]


# This function change consequent identical nucleotide to one tag
# this could be used to transform the orignal singal prediction label to nucleotide sequence
def toBases(logits, seq_len = None):
    
    idx = 0
    seq, count = [], []
    
    if seq_len == None:
        seq_len = len(logits)
        
    while(idx < seq_len):
        current = logits[idx]
        tmp_count = 1
        
    while( idx + 1 < seq_len and logits[idx+1] == current ): 
        idx += 1
        tmp_count += 1
        
        seq.append(current)
        count.append(tmp_count)
        idx += 1
        
    return(seq)


def ed(logits, seq_len, gold):

    ## calcuate the editor distannce between predicted and gold label
    seq = toBases(logits, seq_len)
    return editDistance(seq, gold)/len(gold)


###########################################
# plot currency segments of the results.
###########################################
def vis_segment(inputX, label_seg, label,  outputName):

    label = validRawLabel(label)
    label_seg = validRawLabel(label_seg)

    bks = np.cumsum(label_seg)
    bks = [0] + bks.tolist()

    fig=plt.figure(figsize=(10,2))
    plt.plot(np.array(range(len(inputX)), dtype=np.int16), inputX)

    plt.xticks(bks[:-1], ind2base(label), color="brown",fontsize=26)
    for bk in bks[:-1]:
        plt.axvline(bk, linestyle="-.", color="red")
            
    plt.savefig(outputName)
    plt.close("all") 

# true for visualization the std varance shift. 
def vis_prediction(inputX, label_seg, label,  pred_seg, pred, outputName, showRaw=True):

    label = validRawLabel(label)
    label_seg = validRawLabel(label_seg)

    bks = np.cumsum(label_seg)
    bks = [0] + bks.tolist()

    if showRaw == True:
        fig=plt.figure(figsize=(20,12))
    else:
        fig=plt.figure(figsize=(20,8))

    if showRaw == True:
        plt.subplot(311)
    else:
        plt.subplot(211)
    plt.plot(np.array(range(len(inputX)), dtype=np.int16), inputX)
    plt.xticks(bks[:-1], label, color="brown",fontsize=16)
    for bk in bks[:-1]:
        plt.axvline(bk, linestyle="-.", color="red")

    pred = validRawLabel(pred)
    pred_seg = validRawLabel(pred_seg)

    bks = np.cumsum(pred_seg)
    bks = [0] + bks.tolist()

    if showRaw == True:
        plt.subplot(312)
    else:
        plt.subplot(212)

    plt.plot(np.array(range(len(inputX)), dtype=np.int16), inputX)
    plt.xticks(bks[:-1], pred, color="brown",fontsize=16)
    for bk in bks[:-1]:
        plt.axvline(bk, linestyle="-.", color="blue")

    if showRaw == True:
        plt.subplot(313)
        #plot the raw infomration 
        plt.plot(np.array(range(len(inputX)), dtype=np.int16), inputX)
        ## plot std variance shift
        #varShift = input_aug_localVariance_single(inputX)
        #print(varShift)
        #plt.plot(np.array(range(len(inputX)), dtype=np.int16), varShift)
        
            
    plt.savefig(outputName)
    plt.close("all") 


### special plot, for the outliner and input-preprocessing results.


################################
#Tail labels processing
################################
def validRawLabel(raw_label):

    idx = len(raw_label)-1
    while (raw_label[idx] == 0 and idx > 0):
        idx = idx -1

    return raw_label[:idx+1]

def validPredLabel(pred):

    idx = len(pred) - 1
    while(pred[idx] == 'X' and idx > 0):
        idx = idx - 1

    return pred[:idx+1]


if __name__ == "__main__":

    logits = [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 5]
    gold =   [0, 1, 0, 1, 2, 2 ]

    print(ed(logits, 6, gold))


##################################
def plot_confusion_matrix(y_true, y_pred, classes, outputName,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, classes)
    print(cm.shape)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.hlines([1.5, 3.5, 5.5], xmin=0, xmax=8, linestyles='dashed')
    plt.vlines([1.5, 3.5, 5.5], ymin=0, ymax=8, linestyles='dashed')

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    #fmt = '.2f' if normalize else 'd'
    #thresh = cm.max() / 2.
    #for i in range(cm.shape[0]):
    #    for j in range(cm.shape[1]):
    #        ax.text(j, i, format(cm[i, j], fmt),
    #                ha="center", va="center",
    #                color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    plt.savefig("../experiment/devLOG/analysis_figure/"+ outputName + "_confustion_matrix.png")
    return ax


##################################################
# segments to weights
# 2019/06/96
##################################################

def betaWeights(seg, a=0.25,b=0.25):
    val = beta.pdf(np.linspace(1/seg/2, 1-1/seg/2, seg), a, b)
    val = val/np.sum(val)
    return val

# long prefer, middle part is less sensitive.
def seg2weights(segs):
    weights = []
    all_len = np.sum(segs)
    for seg in segs:
        weights.extend(betaWeights(seg) * seg/all_len)
    weights = weights/np.sum(weights)
    return weights

# generate the whole weight for the evluation data
def Y2weights(segs):
    weightMat = []
    for i in range(segs.shape[0]):
        #label, label_seg = seqCompact(Y[i])
        label_seg = validRawLabel(segs[i])
        weights = seg2weights(label_seg)
        weightMat.append(weights)
    return(np.array(weightMat))

##########################################################################

def get_unet_model_name(params, args):

    kernel_size = params["kernel_size"]
    stride=params["stride"]
    maxpooling_len = params["maxpooling_len"]
    conv_window_len = params["conv_window_len"]
    dropoutRate = params["DropoutRate"]
    lr = params["lr"]
    epoch = params["epoch"]
    bs= params["batchSize"]

    # model parameters
    model_name = "UNet" + ("_LSTM-"+str(args.networkID)  if args.networkID else "") \
            + "_model_loss-" + args.loss \
            + "-kernel_size_"+"-".join([str(l) for l in kernel_size]) \
            + "-maxpoolingLen_"+"-".join([str(l) for l in maxpooling_len]) \
            + "-convWindowLen_" + str(conv_window_len) \
            + "-stride_" + str(stride) \
            + "-lr_" + str(lr) \
            + "-dropout" + str(dropoutRate) \
            + "-Epoch" + str(epoch) \
            + "-batchSize_" + str(bs) \
            + ("-inputNorm"  if args.norm != "" else "") \
            + "-segLen_" + str(args.segment_len) \
            + ("-TAG_" + args.tag  if args.tag != "" else "") \
            + ("-signalAugWin_" + str(args.inputAug_winLen) if args.inputAug_winLen > 0 else "")

    return model_name

def load_modelParam(paramFile):
    
    with open(paramFile, "r") as f:
        param_dic = json.load(f)
        return param_dic

######################################
# added 2019-06-18
# the local normalization function.
######################################
def independ_sample_norm(X):
    for i in range(X.shape[0]):
        m_local, m_std = np.mean(X[i]), np.std(X[i])

        # output the strange input signals
        if np.isinf(m_local) or np.isinf(m_std):
            print(X[i])
            print("m=%f,std=%f" %(m_local, m_std))

        X[i] = (X[i] - m_local) / m_std
    return X

# if use data augmentation, it will changed to the 2D working. take care as the final step. 
def signalAug(X, win_len=11):
    varAug = [ input_aug_localVariance_single(X[i], win_len) for i in range(len(X))]
    varAug = np.array(varAug)
    return varAug.reshape((varAug.shape[0], varAug.shape[1], 1))









