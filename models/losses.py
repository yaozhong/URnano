# Copyright 2019 
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#


from __future__ import division

from keras import callbacks, losses
from keras import backend as K
from keras.utils import to_categorical

## Evluation metric for the final score calcuation
## 1. Editor distance
def edit_distance(y_true, y_pred):

    y_true = create_sparse(K.tf.argmax(y_true, axis=-1))
    y_pred = create_sparse(K.tf.argmax(y_pred, axis=-1))

    return(K.tf.edit_distance(y_pred, y_true, normalize=True))

## not used in the current version
def dice_score(gold, pred):
    gold, pred = gold.flatten(), pred.flatten()
    intersection = np.sum((gold-pred) == 0)
    df = (2. * intersection + np.finfo(np.float32).eps) / (np.len(gold) + np.len(pred) + np.finfo(np.float32).eps)
    return df

def IOU(rec1, rec2):
    ### vector processing to the start index.
    # computing area of each rectangles
    S_rec1 = (rec1[1] - rec1[0])
    S_rec2 = (rec2[1] - rec2[0])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[1], rec2[1])
 
    # judge if there is an intersect
    if rec1[1] < rec2[0] or rec2[1] < rec1[0]:
        return 0
    else:
        intersect = (right_line - left_line) 
        return intersect / (sum_area - intersect)
 

##################################################
## 2. Dice loss for multi class
##################################################

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2.*intersection + K.epsilon()) / ( K.sum(y_true) + K.sum(y_pred) + K.epsilon())

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

## change the name in the later stage, balanced categorical loss 
## now you can only say it is running in a mixed model, this is not correct in the current version
def bce_dice_loss(y_true, y_pred):
    return 0.5*losses.categorical_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)

## no weight is used
def ce_dice_loss(y_true, y_pred):
    return losses.categorical_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)

## 20190607
def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)
    return categorical_focal_loss_fixed


# 20180608 test version, wights, 
def weight_categorical_loss(weights):

    weights = K.variable(weights)
    
    def weight_categorical_loss_fix(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True) 
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return weight_categorical_loss_fix


