# Copyright 2019 
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

"""
Date: 2019-04-25
Description: Hyperopt for search the best model performance
"""

# 20190607 reivision to the update version

from __future__ import division
from utility import *

from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval
from models.model_unet import *

## the data should be expose to the level of objective function
CB = [ callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="auto", restore_best_weights=True) ] 

# define parameter space
space ={
        'kernel_size':hp.choice('kernel_size', [[64,128,256, 512], [32,64,128,256]]),
        'conv_window_len':hp.choice('conv_window_len',[3, 5, 7, 9, 11]),
        'maxpooling_len':hp.choice('maxpooling_len', [[3,2,2], [2,2,3], [5,3,2]]),
        'stride': hp.choice("stride", [1]),
         # 256 will be out of memory
        'batchSize': hp.choice('batchSize', [32, 64, 128, 256]),
        'lr': hp.choice('lr', [ 1e-4, 1e-3, 1e-2, 0.1]),
        'DropoutRate': hp.choice("DropoutRate", [0, 0.2, 0.5]),
         #'BN': hp.choice("BN", [True, False]),
        'epoch': hp.choice("epoch",[25])
}

def get_ed_list(X, Y, model):

    preds = model.predict(X, verbose=0)
    golds = Y
    eds = []

    for i in range(X.shape[0]):

        logits = np.argmax(preds[i], -1)
        label_seq, noused_count = seqCompact(logits) 
        pred = getOrigin_from_label2Dseq_11(label_seq)
        pred = validPredLabel(pred)
        
        # gold = ind2base(validRawLabel(golds[i]))
        gold = np.argmax(golds[i], -1)
        label_seq, noused_count = seqCompact(gold)
        gold = getOrigin_from_label2Dseq_11(label_seq)
        gold = validPredLabel(gold)

        ed =editDistance(pred, gold)
        eds.append(ed/len(gold))
    
    return eds
      
# define objective function
def objective(params):

    signals = Input(name='input', shape=[X_train.shape[1], X_train.shape[-1]], dtype=np.float32)

    print("** START hyper-parameters tunning of UNet-GRU3")
    model = UNet_GRU3(signals, params["kernel_size"], params["conv_window_len"],  params["maxpooling_len"],  \
        params["stride"], True, params["DropoutRate"])
    
    model.compile(optimizer=Adam(params["lr"]), loss = ce_dice_loss , metrics=[ metrics.categorical_accuracy ])
  
    model.fit(X_train,  Y_train, epochs=params["epoch"], batch_size=params["batchSize"], verbose=0, \
            callbacks = CB, validation_split=0.2)
    
    # test part, loss is edit distance loss
    eds = get_ed_list(X_test, Y_test, model)

    return {'loss':np.mean(eds), 'status':STATUS_OK}


#######################################################
# call API, through the function 
#######################################################
def do_hyperOpt(data, tryTime = 100, paramFile=None):
    
    x_data, y_data = data
    
    # set the local global name
    global X_train, Y_train, X_test, Y_test
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1234)

    print "* Model hyper parmaters tunning start ..."
 
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=tryTime, trials=trials, verbose=0)
    param_dic = space_eval(space, best)
    
    jd = json.dumps(param_dic)
    output = open(paramFile, "w")
    output.write(jd)
    output.close()



    
