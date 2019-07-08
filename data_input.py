# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

"""
This file is a revision of  Chiron(V0.3)'s chiron_input.py file.
(https://github.com/haotianteng/Chiron)
We make the following changes from the original one:
1. different padding approach for generating fixed-length signal vector.
2. revision of read_raw() that returns varibles including segment length, gold labels and signal vector of fixed length. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections, os, sys, tempfile

import h5py
import numpy as np
from statsmodels import robust
from six.moves import range, zip

import tensorflow as tf
from chiron.utils import progress
#from chiron.chiron_input import read_raw_data_sets

from utility import *

import matplotlib
matplotlib.use('Agg')  # this need for the linux env
import matplotlib.pyplot as plt


##################Previous implementation  ################
raw_labels = collections.namedtuple('raw_labels', ['start', 'length', 'base'])


class Flags(object):
    def __init__(self):
        self.max_segments_number = None
        self.MAXLEN = 1e9  # original run 1e4 Maximum Length of the holder in biglist. 1e5 by default

FLAGS = Flags()

class biglist(object):
    """
    biglist class, read into memory if reads number < MAXLEN, otherwise read into a hdf5 file.
    """

    def __init__(self, 
                 data_handle, 
                 dtype='float32', 
                 length=0, 
                 cache=False,
                 max_len=1e5):
        self.handle = data_handle
        self.dtype = dtype
        self.holder = list()
        self.length = length
        self.max_len = max_len
        self.cache = cache  # Mark if the list has been saved into hdf5 or not

    @property
    def shape(self):
        return self.handle.shape

    def append(self, item):
        self.holder.append(item)
        self.check_save()

    def __add__(self, add_list):
        self.holder += add_list
        self.check_save()
        return self

    def __len__(self):
        return self.length + len(self.holder)

    def resize(self, size, axis=0):
        self.save_rest()
        if self.cache:
            self.handle.resize(size, axis=axis)
            self.length = len(self.handle)
        else:
            self.holder = self.holder[:size]

    def save_rest(self):
        if self.cache:
            if len(self.holder) != 0:
                self.save()

    def check_save(self):
        if len(self.holder) > self.max_len:
            self.save()
            self.cache = True

    def save(self):
        if type(self.holder[0]) is list:
            max_sub_len = max([len(sub_a) for sub_a in self.holder])
            shape = self.handle.shape
            for item in self.holder:
                item.extend([0] * (max(shape[1], max_sub_len) - len(item)))
            if max_sub_len > shape[1]:
                self.handle.resize(max_sub_len, axis=1)
            self.handle.resize(self.length + len(self.holder), axis=0)
            self.handle[self.length:] = self.holder
            self.length += len(self.holder)
            del self.holder[:]
            self.holder = list()
        else:
            self.handle.resize(self.length + len(self.holder), axis=0)
            self.handle[self.length:] = self.holder
            self.length += len(self.holder)
            del self.holder[:]
            self.holder = list()

    def __getitem__(self, val):
        if self.cache:
            if len(self.holder) != 0:
                self.save()
            return self.handle[val]
        else:
            return self.holder[val]



class DataSet(object):
    def __init__(self, event, event_length, label, label_length, label_vec, label_segs, for_eval=False,):
        """Custruct a DataSet."""
        
        if for_eval == False:
            assert len(event) == len(label) and len(event_length) == len(label_length) and len(event) == len(event_length), "Sequence length for event \
            and label does not of event and label should be same, \
            event:%d , label:%d" % (len(event), len(label))
        
        self._event = event
        self._event_length = event_length
        self._label = label

        self._label_vec = label_vec # new added
        self._label_segs = label_segs # new added

        self._label_length = label_length
        self._reads_n = len(event)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._for_eval = for_eval
        self._perm = np.arange(self._reads_n)

    @property
    def event(self):
        return self._event

    @property
    def label(self):
        return self._label

    @property
    def label_vec(self):
        return self._label_vec

    @property
    def label_segs(self):
        return self._label_segs

    @property
    def event_length(self):
        return self._event_length

    @property
    def label_length(self):
        return self._label_length

    @property
    def reads_n(self):
        return self._reads_n

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def for_eval(self):
        return self._for_eval

    @property
    def perm(self):
        return self._perm

    def read_into_memory(self, index):
        event = np.asarray(list(zip([self._event[i] for i in index],
                                    [self._event_length[i] for i in index])))
        
        if not self.for_eval:
            label = np.asarray(list(zip([self._label[i] for i in index],
                                        [self._label_length[i] for i in index])))
            
            label_vec = np.asarray(list(zip([self._label_vec[i] for i in index],
                                        [self._label_segs[i] for i in index])))

        else:
            label = []
            label_vec = []

        return event, label, label_vec


    def next_batch(self, batch_size, shuffle=True, sig_norm=False):
        """Return next batch in batch_size from the data set.
            Input Args:
                batch_size:A scalar indicate the batch size.
                shuffle: boolean, indicate if the data should be shuffled after each epoch.
                sig_norm: If the signal need to be normalized, if sig_norm set
                to True when read the data, then the redundant sig_norm is not required.
            Output Args:
                inputX,sequence_length,label_batch: tuple of (indx,vals,shape)
        """

        if self.epochs_completed>=1 and self.for_eval:
            print("Warning, evaluation dataset already finish one iteration.")
        
        start = self._index_in_epoch
        # Shuffle for the first epoch
        
        if self._epochs_completed == 0 and start == 0:
            if shuffle:
                np.random.shuffle(self._perm)
        
        # Go to the next epoch
        if start + batch_size > self.reads_n:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest samples in this epoch
            rest_reads_n = self.reads_n - start
            event_rest_part, label_rest_part, label_vec_rest_part = self.read_into_memory(self._perm[start:self._reads_n])
            
            start = 0
            if self._for_eval:
                event_batch = event_rest_part
                label_batch = label_rest_part
                label_vec_batch = label_vec_rest_part
                self._index_in_epoch = 0
                end = 0
            # Shuffle the data
            else:
                if shuffle:
                    np.random.shuffle(self._perm)
                # Start next epoch
                self._index_in_epoch = batch_size - rest_reads_n
                end = self._index_in_epoch
                event_new_part, label_new_part, label_vec_new_part = self.read_into_memory(self._perm[start:end])
                
                if event_rest_part.shape[0] == 0:
                    event_batch = event_new_part
                    label_batch = label_new_part
                    label_vec_batch = label_vec_new_part
                    
                elif event_new_part.shape[0] == 0:
                    event_batch = event_rest_part
                    label_batch = label_rest_part
                    label_vec_batch = label_vec_rest_part
                else:
                    event_batch = np.concatenate((event_rest_part, event_new_part), axis=0)
                    label_batch = np.concatenate((label_rest_part, label_new_part), axis=0)
                    label_vec_batch = np.concatenate((label_vec_rest_part, label_vec_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            event_batch, label_batch, label_vec_batch = self.read_into_memory(self._perm[start:end])
        
        # errors happens here 
        if(len(label_vec_batch) > 0):
            label_segs = [ x  for x in label_vec_batch[:,1] ]
            label_raw = [ x for x in label_batch[:,0]]
        else:
            label_segs, label_raw = None, None

        if not self._for_eval:
            label_batch = batch2sparse(label_batch)
        
        seq_length = event_batch[:, 1].astype(np.int32)

        # conditional added 
        if(len(label_vec_batch) > 0):
            return np.vstack(event_batch[:, 0]).astype(np.float32), seq_length, label_batch, \
                np.vstack(label_vec_batch[:,0]).astype(np.int32), label_segs, label_raw
        else:
            return np.vstack(event_batch[:, 0]).astype(np.float32), seq_length, label_batch, \
                None, None, None


def read_data_for_eval(file_path, 
                       start_index=0,
                       step=20, 
                       seg_length=300, 
                       sig_norm="median",
                       reverse = False):
    """
    Input Args:
        file_path: file path to a signal file.
        start_index: the index of the signal start to read.
        step: sliding step size.
        seg_length: length of segments.
        sig_norm: if the signal need to be normalized.
        reverse: if the signal need to be reversed.
    """
    if not file_path.endswith('.signal'):
        raise ValueError('A .signal file is required.')
    else:
        event = list()
        event_len = list()
        label = list()
        label_len = list()
        f_signal = read_signal(file_path)
        if reverse:
            f_signal = f_signal[::-1]
        f_signal = f_signal[start_index:]
        sig_len = len(f_signal)
        for indx in range(0, sig_len, step):
            segment_sig = f_signal[indx:indx + seg_length]
            segment_len = len(segment_sig)
            padding(segment_sig, seg_length)
            event.append(segment_sig)
            event_len.append(segment_len)

        #evaluation = DataSet(event=event, event_length=event_len, label=label, label_length=label_len, for_eval=True)
        ## take care about the label_vec, label_segs
        evaluation = DataSet(event=event, event_length=event_len, label=label, label_length=label_len, label_vec = list(), label_segs= list(), for_eval=True)

    return evaluation


def read_cache_dataset(h5py_file_path):
    """Notice: Return a data reader for a h5py_file, call this function multiple
    time for parallel reading, this will give you N dependent dataset reader,
    each reader read independently from the h5py file."""
    hdf5_record = h5py.File(h5py_file_path, "r")
    event_h = hdf5_record['event/record']
    event_length_h = hdf5_record['event/length']
    label_h = hdf5_record['label/record']
    label_vec_h = hdf5_record['label/record_vec']
    label_seg_h = hdf5_record['label/record_seg']
    label_length_h = hdf5_record['label/length']
    event_len = len(event_h)
    label_len = len(label_h)
    assert len(event_h) == len(event_length_h)
    assert len(label_h) == len(label_length_h)
    
    event = biglist(data_handle=event_h, length=event_len, cache=True)
    event_length = biglist(data_handle=event_length_h, length=event_len,
                           cache=True)
    label = biglist(data_handle=label_h, length=label_len, cache=True)
    label_length = biglist(data_handle=label_length_h, length=label_len,
                           cache=True)
    label_vec = biglist(data_handle=label_vec_h, length=event_len, cache=True)
    label_seg = biglist(data_handle=label_seg_h, length=label_len, cache=True)

    return DataSet(event=event, event_length=event_length, label=label,
                   label_length=label_length, label_vec = label_vec, label_segs= label_seg)


## Read from raw data    
def read_raw_data_sets(data_dir, h5py_file_path=None, seq_length=300, k_mer=1, max_segments_num=FLAGS.max_segments_number):
    
    # make temp record
    if h5py_file_path is None:
        h5py_file_path = tempfile.mkdtemp() + '/temp_record.hdf5'
    else:
        try:
            os.remove(os.path.abspath(h5py_file_path))
        except:
            pass
        if not os.path.isdir(os.path.dirname(os.path.abspath(h5py_file_path))):
            os.mkdir(os.path.dirname(os.path.abspath(h5py_file_path)))
    
    
    with h5py.File(h5py_file_path, "a") as hdf5_record:
        
        event_h = hdf5_record.create_dataset('event/record', dtype='float32', shape=(0, seq_length), maxshape=(None, seq_length))
        event_length_h = hdf5_record.create_dataset('event/length', dtype='int32', shape=(0,), maxshape=(None,), chunks=True)
        
        label_h = hdf5_record.create_dataset('label/record', dtype='int32',shape=(0, 0), maxshape=(None, seq_length))
        label_length_h = hdf5_record.create_dataset('label/length',dtype='int32', shape=(0,), maxshape=(None,))
        
        label_vec_h = hdf5_record.create_dataset('label/record_vec', dtype='int32',shape=(0, seq_length), maxshape=(None, seq_length))
        label_segs_h = hdf5_record.create_dataset('label/record_seg', dtype='int32',shape=(0, 0), maxshape=(None, seq_length))

        event = biglist(data_handle=event_h, max_len=FLAGS.MAXLEN)
        event_length = biglist(data_handle=event_length_h, max_len=FLAGS.MAXLEN)
        
        label = biglist(data_handle=label_h, max_len=FLAGS.MAXLEN)
        label_length = biglist(data_handle=label_length_h, max_len=FLAGS.MAXLEN)
        label_vec = biglist(data_handle=label_vec_h, max_len=FLAGS.MAXLEN)
        label_segs = biglist(data_handle=label_segs_h, max_len=FLAGS.MAXLEN)

        count = 0
        file_count = 0
        
        for name in os.listdir(data_dir):
            
            if name.endswith(".signal"):
                file_pre = os.path.splitext(name)[0]
                
                f_signal = read_signal(data_dir + name)

                if len(f_signal) == 0:
                    continue
                try:
                    f_label = read_label(data_dir + file_pre + '.label',
                                         skip_start=10,
                                         window_n=int((k_mer - 1) / 2))
                except:
                    sys.stdout.write("Read the label %s fail.Skipped." % (name))
                    continue

                # read_raw singals, it will be segmented into (event, label)
                tmp_event, tmp_event_length, tmp_label, tmp_label_length, tmp_label_vec, tmp_label_segs = read_raw(f_signal, f_label, seq_length)
                
                event += tmp_event
                event_length += tmp_event_length

                label += tmp_label
                label_length += tmp_label_length
                label_vec += tmp_label_vec
                label_segs += tmp_label_segs
                
                del tmp_event
                del tmp_event_length
                del tmp_label
                del tmp_label_length
                del tmp_label_vec
                del tmp_label_segs
            
                count = len(event)
                
                if file_count % 10 == 0:
                    
                    if max_segments_num is not None:
                        sys.stdout.write("%d/%d events read.   \n" % (count, max_segments_num))
                        
                        if len(event) > max_segments_num:
                            event.resize(max_segments_num)
                            label.resize(max_segments_num)
                            label_vec.resize(max_segments_num)
                            label_segs.resize(max_segments_num)

                            event_length.resize(max_segments_num)
                            label_length.resize(max_segments_num)
                            
                            break
                    else:
                        sys.stdout.write("%d lines read.   \n" % (count))
                file_count += 1

    if event.cache:
        train = read_cache_dataset(h5py_file_path)
    else:
        train = DataSet(event=event, event_length=event_length, label=label,
                        label_length=label_length, label_vec=label_vec, label_segs= label_segs)

    return train

#data normalization applied here
def read_signal(file_path, normalize="median"):
    
    f_h = open(file_path, 'r')
    signal = list()
    
    for line in f_h:
        signal += [float(x) for x in line.split()]
    signal = np.asarray(signal)
    
    if len(signal) == 0:
        return signal.tolist()
    
    if normalize == "mean":
        signal = (signal - np.mean(signal)) / np.float(np.std(signal))
    
    elif normalize == "median":
        signal = (signal - np.median(signal)) / np.float(robust.mad(signal))
    
    return signal.tolist()


def read_label(file_path, skip_start=10, window_n=0):
    
    f_h = open(file_path, 'r')
    start = list()
    length = list()
    base = list()
    all_base = list()
    
    if skip_start < window_n:
        skip_start = window_n
    
    for line in f_h:
        record = line.split()
        all_base.append(base2ind(record[2]))
    
    f_h.seek(0, 0)  # Back to the start
    file_len = len(all_base)
    
    for count, line in enumerate(f_h):
        
        record = line.split()
        
        if count < skip_start or count > (file_len - skip_start - 1):
            continue
        
        start.append(int(record[0]))
        length.append(int(record[1]) - int(record[0]))
        
        k_mer = 0
        
        for i in range(window_n * 2 + 1):
            k_mer = k_mer * 4 + all_base[count + i - window_n]
        base.append(k_mer)
    
    return raw_labels(start=start, length=length, base=base)



# joint loading, can revise here
def read_raw(raw_signal, raw_label, max_seq_length):
    
    label_val = list()
    label_length = list()
    label_val_vec = list()
    label_segs = list()

    event_val = list()
    event_length = list()
    
    current_length = 0
    current_label = []
    current_event = []
    current_label_vec = []
    current_segs = []

    
    for indx, segment_length in enumerate(raw_label.length):
        
        current_start = raw_label.start[indx]
        current_base = raw_label.base[indx]

        if current_length + segment_length < max_seq_length:
            current_event += raw_signal[current_start:current_start + segment_length]
            current_label.append(current_base)
            current_label_vec += [current_base] * segment_length
            current_length += segment_length
            current_segs += [segment_length]

        else:
            if current_length > (max_seq_length / 2) and len(current_label) > 5:

                current_event += raw_signal[current_start: current_start + (max_seq_length - current_length)]
                # padding(current_event, max_seq_length, raw_signal[current_start : current_start + max_seq_length])
                padding(current_label_vec, max_seq_length, [ current_base ] * (max_seq_length - current_length))

                # 20190515 add the incomplete information
                current_label.append(current_base)
                current_segs += [ max_seq_length - current_length ]
                # update the length last
                current_length += max_seq_length - current_length

                event_val.append(current_event)
                event_length.append(current_length)
        
                # add to the list
                label_val.append(current_label)
                label_length.append(len(current_label))
                label_val_vec.append(current_label_vec)
                label_segs.append(current_segs)

            # Begin a new event-label, resetting
            current_event = raw_signal[current_start:current_start + segment_length]
            current_length = segment_length
            current_label = [current_base]
            
            current_label_vec = [current_base] * segment_length
            current_segs = [segment_length]

        if segment_length == 0:
            print("----val-%s set==0---" %(current_label))

    #print(len(event_val))
    #print("-"*30)
    #print(label_length)
    #print("="*30)

    return event_val, event_length, label_val, label_length, label_val_vec, label_segs


# padding the rest of part with vector
def padding(x, L, padding_list=None):
    """Padding the vector x to length L"""
    len_x = len(x)
    assert len_x <= L, "Length of vector x is larger than the padding length"
    zero_n = L - len_x
    if padding_list is None:
        x.extend([0] * zero_n)
    elif len(padding_list) < zero_n:
        x.extend(padding_list + [0] * (zero_n - len(padding_list)))
    else:
        x.extend(padding_list[0:zero_n])
    return None


def batch2sparse(label_batch):
    """Transfer a batch of label to a sparse tensor
    """
    values = []
    indices = []
    for batch_i, label_list in enumerate(label_batch[:, 0]):
        for indx, label in enumerate(label_list):
            if indx >= label_batch[batch_i, 1]:
                break
            indices.append([batch_i, indx])
            values.append(label)
    shape = [len(label_batch), max(label_batch[:, 1])]
    return indices, values, shape



################################################
# loading data
################################################

def loading_data(fileConfigSet, cacheFile, len_encoding=True):

    if os.path.exists(cacheFile):
        
        with h5py.File(cacheFile, "r") as hf:
            
            X = hf["X_data"][:]
            seq_len = hf["seq_len"][:]
            label = [hf["Y_ctc/index"][:], hf["Y_ctc/value"][:], hf["Y_ctc/shape"]]
            label_vec = hf["Y_vec"][:]
            
            label_seg = [ hf["Y_seg/"+str(i)][:] for i in range(len(X)) ]
            # checking the h5py loading process
            label_raw = [ hf["label_raw/"+str(i)][:] for i in range(len(X))]

    else:
        
        print("Now caching the data ... ")
        ds = read_raw_data_sets(fileConfigSet[0], fileConfigSet[1], 300, 1)

        # call the chiron function for loading the data
        X, seq_len, label, label_vec, label_seg, label_raw = ds.next_batch(ds._reads_n)
        
        with h5py.File(cacheFile, "w") as hf:

            hf.create_dataset("X_data", data=X)
            hf.create_dataset("seq_len", data=seq_len)
            # this used for sparse matrix
            hf.create_dataset("Y_vec", data=label_vec)
            hf.create_dataset("Y_ctc/index", data=label[0])
            hf.create_dataset("Y_ctc/value", data=label[1])
            hf.create_dataset("Y_ctc/shape", data=label[2])

            for i in range(len(label_raw)):
                hf.create_dataset("Y_seg/"+str(i), data=label_seg[i])
                # be careful about the label-0 conflict issues
                hf.create_dataset("label_raw/"+str(i), data=np.array(label_raw[i], dtype=int))

        print("[%d] segments Data loading Done!" %(X.shape[0]))

    if len_encoding == False:
        return X, seq_len, label, label_vec, label_seg, label_raw
    
    ########## 2019/06/15 added #########
    # print("@ (*) 8-label (1,1) split transformation AAA to A1A2A1 ... ")
    label_vec_new = []  

    for i in range(len(label_raw)):

        gold_raw = validRawLabel(label_raw[i]) # remove the end-tails of 0s

        # transform to new labels
        newLabels = label2D_Transform_11(gold_raw)
        newLabels_noRecovery = newLabels  #label2D_Transform_norecovery_11(gold_raw)

        assert(len(gold_raw) == len(newLabels_noRecovery))

        newLabelVec = []
        # generate labels for the signals
        for j in range(len(gold_raw)):
            newLabelVec.extend([ newLabels_noRecovery[j] ] * label_seg[i][j])
                
        # padding,take care about this padding part!!!, why the length is different
        assert(len(newLabelVec) == len(label_vec[i]))
        label_vec_new.append(newLabelVec)
        
    label_new = np.array(label_vec_new)

    return X, seq_len, label, label_vec, label_seg, label_raw, label_new


def test():
    ### Input Test ###
    Data_dir = '/data/workspace/nanopore/data/chiron_data/train/'
    #train = read_tfrecord(Data_dir,"train.tfrecords",seq_length=1000)
    eval = read_raw_data_sets(Data_dir)
    
    for i in range(5):
        inputX, sequence_length, label, label_vec, label_seg = eval.next_batch(1)
        bks = np.cumsum(label_seg[0])
        
        print(inputX.shape)
        print(label[2])
        print(label_vec.shape)
        print("-"*10)
        print(label_seg[0])

        bks = np.cumsum([0] + label_seg[0])
        print(bks)
        """
        print(len(label_seg[0]))
        print(label_vec)
        print(label[1])
        print(len(label[1]))
        print(sequence_length)
        """
        doPlot=True

        if doPlot:
            fig=plt.figure(figsize=(50,10))
            plt.plot(np.array(range(inputX.shape[1]), dtype=np.int16), \
                    inputX[0,:])
            plt.xticks(bks[:-1], ind2base(label[1]), color="brown",fontsize=26)
            for bk in bks:
                plt.axvline(bk, linestyle="-.", color="red")
            
            plt.savefig("/dropbox/currency"+str(i)+".png")
            plt.close("all")

# filtering out the outliner signals in the training. 
def signalFiltering(X, threshold=10):

    Idx = []
    count_filtered = 0
    for i in range(X.shape[0]):
        tag = True
        for j in range(X.shape[1]):
            if np.abs(X[i,j]) > threshold:
                #print(X[i,j])
                count_filtered += 1
                tag = False
        
        if tag == True:
            Idx.append(i)

    print("@@ Total filtering [%d] samples that exceed max Currency strength of %d" %(count_filtered, threshold))

    return Idx

if __name__ == '__main__':
    #test()

    for i in range(13):
        print("- Decompisiotn of [%d]" %i)
        print(number_decomposition(i))

