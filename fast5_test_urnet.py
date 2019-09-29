# Copyright 2019 
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

"""
The fast5 reading and merging of base-calling results for a whole read part
are based on Chiron's (v0.3) code.
For compacting the code, according functions are direclty embedded here.
"""

from utils import raw
#from utils.easy_assembler import simple_assembly
from easy_assembler import simple_assembly
from utils.unix_time import unix_time
from tqdm import tqdm
import h5py
import numpy as np

import argparse, os, sys, time
from data_input import read_data_for_eval,read_data_for_eval2

from utility import *
from models.model_unet import *

##########################################################
#  Fast5 extraction to signal files
##########################################################
def extract_file(input_file,mode = 'dna'):
    try:
        input_data = h5py.File(input_file, 'r')
    except IOError as e:
        print(e)
        raise IOError(e)
    except Exception as e:
        print(e)
        raise Exception(e)

    raw_signal = list(input_data['/Raw/Reads'].values())[0]['Signal'].value
    
    if mode == 'rna':
        raw_signal = raw_signal[::-1]
        
    try:
        reference = input_data['Analyses/Basecall_1D_000/BaseCalled_template/Fastq'].value
        reference = '@%s\n'%(os.path.basename(input_file).split('.')[0]) + '\n'.join(reference.decode('UTF-8').split('\n')[1:])
    except:
        try:
            reference = input_data['Analyses/Alignment_000/Aligned_template/Fasta'].value
        except Exception as e:
            print('Generateing %s without reference.'%(input_file))
            reference = ''

    return raw_signal, reference


def extract_fast5(args):

    tqdm.monitor_interval = 0
    count = 1
    root_folder, out_folder = args.input, args.output

    if not os.path.isdir(root_folder):
        raise IOError('Input directory does not found.')
    if out_folder is None:
        raw_folder = os.path.abspath(os.path.join(out_folder, 'raw'))
        ref_folder = os.path.abspath(os.path.join(out_folder, 'reference'))
    else:
        if not os.path.isdir(args.output):
            os.makedirs(args.output)

        raw_folder = os.path.abspath(os.path.join(out_folder, 'raw'))
        ref_folder = os.path.abspath(os.path.join(out_folder, 'reference'))

    if not os.path.isdir(raw_folder):
        os.mkdir(raw_folder)
    if not os.path.isdir(ref_folder):
        os.mkdir(ref_folder)
    
    if args.recursive:
        dir_list = os.walk(root_folder)
    else:
        dir_list = [root_folder]

    for dir_tuple in tqdm(dir_list,desc = "Subdirectory processing:",position = 0):
        if args.recursive:
            directory = dir_tuple[0]
            file_list = dir_tuple[2]
        else:
            file_list = os.listdir(dir_tuple)

        for file_n in tqdm(file_list,desc = "Signal processing:",position = 1):
            print(file_n)
            if args.recursive:
                full_file_n = os.path.join(directory,file_n)
            else:
                full_file_n = os.path.join(root_folder,file_n)

            if file_n.endswith('fast5'):
                try:
                    raw_signal, reference = extract_file(full_file_n, args.mode)
                    if raw_signal is None:
                        raise ValueError("Fail in extracting raw signal.")
                    if len(raw_signal) == 0:
                        raise ValueError("Got empty raw signal")
                    count += 1
                except Exception as e:
                    print("!! Cannot extact file %s. %s"%(full_file_n,e))
                    continue
                with open(os.path.join(raw_folder, os.path.splitext(file_n)[0] + '.signal'), 'w+') as signal_file:
                    signal_file.write(" ".join([str(val) for val in raw_signal]))
                if len(reference) > 0:
                    with open(os.path.join(ref_folder, os.path.splitext(file_n)[0] + '_ref.fastq'), 'w+') as ref_file:
                        ref_file.write(reference)

## this function is used when generating simple_assembly and soft_merging assembly simultaneously
def write_output2(args, segments, consensus,arda_consensus, time_list, file_pre, concise=False, suffix='fastq', seg_q_score=None,
                 q_score=None):
    """
    seg_q_score: A length seg_num string list. Quality score for the segments.
    q_socre: A string. Quality score for the consensus sequence.
    """
    start_time, reading_time, basecall_time, assembly_time = time_list
    result_folder = os.path.join(args.output, 'result')
    seg_folder = os.path.join(args.output, 'segments')
    meta_folder = os.path.join(args.output, 'meta')
    path_con = os.path.join(result_folder, file_pre + '.' + suffix)
    path_con_arda = os.path.join(result_folder,file_pre +'arda'+ '.' + suffix)
    if not concise:
        path_reads = os.path.join(seg_folder, file_pre + '.' + suffix)
        path_meta = os.path.join(meta_folder, file_pre + '.meta')
    with open(path_reads, 'w+') as out_f, open(path_con, 'w+') as out_con,open(path_con_arda, 'w+') as out_con_arda:
        if not concise:
            for indx, read in enumerate(segments):
                out_f.write(file_pre + str(indx) + '\n')
                out_f.write(read + '\n')
                if (suffix == 'fastq') and (seg_q_score is not None):
                    out_f.write('+\n')
                    out_f.write(seg_q_score[indx] + '\n')
        q_score = "".join(["+" for i in range(len(consensus))])
        if (suffix == 'fastq') and (q_score is not None):
            arda_qscore = "".join(["!" for i in range(len(arda_consensus))])
            out_con.write('@{}\n{}\n+\n{}\n'.format(file_pre, consensus, q_score))
            out_con_arda.write('@{}\n{}\n+\n{}\n'.format(file_pre, arda_consensus, arda_qscore))
        else:
            out_con.write('{}\n{}'.format(file_pre, consensus))
    if not concise:
        with open(path_meta, 'w+') as out_meta:
            total_time = time.time() - start_time
            output_time = total_time - assembly_time
            assembly_time -= basecall_time
            basecall_time -= reading_time
            total_len = len(consensus)
            total_time = time.time() - start_time
            out_meta.write("# Reading Basecalling assembly output total rate(bp/s)\n")
            out_meta.write("%5.3f %5.3f %5.3f %5.3f %5.3f %5.3f\n" % (
                reading_time, basecall_time, assembly_time, output_time, total_time, total_len / total_time))
            out_meta.write("# read_len batch_size segment_len jump start_pos\n")
            out_meta.write(
                "%d %d %d %d %d\n" % (total_len, args.batch_size, args.segment_len, args.jump, args.start))
            out_meta.write("# input_name model_name\n")
            out_meta.write("%s %s\n" % (args.input, args.model))
def write_output(args, segments, consensus, time_list, file_pre, concise=False, suffix='fastq', seg_q_score=None,
                 q_score=None,arda=True):
    """
    seg_q_score: A length seg_num string list. Quality score for the segments.
    q_socre: A string. Quality score for the consensus sequence.
    """
    start_time, reading_time, basecall_time, assembly_time = time_list
    result_folder = os.path.join(args.output, 'result')
    seg_folder = os.path.join(args.output, 'segments')
    meta_folder = os.path.join(args.output, 'meta')
    path_con = os.path.join(result_folder, file_pre + '.' + suffix)
    if not concise:
        path_reads = os.path.join(seg_folder, file_pre + '.' + suffix)
        path_meta = os.path.join(meta_folder, file_pre + '.meta')
    with open(path_reads, 'w+') as out_f, open(path_con, 'w+') as out_con:
        if not concise:
            for indx, read in enumerate(segments):
                if arda:
                    q_score = "".join(["+" for i in range(len(consensus))])
                    break
                out_f.write(file_pre + str(indx) + '\n')
                out_f.write(read + '\n')
                if (suffix == 'fastq') and (seg_q_score is not None):
                    out_f.write('+\n')
                    out_f.write(seg_q_score[indx] + '\n')
        if (suffix == 'fastq') and (q_score is not None):
            out_con.write('@{}\n{}\n+\n{}\n'.format(file_pre, consensus, q_score))
        else:
            out_con.write('{}\n{}'.format(file_pre, consensus))
    if not concise:
        with open(path_meta, 'w+') as out_meta:
            total_time = time.time() - start_time
            output_time = total_time - assembly_time
            assembly_time -= basecall_time
            basecall_time -= reading_time
            total_len = len(consensus)
            total_time = time.time() - start_time
            out_meta.write("# Reading Basecalling assembly output total rate(bp/s)\n")
            out_meta.write("%5.3f %5.3f %5.3f %5.3f %5.3f %5.3f\n" % (
                reading_time, basecall_time, assembly_time, output_time, total_time, total_len / total_time))
            out_meta.write("# read_len batch_size segment_len jump start_pos\n")
            out_meta.write(
                "%d %d %d %d %d\n" % (total_len, args.batch_size, args.segment_len, args.jump, args.start))
            out_meta.write("# input_name model_name\n")
            out_meta.write("%s %s\n" % (args.input, args.model))

## soft_merging algorithm
def concatenate_reads(raw_reads,jump=290):
    final_raw_reads= np.zeros((300+jump*(len(raw_reads)-1),len(raw_reads[0][0])))
    final_raw_reads[:jump] = raw_reads[0][:jump]
    for i in range(1,len(raw_reads)):
        final_raw_reads[i*jump:i*jump+len(raw_reads[0])]+= raw_reads[i]
    logits = np.argmax(final_raw_reads,-1)
    pred_2d, pred_seg = seqCompact(logits)
    pred = getOrigin_from_label2Dseq_11(pred_2d)
    pred = validPredLabel(pred)
    #print("Final length of prediction after compression %d "%len(pred))
    return pred


## basecaller for the new assembly method
## collapsing is delayed until the next step
def unet_basecaller2(model, X,jump = 30):
    preds = model.predict(X, verbose=0) 
    return preds


def unet_basecaller(model, X):

    preds = model.predict(X, verbose=0)
    reads = []
    for i in range(preds.shape[0]):
        logits = np.argmax(preds[i], -1)
        #print("logits")
        print("".join([str(x) for x in logits]))
        #return logits
        print("".join(getOrigin_from_label2Dseq_11(logits)))
        #return
        pred_2d, pred_seg = seqCompact(logits)
        pred = getOrigin_from_label2Dseq_11(pred_2d)
        pred = validPredLabel(pred)
        reads.append("".join(pred[1:-1] if len(pred)>3 else pred))

    return reads
            

##########################################################
# Trying new stuff for the basecalling part
##########################################################
def evaluation2(signal_input, args):

    print("@ Loading U-net model ...")
    if args.model_param != "":
        params = load_modelParam(args.model_param)
    else:
        print("! Unable to load the model parameters, pls check!")
        exit()
    model_name = get_unet_model_name(params, args)
    unet_model = models.load_model("./experiment/model/weights/" + model_name+ ".h5", \
        custom_objects={'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef, 'bce_dice_loss': bce_dice_loss, \
        'categorical_focal_loss_fixed':categorical_focal_loss(gamma=2., alpha=.25), \
        'ce_dice_loss': ce_dice_loss})

    if args.norm != "":
        print("@ Perform data normalization ... ")
        print("- Loading form %s" %(args.norm))

        pickle_in = open(args.norm,"rb")
        stat_dict = pickle.load(pickle_in)
        print("- Training Data statistics m=%f, s=%f" %(stat_dict["m"], stat_dict["s"]))

    #############################################################################
    print("@ loading signal files ...")
    if os.path.isdir(signal_input):
        file_list = os.listdir(signal_input)
        file_dir = signal_input
    else:
        file_list = [os.path.basename(signal_input)]
        file_dir = os.path.abspath(os.path.join(signal_input, os.path.pardir))
    ## make the subfold
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(os.path.join(args.output, 'segments')):
        os.makedirs(os.path.join(args.output, 'segments'))
    if not os.path.exists(os.path.join(args.output, 'result')):
        os.makedirs(os.path.join(args.output, 'result'))
    if not os.path.exists(os.path.join(args.output, 'meta')):
        os.makedirs(os.path.join(args.output, 'meta'))

    # start processing files in the file list
    for i,name in enumerate(file_list):
        start_time = time.time()
        if not name.endswith('.signal'):
            continue
        print("Processing signal %d"%i)
        file_pre = os.path.splitext(name)[0]
        print("- Processing read %s" %(file_pre))
        input_path = os.path.join(file_dir, name)

        # reading files, take care about normalization issue. @@ chekcing the data normalization issues
        eval_data = read_data_for_eval2(input_path, args.start, args.jump, args.segment_len)

        reads_n = eval_data.reads_n
        reading_time = time.time() - start_time

        reads = list()
        signals = np.empty((0, args.segment_len), dtype=np.float)
        raw_reads=  list()
        # doing the base-calling for the loaded signals
        for i in range(0, reads_n, args.batch_size):
            # get input signals
            X, seq_len, _, _, _, _ = eval_data.next_batch(args.batch_size, shuffle=False)

            # call different basecallers here.
            X = X.reshape(X.shape[0], X.shape[1], 1).astype("float32")
            # normalization of the data
            
            if args.norm != "":
                X = (X - stat_dict["m"])/(stat_dict["s"])
            raw_preds = unet_basecaller2(unet_model,X)
            #reads += output
            #output2 = unet_basecaller2(unet_model,X,jump = args.jump)
            if len(raw_reads)==0:
                raw_reads= np.array(raw_preds)
            else:

                raw_reads=np.concatenate((raw_reads,np.array(raw_preds)))
        
        final_read = concatenate_reads(raw_reads,jump = args.jump)
        print("Segment reads base calling finished, begin to assembly. %5.2f seconds" % (time.time() - start_time))
        basecall_time = time.time() - start_time
        

        # old way of assembling
        # doing simple assembly methods
        #print("old way of reads ")
        #print(reads[0])
        #consensus = simple_assembly(reads)
        #consensus  = simple_assembly(reads, args.jump/args.segment_len, error_rate = 0.2,kernal = 'glue')
        #c_bpread = index2base_0(np.argmax(consensus, axis=0))

        #print("Final read by simple assembly length : %d" %len(c_bpread))
        #print("Final read by arda assembly length : %d "%len(final_read))
        assembly_time = time.time() - start_time
        print("Assembly finished, begin output. %5.2f seconds" % (time.time() - start_time))

        # writing the files to the fold
        list_of_time = [start_time, reading_time, basecall_time, assembly_time]
        #write_output2(args, reads, "".join(c_bpread),"".join([x for x in final_read]), list_of_time, file_pre)
        write_output(args,raw_reads,"".join([x for x in final_read]),list_of_time,file_pre)
def evaluation(signal_input, args):

    print("@ Loading U-net model ...")
    if args.model_param != "":
        params = load_modelParam(args.model_param)
    else:
        print("! Unable to load the model parameters, pls check!")
        exit()
    model_name = get_unet_model_name(params, args)
    unet_model = models.load_model("./experiment/model/weights/" + model_name+ ".h5", \
        custom_objects={'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef, 'bce_dice_loss': bce_dice_loss, \
        'categorical_focal_loss_fixed':categorical_focal_loss(gamma=2., alpha=.25), \
        'ce_dice_loss': ce_dice_loss})

    if args.norm != "":
        print("@ Perform data normalization ... ")
        print("- Loading form %s" %(args.norm))

        pickle_in = open(args.norm,"rb")
        stat_dict = pickle.load(pickle_in)
        print("- Training Data statistics m=%f, s=%f" %(stat_dict["m"], stat_dict["s"]))

    #############################################################################
    print("@ loading signal files ...")
    if os.path.isdir(signal_input):
        file_list = os.listdir(signal_input)
        file_dir = signal_input
    else:
        file_list = [os.path.basename(signal_input)]
        file_dir = os.path.abspath(os.path.join(signal_input, os.path.pardir))
    ## make the subfold
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(os.path.join(args.output, 'segments')):
        os.makedirs(os.path.join(args.output, 'segments'))
    if not os.path.exists(os.path.join(args.output, 'result')):
        os.makedirs(os.path.join(args.output, 'result'))
    if not os.path.exists(os.path.join(args.output, 'meta')):
        os.makedirs(os.path.join(args.output, 'meta'))

    # start processing files in the file list
    for i,name in enumerate(file_list):
        start_time = time.time()
        if not name.endswith('.signal'):
            continue
        print("Processing signal %d"%i)
        file_pre = os.path.splitext(name)[0]
        print("- Processing read %s" %(file_pre))
        input_path = os.path.join(file_dir, name)

        # reading files, take care about normalization issue. @@ chekcing the data normalization issues
        eval_data = read_data_for_eval(input_path, args.start, args.jump, args.segment_len)

        reads_n = eval_data.reads_n
        reading_time = time.time() - start_time

        reads = list()
        signals = np.empty((0, args.segment_len), dtype=np.float)
        raw_reads=  list()
        # doing the base-calling for the loaded signals
        for i in range(0, reads_n, args.batch_size):
            # get input signals
            X, seq_len, _, _, _, _ = eval_data.next_batch(args.batch_size, shuffle=False)

            # call different basecallers here.
            X = X.reshape(X.shape[0], X.shape[1], 1).astype("float32")
            # normalization of the data
            
            if args.norm != "":
                X = (X - stat_dict["m"])/(stat_dict["s"])

            raw_preds = unet_basecaller2(unet_model,X,jump=args.jump)

            if len(raw_reads)==0:
                raw_reads= np.array(raw_preds)
            else:
                raw_reads=np.concatenate((raw_reads,np.array(raw_preds)))
        print("Segment reads base calling finished, begin to assembly. %5.2f seconds" % (time.time() - start_time))
        basecall_time = time.time() - start_time
        final_read = concatenate_reads(raw_reads,jump = args.jump)
        assembly_time = time.time() - start_time
        print("Assembly finished, begin output. %5.2f seconds" % (time.time() - start_time))
        # writing the files to the fold
        list_of_time = [start_time, reading_time, basecall_time, assembly_time]
        #write_output2(args, reads, "".join(c_bpread),"".join([x for x in final_read]), list_of_time, file_pre)
        write_output(args,raw_reads,"".join([x for x in final_read]),list_of_time,file_pre)

def eval_pipe_fast5(args):

    #time_dict = unix_time(evaluation, args)
    print("@ Extracting Fast5 files to Singal files ..")
    extract_fast5(args)

    # signal normlaized applied in the later stage.
    print("\n@ Basecalling for Signal files ... ")
    signal_input = args.output + "/raw/"
    evaluation(signal_input, args)


def eval_pipe_signal(args):

    # signal normlaized applied in the later stage.
    print("\n@ Basecalling for Signal files ... ")
    signal_input = args.input
    evaluation(signal_input, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='UNano', description='A Unet based basecaller.')
    parser.add_argument('-i', '--input', default='', required=True, help="File path or Folder path to the fast5 file.")
    parser.add_argument('-o', '--output', default='./experiment/fast5_eval', help="Output Folder name")
    parser.add_argument('-r', '--recursive', type=bool, default=False, help="Scan the fold recursively")
    parser.add_argument('-mode','--mode', default="dna", help="Mode, dna or rna.")


    parser.add_argument('-m', '--model', default='./experiment/model', help="model folder")
    parser.add_argument('-s', '--start', type=int, default=0, help="Start index of the signal file.")
    parser.add_argument('-b', '--batch_size', type=int, default=10000)
    parser.add_argument('-l', '--segment_len', type=int, default=300, help="Segment length to be divided into.")
    parser.add_argument('-j', '--jump', type=int, default=290, help="Step size for segment")
    
    parser.add_argument('-e', '--extension', default='fastq', help="Output file extension.")
    parser.add_argument('--concise', action='store_true', help="Concisely output the result, the meta and segments files will not be output.")
    
    # model parameters
    parser.add_argument('-mp', '--model_param', default="", required=True, help="loss function used to learn the segmentation model.")
    parser.add_argument('-loss', '--loss', default="categorical_loss", help="loss function used to learn the segmentation model.")
    parser.add_argument('-nID', '--networkID', default=3, type=int, help="Selection of different network architectures.{0:UNet_only, 1:GRU3_solo, 2:UNet_GRU3, 3:UR-net}")
    
    parser.add_argument('-norm', '--norm', default="",type=str,  help="Training data statistics of saved file")

    parser.add_argument('-it', '--input_type', default="signal", required=True, type=str,  help="Training data statistics of saved file")
    parser.add_argument('-tag', '--tag', default="",type=str,  help="Model tag information.")

    parser.add_argument('-iaw', '--inputAug_winLen', default=0,type=int,  help="input Signal augmentation with the windowScreen variance detection.")


    args = parser.parse_args(sys.argv[1:])
    
    if args.input_type == "signal":
        print("** Basecalling signal files ...")
        eval_pipe_signal(args)

    if args.input_type == "fast5":
        print("** Basecalling fast5 files ...")
        eval_pipe_fast5(args)







