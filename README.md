# URnano: Nanopore base-calling from a perspective of instance segmentation

[Notification]: We are now updating package versions and developing several new features (e.g. tensorflow2, data I/O, Bonito train data benchmarking, etc.).
The next version release is scheduled on 2021/11/30.

URnano is a nanopore base-caller that performing base-calling as a multi-label segmenation task.
More details can be found in [BMC Bioinformatics](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-3459-0).

## Docker enviroment
We provide a docker image for running this code
```
docker pull yaozhong/keras_r_tf:1.8
```

* ubuntu 14.04.4
* Tensorflow 1.8.0
* Keras 2.2.4

```
nvidia-docker run -it --rm yaozhong/keras_r_tf:1.8 bash
```

## Data
We use curated data provided by Chiron, which can be downloaded from http://dx.doi.org/10.5524/100425.

## Training
```
# setting model parameters
## Network structure
MODEL="../experiment/model/URnet.model.parameters.json"
## Training data cache
TRAINDATA="../data/cache/train.h5"
## If no data cache is given, provide the signal and labels as used in Chiron.
TRAIN_IN="../data/chiron_data/train/"
## Currency strength statistics saving path
NORM_FILE_SAVE="../experiment/model/statistics/all_data_stats.pickle"

python train_urnet.py -cf $TRAINDATA -i $TRAIN_IN -mp $MODEL -l $LOSS -nID 3 -fSignal 10 -norm $NORM_FILE_SAVE -tag URnet.all
```


## Testing
### (1). Non-overlapping evaluation
```
## Network structure
MODEL="../experiment/model/URnet.model.parameters.json"
LOSS="ce_dice_loss"
TESTDATA="../data/cache/test.h5"
## If no data cache is given, provide the signal and labels as used in Chiron.
TEST_IN="../data/chiron_data/test/"
## Currency strength statistics saving path
NORM_FILE_SAVE="../experiment/model/statistics/all_data_stats.pickle"

python test_urnet.py -tm plt -cf $TESTDATA -i $TEST_IN -mp $MODEL -l $LOSS -nID 3 -norm $NORM_FILE_SAVE -tag URnet.all
```


### (2). whole read base-calling from fast5
```
spiece="ecoli"
SIGNAL_FOLD="../data/chiron_data/paper_eval/unet_result/signals/ecoli/"
OUTPUT="../experiment/basecall/basecalling_clip/$spiece/"
MODEL="../experiment/model/Unet.model.parameters.json"
LOSS="ce_dice_loss"
NORM_FILE_SAVE="../experiment/model/statistics/all_data_stats.pickle"

python fast5_test_urnet.py -i $SIGNAL_FOLD -it signal -o $OUTPUT -mp $MODEL -loss $LOSS -nID 3 -norm $NORM_FILE_SAVE -tag URnet.all
```


## Acknowledgement
We thank Chiron authors for providing [source code](https://github.com/haotianteng/Chiron) and [dataset](http://gigadb.org/dataset/100425).
The signal reading part and merging of base-calling results for a whole read part are revised based on Chiron (V0.3)'s code following MPL 2.0.
