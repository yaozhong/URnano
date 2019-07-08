# URnano: Nanopore base-calling from a perspective of instance segmentation

URnano is a nanopore base-caller that 
More details can be found in <bioAxiv link>

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
We also provide data caching files can be directly used. 

## Training
```
# setting model parameters
## Network structure
MODEL="../experiment/model/hpt.11.URnet.model.parameters.json"
## Training data cache
TRAINDATA="../data/cache/train.h5"
## If not data cache is given, provide the signal and labels as used in Chiron.
TRAIN_IN="../data/chiron_data/train/"
## Currency strength statistics saving path
NORM_FILE_SAVE="../experiment/model/statistics/all_data_stats.pickle"

python train_unet_gplabel.py -cf $TRAINDATA -i $TRAIN_IN -mp $MODEL -l $LOSS -lstm 12 -fSignal 10 -norm $NORM_FILE_SAVE -tag en11.URnet.all.hpt
```


## Testing

## Acknowledgement
