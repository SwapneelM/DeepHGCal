Run TNTuples
============

This code is part of DeepHGCal but extracted to run outside without the need for installing its dependencies.

## Create the Conda Environment
```
conda env create -f DeepHGCal/deephgcal.yml -n deephgcalenv
conda activate deephgcalenv
```


## Check Available Free GPUs
```
gpustat  # check which number GPU is free on the CMG-GPU Farm and use that in place of 'x' below
export CUDA_SET_VISIBLE_DEVICES=x  # where x is an int value 0-7 representing one of the 8 CMG-GPUs
```


## Run the Training
```
cd DeepHGCal/python  # not necessary but convenient location to start the training from
python standalone-models/tntuples.py configs/Jan19cfg.ini tntuples
```

## Run Tensorboard Logging

* Requires you to open a new window/session to the CMG-GPU
```
ssh <username>@cmg-gpu1080.cern.ch
conda activate deephgcalenv
tensorboard --logdir=<summary_path_from_config>  # obtain the 'summary_path' from the 'tntuples' configuration in Jan19cfg.ini 
```

