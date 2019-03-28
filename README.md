DeepHGCal 
=========

Based on the DeepJetCore framework (https://github.com/DL4Jets/DeepJetCore) [CMS-AN-17-126] for HGCal reconstruction purposes.

The framework consists of two parts:
1) HGCal ntuple converter (Dependencies: root5.3/6)
2) DNN training/evaluation (Dependencies: DeepJetCore and all therein).
   
The DeepJetCore framework and the DeepHGCal framework should be checked out to the same parent directory.
Before usage, always set up the environment by sourcing XXX_env.sh

## Installation

```
git clone https://github.com/DL4Jets/DeepJetCore.git
git clone https://github.com/jkiesele/DeepHGCal.git
cd DeepHGCal
conda env create -f deephgcal.yml
source lxplus_env.sh
```

* You will need the correct version of `tensorflow`, `tensorflow-gpu`(1.8.0) and `cudnn` in order to run on the CMG GPU.
```
conda install -c anaconda cudnn=7.1.2  # also installs cudatoolkit=9.0
```

* You will (probably) need to symlink (`ln -s`) your libstdc++.so.6 from libstdc++.so.6.0.19 to 6.0.25
```
unlink $CONDA_PREFIX/lib/libstdc++.so.6
ln -s $CONDA_PREFIX/lib/libstdc++.so.6.0.25 $CONDA_PREFIX/lib/libstdc++.so.6

```


## Usage

The experiments are usually conducted in three steps:
1. Training
2. Testing (dumping of inference result somewhere on disk)
3. Plotting and anlysis

### Training

* When training for the first time, ensure that the `from_scratch` configuration value is set to 1.
* Check using the `gpustat` command and accordingly set the environment variable `CUDA_VISIBLE_DEVICES` to the available GPU ID.
* Check out off-the-shelf combinations of training configurations:

| Path to Train File                    | Path to Config File   | Config Name                   |
| :------------------------------------:|:---------------------:|:-----------------------------:|
| bin/train/sparse_conv_clustering      | configs/Jan19cfg.ini  | single_neighbours             |
| bin/train/sparse_conv_clustering      | configs/Jan19cfg.ini  | hidden_aggregators_plusmean   |
| bin/train/sparse_conv_clustering      | configs/Jan19cfg.ini  | tntuples                      |


``` 
python <path to train file> <path to config file> config_name
```


### Testing

```
python <path to train file> <path to config file> config_name --test True
```


### Plotting and analysis

```
python bin/plot/plot_file.py <path to config file> config_name
```

For clustering, the plot_file can be `plot_inference_clustering.py`

It will plot the resolution histogram as well as output mean and variance of resolution on stdout.


### Issues

1. Tensorflow and CUDA compatibility issue: Check thread on [Tensorflow's Github Issues](https://github.com/tensorflow/tensorflow/issues/15604) for your specific version of `libcublas.so`

2. `No module named <module_name>`: This means you have not initialized the `$PYTHONPATH`, `$LD_PRELOAD` or `$PATH` and the script cannot find the modules you want to import.
(Note that `DeepHGCal/python` must be placed on the `$PYTHONPATH` for the internal modules to be found)

3. `KeyError`: Ensure that you are using the correct configuration file with the model of your choice and it has all the necessary keys initialized.

4. `PermissionError: [Errno 1] Operation not permitted`: This is most likely a file-writing error and is expected for an off-the-shelf run of the model training. Modify the config to point to your own output directory where you do have write permission. 