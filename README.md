# MuLANN
Code and data for _Multi-domain adversarial learning_ by Schoenauer-Sebag A., Heinrich L., Schoenauer M., Sebag M., Wu L. and Altschuler S., accepted at ICLR 2019. [Manuscript and reviews](https://openreview.net/forum?id=Sklv5iRqYX)

Multi-domain learning (MDL) aims at obtaining a model with minimal average risk across multiple domains. Our empirical motivation is automated microscopy data, where cultured cells are imaged after being exposed to known and unknown chemical perturbations, and each dataset displays significant experimental bias. This paper presents a multi-domain adversarial learning approach, MuLANN, to leverage multiple datasets with overlapping but distinct class sets, in a semi-supervised setting.
Our contributions include: i) a bound on the average- and worst-domain risk in MDL, obtained using the H-divergence; ii) a new loss to accommodate semi-supervised multi-domain learning and domain adaptation; iii) the experimental validation of the approach, improving on the state-of-the-art on three standard image benchmarks, and a novel bioimage dataset, [Cell](https://drive.google.com/file/d/1pdVC1bQN59uWrp2OgB9sKfFBuKW_UFwv/view?usp=sharing).
# Note
I'm currently translating this repo from Torch7 to PyTorch, so if your favorite run is not yet in PyTorch, it should be very soon. 
Feel free to open up an issue in the meanwhile, so that I can make it a priority.

# Table of contents

1. [Mnist runs](#mnist-runs)
1. [Office runs](#office-runs)
2. [Cell runs](#cell-dataset-and-runs)

# MNIST runs

## Dependencies
You need PyTorch and Python2 or 3. Specific packages:

* tensorboard
* torchvision

## Get MNIST and MNIST-M
Datasets will be downloaded by default, in the <data_dir> of your choice, when you launch training.

## Launch a run
```
$ cd <code_folder>
$ python pysrc/mnist_exp.py --data_folder <data_dir> --result_folder <result_dir>
```
Other options:
* gpu: number of desired GPU
* num_workers: number of workers to compute mini-batches
* seed: number for random seed
* eta0: learning rate
* domain_method: DANN, MADA or MuLANN (only DANN and no DA implemented in PyTorch currently)
* source: either mnist or mnistm
* target: either mnist or mnistm
* domain_lambda: no DA if no value is provided, otherwise value for hyper-parameter lambda
* info_zeta: value for hyper-parameter zeta (MuLANN)
* unknown_perc: value for hyper-parameter p (MuLANN)

# Office runs 

## Dependencies
You need Torch7 and Python2 or 3. Specific packages:

* Torch7: csvigo, loadcaffe, threads (use $ luarocks install csvigo && luarocks install loadcaffe && luarocks install threads). To use our MADA implementation, one needs to fork [my fork](https://github.com/aschoenauer-sebag/nn) of the Torch package nn, or at least copy OuterProduct.lua in `<where you installed Torch>/install/share/lua/5.1/nn` (as well as add `require('nn.OuterProduct')` in the `nn/init.lua` file).
* Python: opencv

## Get the Office data and VGG-16 model
1. Download Office from [here](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE). Let us call <office_dir> where you unpacked it.

2. Prepare everything (down-scaling images to 3x256x256, creating folders, and dl VGG-16):
```
$ cd <code_folder>
$ python pysrc/data/preparation_office.py --data_folder <office_dir>
```

If you already have the Office dataset downscaled somewhere, just add `office_folder=<office_folder>` at the end of `luasrc/settings.lua`. You still need to download VGG-16 one way or another, e.g. by commenting l. 76 of `prepation_office.py` and still running it.
## Launch a run
```
$ cd <code_folder>/luasrc
$ THC_CACHING_ALLOCATOR=0 th office_script.lua eta0 source target lambda fold fully_transductive method zeta p
```
where
* eta0: learning rate
* source: either amazon, webcam or dslr
* target: either amazon, webcam or dslr
* lambda: -1 for no DA, otherwise value for hyper-parameter lambda
* fold: training fold (between 0 and 4)
* fully_transductive: true or false, to use the same unlabeled images for train and prediction or different ones
* method: DANN, MADA or MuLANN
* zeta: value for hyper-parameter zeta
* p: value for hyper-parameter p

For example, to reproduce the paper results for MuLANN on A>W for the first fold:
```
$ cd <code_folder>/luasrc
$ THC_CACHING_ALLOCATOR=0 th office_script.lua 0.0001 amazon webcam 0.1 0 true MuLANN 0.1 0.7
```
## Launch a run with class asymmetry
```
$ cd <code_folder>/luasrc
$ THC_CACHING_ALLOCATOR=0 th office_script.lua eta0 source target lambda fold fully_transductive method zeta p asymmetry_type
```
where the arguments are the same as above, and
* asymmetry_type: can be 'symm', 'full', 'labelled' or 'unlabelled'

For example, to reproduce the paper results for MuLANN on A>W for the first fold, in the case of full asymmetry:
```
$ cd <code_folder>/luasrc
$ THC_CACHING_ALLOCATOR=0 th office_script.lua 0.0001 amazon webcam 0.1 0 true MuLANN 0.1 0.7 full
```

# Cell dataset and runs
## Dependencies
You need Torch7, FiJi (if England dataset will be used) and Python2 or 3. Specific packages:

* Torch7: csvigo, loadcaffe, threads (use $ luarocks install csvigo && luarocks install loadcaffe && luarocks install threads). To use our MADA implementation, one needs to fork [my fork](https://github.com/aschoenauer-sebag/nn) of the Torch package nn, or at least copy OuterProduct.lua in `<where you installed Torch>/install/share/lua/5.1/nn` (as well as add `require('nn.OuterProduct')` in the `nn/init.lua` file).
* Python: opencv

## Get the data and VGG-16 model, pre-process the data
1. Download California and Texas from [here](https://drive.google.com/file/d/1pdVC1bQN59uWrp2OgB9sKfFBuKW_UFwv/view?usp=sharing). Let us call <cell_dir> where you unpacked it.

2. Prepare (creating folders, and dl VGG-16):
```
$ cd <code_folder>
$ python pysrc/data/preparation_Texas_California.py --data_folder <cell_dir>
```

3. If you just want Texas and California, you can stop here. If you also want England, follow me. We will download the dataset from the database where it is stored [Ljosa et al., 2012],
then we will stitch the images together, and finally down-scale them to the same size and scale as the others.
```
$ cd <code_folder>
$ python pysrc/data/dl_England.py --data_folder <cell_dir>
$ python pysrc/data/stitch_England.py --data_folder <cell_dir> --fiji <location of your FiJi executable, eg link to ImageJ-linux64>
$ python pysrc/data/scale_England.py  --data_folder <cell_dir>
```

## Launch a run
```
$ cd <code_folder>/luasrc
$ THC_CACHING_ALLOCATOR=0 th bio_script.lua eta0 source target lambda fold method zeta p
```
where
* eta0: learning rate
* source: either California, Texas or England
* target: either California, Texas or England
* lambda: -1 for no DA, otherwise value for hyper-parameter lambda
* fold: training fold (between 0 and 4)
* fully_transductive: true or false, to use the same unlabeled images for train and prediction or different ones
* method: DANN, MADA or MuLANN
* zeta: value for hyper-parameter zeta
* p: value for hyper-parameter p

### License
Copyright 2018-2021, University of California, San Francisco

Author: Alice Schoenauer Sebag for the Altschuler and Wu Lab

All rights reserved.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more [details](http://www.gnu.org/licenses/).
