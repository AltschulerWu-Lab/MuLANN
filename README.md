# MuLANN
Code for MuLANN publication

# Office runs

## Dependencies
You need Torch7 and Python2 or 3. Specific packages:

* Torch7: csvigo, loadcaffe, threads (use $ luarocks install csvigo && luarocks install loadcaffe && luarocks install threads). To use our MADA implementation, one needs to fork [my fork](https://github.com/aschoenauer-sebag/nn) of the Torch package nn, or at least copy OuterProduct.lua in `<where you installed Torch>/install/share/lua/5.1/nn` (as well as add `require('nn.OuterProduct')` in the `nn/init.lua` file).
* Python: opencv

## Get the Office data and VGG-16 model
1. Download Office from [here](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE). Let us call <office_dir> where you unpacked it.

2. Prepare everything (down-scaling images to 3x256x256, creating folders, and dl VGG-16):
```
$ cd <code_folder>/pysrc
$ python preparation_office.py --data_folder <office_dir>
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
$ cd <code_folder>/pysrc
$ python preparation_Texas_California.py --data_folder <cell_dir>
```

3. If you just want Texas and California, you can stop here. If you also want England, follow me. We will download the dataset from the database where it is stored [Ljosa et al., 2012],
then we will stitch the images together, and finally down-scale them to the same size and scale as the others.
```
$ cd <code_folder>/pysrc
$ python dl_England.py --data_folder <cell_dir>
$ python stitch_England.py --data_folder <cell_dir> --fiji <location of your FiJi executable, eg link to ImageJ-linux64>
$ python scale_England.py  --data_folder <cell_dir>
```

## Launch a run
```
$ cd <code_folder>/luasrc
$ THC_CACHING_ALLOCATOR=0 th bio_script.lua eta0 source target lambda fold method zeta p
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

### License
Copyright 2018, University of California, San Francisco

Author: Alice Schoenauer Sebag for the Altschuler and Wu Lab

All rights reserved.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more [details](http://www.gnu.org/licenses/).
