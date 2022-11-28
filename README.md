# CFL

This repository contains the code used for the following two papers: "General Cyclical Training of Neural Networks" and "Cyclical Focal Loss". These new papers are posted to arXiv.org at https://arxiv.org/pdf/2202.08835 and https://arxiv.org/pdf/2202.08978, respctively.

These papers describe several novel techniques that are implemented in this code: cyclical focal loss, cyclical weight decay, cyclical softmax temperature, and cyclical gradient clipping. 

## Setup

We modified the implementation of the asymmetric focal loss to include a cyclical focal loss option.  The original code is available at https://github.com/Alibaba-MIIL/ASL.  The modified version is available here in the file `CFL/TIMM/timm/loss/asl_focal_loss.py`.

In addition, we used *PyTorch Image Models (timm)* as the main framework in our experiments on CIFAR and ImageNet.  The original code is available at https://github.com/rwightman/pytorch-image-models.  To get set up, go to the documentation page at https://rwightman.github.io/pytorch-image-models/ and follow the setup instructions.   We advise downloading and running this code prior to adding our code.  Then add `CFL/TIMM/timm/loss/asl_focal_loss.py` to your `./timm/loss/` directory.  Next you will need to use the `diff_train.py` file to update the `TIMM\train.py` file because *timm* is evolving and copying `TIMM\train.py` is unlikely to work.  This directory provides `TIMM\train.py` for reference only.   In order to run the versions of CIFAR-10 and CIFAR-100 with 4,000 training samples, you will need to use the `diff_loader.py` file to update `./data/loader.py` because *timm* is evolving and replacement might not work.

Furthermore, we made use of the code provided for the open long tailed recognition experiments.  The original code is available at https://github.com/zhmiao/OpenLongTailRecognition-OLTR.   To get set up, go to this URL and follow the instructions for Data Preparation and getting started.  We advise downloading and running this code prior to adding our code.  Then add `CFL/OLTR/loss/asl_focal_loss.py` to your `./loss/` directory.   Next add the files `stage_1_cfl.py` and `stage_1_cfl_meta_embedding.py` to `./config/Places_LT` and `./config/ImageNet_LT`.  Finally, you can make a copy or replace  `main.py` and `run_networks.py`.

ImageNet is needed for these experiments and it is available at https://image-net.org/index.  

## Running

For running cyclical focal loss with the CIFAR-10 dataset the following command line is an example of what was used:
```bash
	CUDA_VISIBLE_DEVICES=0 python train.py  data/cifar10 --dataset torch/cifar10
	-b 384 --model tresnet_m --checkpoint-hist 4 --sched cosine --epochs 200 --lr 0.15 
	--warmup-lr 0.01 --warmup-epochs 3 --cooldown-epochs 1 --weight-decay 5e-4 
	--amp --remode pixel --reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 
	--resplit --split-bn  --dist-bn reduce --focal_loss asym-cyclical --gamma_hc 3 
	--gamma_pos 2 --gamma_neg 2 --cyclical_factor 4
```
Note the new input parameters of `focal_loss, gamma_hc, gamma_pos, gamma_neg, and cyclical_factor`.

For CIFAR-100, the following command line was the same as the one for CIFAR-10 except to replace cifar10 with cifar100 as follows:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py  data/cifar100 --dataset torch/cifar100 
```

An example command line for submitting an experiment on with Imagenet:
```bash
./distributed_train.sh 4 data/imagenet -b=384 --lr=0.6 --warmup-lr 0.02 
	--warmup-epochs 3 --weight-decay 2e-5 --cooldown-epochs 1 
	--model-ema --checkpoint-hist 4 --workers 8 --aa=rand-m9-mstd0.5-inc1 -j=16 --amp 
	--model=tresnet_m --epochs=200 --mixup=0.2 --sched='cosine' --reprob=0.4 
	--remode=pixel --focal_loss asym-cyclical --gamma_hc 3 --gamma_pos 2 --gamma_neg 2 
	--cyclical_factor 4
```

For cyclical weight decay, the new input parameters are:
```bash
--wd_min 1e-5 --wd_max 8e-5
```
If `wd_min` is 0, a constant weight decay is used (i.e., the weight-decay parameter).


For cyclical softmax temperature, the new input parameters are:
```bash
--T_min 1e-5 --T_max 8e-5
```
If `T_min` is 0, a constant softmax temperature of 1 is used.

For cyclical gradient clipping, the new input parameters are:
```bash
--clip_min 0.5 --clIp_max 2
```
If `clip_min` is 0, clipping is governed by clip-grad (i.e., the default is None).
