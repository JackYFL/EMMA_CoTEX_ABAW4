# Affective Behaviour Analysis Using Pretrained Model with Facial Priori for ABAW4

This repository is the codebase for [ABAW4](https://ibug.doc.ic.ac.uk/resources/eccv-2023-4th-abaw/) challenge, which includes EMMA for multi-task-learning (MTL) and masked CoTEX for learning from synthetic data (LSD) challenge.

The urls of pretrained models are provided: MAE_ViT_pretrained_on_CelebA \[ [baidu, code: ABAW](https://pan.baidu.com/s/1aedEeEHeIslvx0WsFVWxDw) \] and AffectNet_pretrained_DAN \[ [baidu, code: ABAW](https://pan.baidu.com/s/1MNSkd7KWSL5USywPG3XVfw) \].

## Requirements
This codebase is based on `Python 3.7`. 
Ensure you have installed all the necessary Python packages, run `python install -r requirements.txt`

##Data
Please download the ABAW4 data including MTL and LSD before running the code. 

##Training
###EMMA
- First you need to change the pretrained model and dataset directories in the script [`shs/train_EMMA.sh`](./shs/train_EMMA.sh)

- Second, run the following command:

```
sh shs/train_EMMA.sh
```
###Masked CoTEX
- First you need to change the pretrained model and dataset directories in the script [`shs/train_masked_CoTEX.sh`](./shs/train_masked_CoTEX.sh)

- Second, run the following command:

```
sh shs/train_masked_CoTEX.sh
```