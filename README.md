# Affective Behaviour Analysis Using Pretrained Model with Facial Priori for ABAW4 
[[Paper](https://arxiv.org/abs/2207.11679)], [[slides](https://pan.baidu.com/s/1fojslUcjZwDOjKFMQ8uvRQ?pwd=ABAW)] (code: ABAW), [[video](https://pan.baidu.com/s/12SUQ1mvpH4-sv7_oYWA7Hw?pwd=ABAW)] (code: ABAW)

This repository is the codebase for [ABAW4](https://ibug.doc.ic.ac.uk/resources/eccv-2023-4th-abaw/) challenge, which includes EMMA for multi-task-learning (MTL) and masked CoTEX for learning from synthetic data (LSD) challenge.

## Citing this paper
If you find this repo is useful, please cite the following BibTeX entry. Thank you!
```
@inproceedings{li2023affective,
  title={Affective Behaviour Analysis Using Pretrained Model with Facial Prior},
  author={Li, Yifan and Sun, Haomiao and Liu, Zhaori and Han, Hu and Shan, Shiguang},
  booktitle={European Conference on Computer Vision Workshop},
  pages={19--30},
  year={2023},
  organization={Springer}
}
```

## Pretrained models
The pretrained models for EMMA and COTEX are provided through the following urls:

<b>MAE ViT pretrained on CelebA</b> \[[link](https://pan.baidu.com/s/1aedEeEHeIslvx0WsFVWxDw)\] (code: ABAW) \
<b>DAN pretrained on AffectNet</b> \[[link](https://pan.baidu.com/s/1MNSkd7KWSL5USywPG3XVfw)\] (code: ABAW)

We also provide the pretrained EMMA models:

<b>EMMA</b> \[[link](https://pan.baidu.com/s/12xTjIqhTUdp_FziBNTEd0A?pwd=ABAW)\] (code: ABAW)


## Requirements
This codebase is based on `Python 3.7`. 
Ensure you have installed all the necessary Python packages, run `python install -r requirements.txt`

## Data
Please download the ABAW4 data including MTL and LSD before running the code. 

## Training
### EMMA
- First you need to change the pretrained model and dataset directories in the script [`shs/train_EMMA.sh`](./shs/train_EMMA.sh)

- Second, run the following command:

```
sh shs/train_EMMA.sh
```
### Masked CoTEX
- First you need to change the pretrained model and dataset directories in the script [`shs/train_masked_CoTEX.sh`](./shs/train_masked_CoTEX.sh)

- Second, run the following command:

```
sh shs/train_masked_CoTEX.sh
```

## Reference

This code refers to masked auto-encoder ([MAE](https://github.com/facebookresearch/mae)) and [DAN](https://github.com/yaoing/DAN). Thank you!

