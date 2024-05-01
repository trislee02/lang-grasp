# Language-Driven Grasp Detection
This is the repository of the short work "LGrasp: A End-to-End Transformer-based Network
for Language-Driven Grasp Detection"

## Table of contents
   1. [Installation](#installation)
   1. [Datasets](#datasets)
   1. [Training](#training)
   1. [Testing](#testing)

## Installation
- Create a virtual environment
```bash
$ conda create -n langrasp python=3.9
$ conda activate langrasp
```

- Install pytorch and libraries
```bash
$ conda install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
$ pip install -r requirements.txt
$ pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
$ pip install git+https://github.com/openai/CLIP.git
```

- If you encounter any issues during installing PyTorch-Encoding, please try to install it from PyPi by:
```bash
$ pip install torch-encoding
```


## Datasets
- For datasets, please obtain following their instructions: [Cornell](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp), [Jacquard](https://jacquard.liris.cnrs.fr/), [OCID-grasp](https://github.com/stefan-ainetter/grasp_det_seg_cnn), and [VMRD](https://gr.xjtu.edu.cn/zh/web/zeuslan/dataset).
- For Grasp-Anything++, use the name `grasp-anything` because they share the same folder structure.
- All datasets should be include in the following hierarchy:
```
|- data/
    |- cornell
    |- grasp-anything
        |- seen
        |- unseen
    |- jacquard
    |- OCID_grasp
    |- VMRD
```

## Training
In order to train LGrasp, you can use the following command:
```bash
$ python train_network_lgrasp_pl.py --dataset <dataset> --dataset-path <dataset> --description <your_description> --use-depth 0
```
For example, if you want to train LGrasp on Grasp-Anything++, use the following command:
```bash
$ python train_network.py --dataset grasp-anything --dataset-path data/grasp-anything --description training_grasp_anything --use-depth 0
```

## Testing
For testing procedure, we can apply the similar commands to test on different datasets:
```bash
python evaluate_lgrasp.py --network <path_to_checkpoint> --dataset <dataset> --dataset-path data/<dataset> --iou-eval
```
Important note: `<path_to_checkpoint>` is the path to the checkpoint obtained by training procedure. Usually, the checkpoints obtained by training are stored at `grasp_anything_ckpt/model-epoch=<epoch>-val_loss=<val_loss>.ckpt`. 


## Acknowledgement
Our codebase is developed based on [Vuong et al.](https://github.com/andvg3/Grasp-Anything) and [Li et al.](https://github.com/isl-org/lang-seg).
