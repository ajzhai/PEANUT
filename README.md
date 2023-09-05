# PEANUT: Predicting and Navigating to Unseen Targets

[Albert J. Zhai](https://ajzhai.github.io/), [Shenlong Wang](https://shenlong.web.illinois.edu/)<br/>
University of Illinois at Urbana-Champaign

ICCV 2023

[Paper](https://arxiv.org/abs/2212.02497) │ [Project Page](https://ajzhai.github.io/PEANUT/)

![Example Video](docs/example_vid.gif)

## Requirements
 As required by the [Habitat Challenge](https://github.com/facebookresearch/habitat-challenge), our code uses Docker to run. Install nvidia-docker by following the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker) (only Linux is supported). There is no need to manually install any other dependencies. However, you do need to download and place several files, as follows:


###  File Setup

* Make a folder `habitat-challenge-data/data/scene_datasets/hm3d`
* Download HM3D [train](https://api.matterport.com/resources/habitat/hm3d-train-habitat.tar) and [val](https://api.matterport.com/resources/habitat/hm3d-val-habitat.tar) scenes and extract in `habitat-challenge-data/data/scene_datasets/hm3d/<split>` so that you have `habitat-challenge-data/data/scene_datasets/hm3d/val/00800-TEEsavR23oF` etc.
* Download [episode dataset](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip) and extract in `habitat-challenge-data` so that you have `habitat-challenge-data/objectgoal_hm3d/val` etc.
* Download [Mask-RCNN weights](https://drive.google.com/file/d/1tJ9MFK6Th7SY1iJTPrtpOmXNHB4ztPxC/view?usp=share_link) and place in `nav/agent/utils/mask_rcnn_R_101_cat9.pth`
* Download [prediction network weights](https://drive.google.com/file/d/1Xvly65BKVyy92Ja5GL7YwxryDrsnyO05/view?usp=share_link) and place in `nav/pred_model_wts.pth`
  
The file structure should look like this:
```
PEANUT/
├── habitat-challenge-data/
│   ├── objectgoal_hm3d/
│   │   ├── train/
│   │   ├── val/
│   │   └── val_mini/
│   └── data/
│       └── scene_datasets/
│           └── hm3d/
│               ├── train/
│               └── val/
└── nav/
    ├── pred_model_wts.pth
    └── agent/
        └── utils/
            └── mask_rcnn_R_101_cat9.pth
```

## Usage
In general, you should modify the contents of `nav_exp.sh` to run the specific Python script and command-line arguments that you want. Then, simply run
```bash
sh build_and_run.sh
```
to build and run everything in Docker. Note: depending on how Docker is setup on your system, you may need sudo for this.

### Evaluating the navigation agent
An example script for evaluating ObjectNav performance is provided in `nav/collect.py`. This script is a good entry point for understanding the code and it is what `nav_exp.sh` runs by default. See `nav/arguments.py` for available command-line arguments.

### Collecting semantic maps
An example script for collecting semantic maps and saving them as .npz files is provided in `nav/collect_maps.py`. A link to download the original map dataset used in the paper is provided below.

### Training the prediction model
We use MMSegmentation to train and run PEANUT's prediction model. A custom clone of MMSegmentation is contained in `prediction/`, and a training script is provided in `prediction/train_prediction_model.py`. Please see the MMSegmentation docs in the `prediction/` folder for more info about how to use MMSegmentation.


## Semantic Map Dataset
The original map dataset used in the paper can be downloaded from [this Google Drive link](https://drive.google.com/file/d/134omZAAu_zYUaOYuNQcDMPhZCdxV0zbZ/view?usp=sharing). 

It contains sequences of semantic maps from 5000 episodes (4000 train, 1000 val) of [Stubborn](https://github.com/Improbable-AI/Stubborn)-based exploration  in HM3D. This dataset can be directly used to train a target prediction model using `prediction/train_prediction_model.py`.


## Citation

Please cite our paper if you find this repo useful!
```bibtex
@inproceedings{zhai2023peanut,
  title={{PEANUT}: Predicting and Navigating to Unseen Targets},
  author={Zhai, Albert J and Wang, Shenlong},
  booktitle={ICCV},
  year={2023}
}
```

## Acknowledgments
This project builds upon code from [Stubborn](https://github.com/Improbable-AI/Stubborn), [SemExp](https://github.com/devendrachaplot/Object-Goal-Navigation), and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). We thank the authors of these projects for their amazing work!
