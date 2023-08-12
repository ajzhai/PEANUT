###  Instructions

* Make a folder `habitat-challenge-data/data/scene_datasets/hm3d`
* Download [train](https://api.matterport.com/resources/habitat/hm3d-train-habitat.tar) and [val](https://api.matterport.com/resources/habitat/hm3d-val-habitat.tar) scenes and extract in `habitat-challenge-data/data/scene_datasets/hm3d/<split>` so that you have `habitat-challenge-data/data/scene_datasets/hm3d/val/00800-TEEsavR23oF` etc.
* Download [episode dataset](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip) and extract in `habitat-challenge-data` so that you have `habitat-challenge-data/objectgoal_hm3d/val` etc.
* Download [Mask-RCNN weights](https://drive.google.com/file/d/1tJ9MFK6Th7SY1iJTPrtpOmXNHB4ztPxC/view?usp=share_link) and place in `nav/agent/utils/mask_rcnn_R_101_cat9.pth`
* Download [prediction network weights](https://drive.google.com/file/d/1Xvly65BKVyy92Ja5GL7YwxryDrsnyO05/view?usp=share_link) and place in `nav/pred_model_wts.pth`
* Run `sh build_and_run.sh`