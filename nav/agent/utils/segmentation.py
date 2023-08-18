import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import argparse
import time
import numpy as np

from detectron2.config import get_cfg, LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.model_zoo import get_config
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import DefaultPredictor
import detectron2.data.transforms as T


def debug_tensor(label, tensor):
    print(label, tensor.size(), tensor.mean().item(), tensor.std().item())


class SemanticPredMaskRCNN():

    def __init__(self, args):
        cfg = get_cfg()
        cfg.merge_from_file('nav/agent/utils/COCO-InstSeg/mask_rcnn_R_101_cat9.yaml')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.sem_pred_prob_thr
        cfg.MODEL.WEIGHTS = args.seg_model_wts
        cfg.MODEL.DEVICE = args.sem_gpu_id
        
        self.n_cats = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.predictor = DefaultPredictor(cfg)
        self.args = args

    def get_prediction(self, img, depth=None, goal_cat=None):
        args = self.args
    
        img = img[:, :, ::-1]
        pred_instances = self.predictor(img)["instances"]
        
        semantic_input = torch.zeros(img.shape[0], img.shape[1], self.n_cats + 1, device=args.sem_gpu_id)
        for j, class_idx in enumerate(pred_instances.pred_classes.cpu().numpy()):
            if class_idx in range(self.n_cats):
                idx = class_idx
                confscore = pred_instances.scores[j]
                
                # Higher threshold for target category
                if (confscore < args.sem_pred_prob_thr): 
                    continue
                if idx == goal_cat:
                    if confscore < args.goal_thr:
                        continue
                obj_mask = pred_instances.pred_masks[j] * 1.
                semantic_input[:, :, idx] += obj_mask
        
        return semantic_input.cpu().numpy(), img


def compress_sem_map(sem_map):
    c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
    for i in range(sem_map.shape[0]):
        c_map[sem_map[i] > 0.] = i + 1
    return c_map
