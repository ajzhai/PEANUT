import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import mmseg

import mmcv
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import weighted_loss
from mmcv import Config
from mmseg.apis import set_random_seed
from mmseg.utils import get_device
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor


NUM_TARGET_CATEGORIES = 6
NUM_EXTRA_CATEGORIES = 3


@PIPELINES.register_module()
class LoadMapFromFile(object):
    """
    Load semantic maps from file.
    Requires key "img_info" (a dict that must contain the key "filename"). 
    """

    def __init__(self,
                 to_float32=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='np'):
        self.to_float32 = to_float32
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """
        Call functions to load image and get image meta information.
        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        maps = np.load(filename)
        if filename[-1] == 'z':
            maps = maps['maps']
        img = maps[results['img_info']['t_idx']].transpose(1, 2, 0)
        img = img.astype(np.float32) / 255.

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = img.shape[0]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        
        mask = (img[:, :, 1] > 0)
        goals = range(4, 4 + NUM_TARGET_CATEGORIES)  # channels of semantic map
        
        # Setting the "ground-truth" for prediction here
        results['gt_semantic_seg'] = (maps[-1, goals] * (1 - mask)).transpose(1, 2, 0)
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
    
    
@DATASETS.register_module()
class SemMapDataset(CustomDataset):
    
    CLASSES = ['chair', 'couch', 'potted plant', 'bed', 'toilet', 'tv', 'dining-table', 'oven', 
              'sink', 'refrigerator', 'book', 'clock', 'vase', 'cup', 'bottle']
    PALETTE = np.array([
        1.0, 1.0, 1.0,
        0.6, 0.6, 0.6,
        0.95, 0.95, 0.95,
        0.96, 0.36, 0.26,
        0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
        0.9400000000000001, 0.7818, 0.66,
        0.9400000000000001, 0.8868, 0.66,
        0.8882000000000001, 0.9400000000000001, 0.66,
        0.7832000000000001, 0.9400000000000001, 0.66,
        0.6782000000000001, 0.9400000000000001, 0.66,
        0.66, 0.9400000000000001, 0.7468000000000001,
        0.66, 0.9400000000000001, 0.8518000000000001,
        0.66, 0.9232, 0.9400000000000001,
        0.66, 0.8182, 0.9400000000000001,
        0.66, 0.7132, 0.9400000000000001,
        0.7117999999999999, 0.66, 0.9400000000000001,
        0.8168, 0.66, 0.9400000000000001,
        0.9218, 0.66, 0.9400000000000001,
        0.9400000000000001, 0.66, 0.8531999999999998,
        0.9400000000000001, 0.66, 0.748199999999999]).reshape((20, 3)) * 255
    
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.npz', seg_map_suffix='.npz', 
                         split=None, **kwargs)
        assert osp.exists(self.img_dir) 

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """
        Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []

        for img in self.file_client.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=img_suffix,
                recursive=True):
            for t_idx in range(10):  # use first 10 timesteps as partial map inputs
                img_info = dict(filename=img)
                img_info['t_idx'] = t_idx
                img_infos.append(img_info)
        
        img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos
    
    def get_ann_info(self, idx):
        """
        Get annotation by index.
        """
        # We don't have separate annotation files, everything is in the map sequence
        return None
    

@weighted_loss
def my_loss(pred, target):
    target = torch.permute(target, (0, 3, 1, 2))
    assert pred.size() == target.size() and target.numel() > 0
    wts = [36.64341412, 30.19407855, 106.23704066, 25.58503269, 100.4556983, 167.64383946]  # inverse frequency
    pos_weight = torch.ones(pred[0].shape).to(pred.device) 
    for i, wt in enumerate(wts):
        pos_weight[i] = wts[i]

    loss = F.binary_cross_entropy_with_logits(pred, target / 255., reduction='none')  # no weighting
    # loss = F.binary_cross_entropy_with_logits(pred, target / 255., reduction='none', pos_weight=pos_weight)
    return loss


@LOSSES.register_module
class MyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * my_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
    
    @property
    def loss_name(self):
        return 'loss_bce'


if __name__ == '__main__':
        
    cfg = Config.fromfile('configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py')

    # Since we use only one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.backbone.in_channels = 4 + NUM_TARGET_CATEGORIES + NUM_EXTRA_CATEGORIES + 1
    cfg.model.decode_head.num_classes = NUM_TARGET_CATEGORIES
    cfg.model.decode_head.loss_decode = dict(type='MyLoss', loss_weight=1.0)
    
    cfg.model.auxiliary_head.num_classes = NUM_TARGET_CATEGORIES
    cfg.model.auxiliary_head.loss_decode = dict(type='MyLoss', loss_weight=0.4)
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

    # Modify dataset type and path
    cfg.dataset_type = 'SemMapDataset'
    cfg.data_root = '../data/saved_maps'

    cfg.data.samples_per_gpu = 8
    cfg.data.workers_per_gpu = 8

    cfg.img_norm_cfg = dict(
        mean=[0, 0, 0], std=[1 ,1, 1], to_rgb=False)

    orig_in_size = 960   # the map size
    in_size = orig_in_size
    cfg.crop_size = (in_size, in_size)
    cfg.train_pipeline = [
        dict(type='LoadMapFromFile'),
        dict(type='Resize', img_scale=None, ratio_range=(in_size / orig_in_size, in_size / orig_in_size)),
        dict(type='Pad', size=(int(in_size * 1.25), int(in_size * 1.25)), pad_val=0, seg_pad_val=0),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=1.),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='RandomRotate', prob=1., degree=180, pad_val=0, seg_pad_val=0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

    cfg.test_pipeline = [
        dict(type='LoadMapFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=None,
            img_ratios=[in_size / orig_in_size],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = 'train' 
    cfg.data.train.ann_dir = None
    cfg.data.train.pipeline = cfg.train_pipeline

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = 'val'
    cfg.data.train.ann_dir = None
    cfg.data.val.pipeline = cfg.test_pipeline

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = 'val'  
    cfg.data.train.ann_dir = None
    cfg.data.test.pipeline = cfg.test_pipeline

    # Set up working dir to save files and logs.  
    cfg.work_dir =  '../data/work_dirs/final_model' 

    cfg.runner.max_iters = 60000
    cfg.log_config.interval = 500
    cfg.evaluation.interval = cfg.runner.max_iters + 1  
    cfg.checkpoint_config.interval = 2000
    cfg.optimizer = optimizer = dict(type='Adam', lr=0.0005)
    cfg.lr_config.min_lr = 1e-5
    
    # Set seed to facilitate reproducing the result
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device = get_device()

    # Let's have a look at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')
    
    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_segmentor(cfg.model)
    
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, 'cfg.py'))
    model.train()
    train_segmentor(model, datasets, cfg, distributed=False, validate=False, 
                    meta=dict())


