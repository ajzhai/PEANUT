import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import argparse
import time
import numpy as np
import pickle
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
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
from constants import color_palette
import sys
import cv2



def debug_tensor(label, tensor):
    print(label, tensor.size(), tensor.mean().item(), tensor.std().item())



class SemanticPredMaskRCNN():

    def __init__(self, args):
        cfg = get_cfg()
        cfg.merge_from_file('nav/agent/utils/COCO-InstSeg/mask_rcnn_R_101_cat9.yaml')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.sem_pred_prob_thr
        cfg.MODEL.WEIGHTS = args.seg_model_wts
        cfg.MODEL.DEVICE = args.sem_gpu_id
        #cfg.TEST.AUG.MIN_SIZES = [640, 800]
        
        self.n_cats = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.predictor = DefaultPredictor(cfg)
        #self.predictor.model = GeneralizedRCNNWithTTA(cfg, self.predictor.model)
        self.args = args

    def get_prediction(self, img, depth=None, goal_cat=None):
        args = self.args
        # pdb.set_trace()
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

COLORS = np.array([
    [148,103,188],[151,226,173],[174,198,232],[31,120,180],[255,188,120],[188,189,35],
    [140,86,74],[255,152,151],[213,39,40],[0,0,0],[196,176,213],[196,156,148],
    [23,190,208],[247,183,210],[218,219,141],[254,127,14],[227,119,194],
    [158,218,229],[43,160,45],[112,128,144],[82,83,163]
]).astype(np.uint8)

class SegformerSegmenter():
    def __init__(self,args):
        self.segmenter = FineTunedTSegmenter(model_ckpt = args.segformer_ckpt)
        self.args = args
    def get_prediction(self, img, depth=None, goal_cat=None):
        args = self.args
        seg_out = self.segmenter.get_pred_probs(img)

        img = img[:,:,::-1]
        return seg_out,img

class SegformerHighPrecision():
    def __init__(self,args):
        self.segmenter = FineTunedTSegmenter(model_ckpt = args.segformer_ckpt)
        self.args = args
    def get_prediction(self, img, depth=None, goal_cat=None):
        args = self.args
        seg_out = self.segmenter.get_pred_probs(img)
        # filtering out low probability predictions:
        all_classes = set(np.arange(10))
        non_target = np.array(list(all_classes-set([goal_cat])))
        invalid_non_goal = seg_out[:,:,non_target]<self.args.sem_pred_prob_thr
        invalid_goal = seg_out[:,:,goal_cat] < self.args.goal_thr
        seg_out[:,:,non_target][invalid_non_goal] = 0
        seg_out[:,:,non_target][np.logical_not(invalid_non_goal)] = 1
        seg_out[:,:,goal_cat][invalid_goal] = 0
        seg_out[:,:,goal_cat][np.logical_not(invalid_goal)] = 1
        img = img[:,:,::-1]
        return seg_out,img
    
class FineTunedTSegmenter():
    def __init__(self,temperature = 1,model_ckpt = "./best_model"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_ckpt).to(self.device)
        self.model.eval()
        self.temperature = temperature
        
        self.softmax = nn.Softmax(dim = 1)

        # for idx,new_class in enumerate(self.class_mapping):
        #     class_matrix[idx,new_class] = 1

        # self.cm = torch.from_numpy(class_matrix.astype(np.float32)).to(self.device)
    def set_temperature(self,temperature):
        self.temperature = temperature
        
    def classify(self,rgb,depth = None,x=None,y = None,temperature = None):
        with torch.no_grad():
            image = Image.fromarray(np.uint8(rgb))
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            outputs = self.model(pixel_values=inputs['pixel_values'].to(self.device))
            logits = outputs.logits


            # pred = torch.tensordot(logits,self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)
            if((x == None) or( y == None)):
                pred = F.interpolate(logits, (image.height,image.width),mode='bilinear')
            else:
                pred = F.interpolate(logits, (x,y),mode='bilinear')

            if(temperature):
                # print('applying temperature scaling')
                pred = self.softmax(pred/temperature)
            else:
                pred = self.softmax(pred/self.temperature)

            pred = torch.argmax(pred,axis = 1)

        return pred.squeeze().detach().cpu().numpy()

    def get_pred_probs(self,rgb,depth = None,x = None,y = None,temperature = None):
        with torch.no_grad():
            image = Image.fromarray(np.uint8(rgb))
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            # print(inputs['pixel_values'].shape)
            outputs = self.model(pixel_values=inputs['pixel_values'].to(self.device))
            logits = outputs.logits
            # print(logits.shape)
            # pred = torch.tensordot(logits,self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)
            # pred = self.aggregate_logits(logits)
            pred = logits
            # print(pred.shape)
            # pred = logits.unsqueeze(0)
            
            if(temperature):
                # print('applying temperature scaling')
                pred = self.softmax(pred/temperature)
            else:
                pred = self.softmax(pred/self.temperature)
            if((x == None) or( y == None)):
                pred = F.interpolate(pred, (image.height,image.width),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')
        
        return pred.squeeze().detach().permute((1,2,0)).contiguous().cpu().numpy()


    def get_raw_logits(self,rgb,depth = None,x=None,y = None,temperature = 1):
        with torch.no_grad():
            image = Image.fromarray(np.uint8(rgb))
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            # print(inputs['pixel_values'].shape)
            outputs = self.model(pixel_values=inputs['pixel_values'].to(self.device))
            logits = outputs.logits
            if((x == None) or( y == None)):
                pred = F.interpolate(logits, (image.height,image.width),mode='nearest')
            else:
                pred = F.interpolate(logits, (x,y),mode='nearest')
        return pred.squeeze().detach().permute((1,2,0)).contiguous().cpu().numpy()
    
    def high_precision_classify(self,rgb,depth = None,x=None,y = None,temperature = 1,thold = 0.95):
        probs = self.get_pred_probs(rgb,depth,x,y,temperature)
        probs[probs>thold] = 1
        probs[probs<thold] = 0
        probs[probs.sum(axis = 2) == 0][:,9] = 1
        return probs


class ESANetClassifier:
    def __init__(self,temperature = 1,NYU = False):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        args = pickle.load(open('/workspaces/peanut-temp/nav/agent/utils/ESANet/args.p','rb'))
        self.model, self.device = build_model_ESANet(args, n_classes=40)
        args.ckpt_path = '/workspaces/peanut-temp/nav/agent/utils/ESANet/trained_models/nyuv2/r34_NBt1D_scenenet.pth'
        args.depth_scaling = 0.1
        checkpoint = torch.load(args.ckpt_path,
                                map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model.to(self.device)
        self.dataset, self.preprocessor = prepare_data(args, with_input_orig=True)
        self.class_mapping = np.array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
        0., 13.,  0., 14.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,
        0.,  0., 16.,  0.,  0.,  0.,  0., 17., 18.,  0., 19.,  0.,  0.,
        20.,  0.]).astype(np.uint8)
        class_matrix = np.zeros((40,21))
        self.temperature = temperature

        for idx,new_class in enumerate(self.class_mapping):
            class_matrix[idx,new_class] = 1

        self.cm = torch.from_numpy(class_matrix.astype(np.float32)).to(self.device)
        self.NYU = NYU

        self.pred_dist = np.zeros((480,640,21))
        self.softmax = nn.Softmax(dim = 1)
    def set_temperature(self,temperature):
        self.temperature = temperature
    def classify(self,img_rgb,depth,x = None,y = None,temperature = None):
        with torch.no_grad():
            # preprocess sample
            sample = self.preprocessor({'image': img_rgb, 'depth': depth})

            # add batch axis and copy to device
            image = sample['image'][None].to(self.device)
            depth = sample['depth'][None].to(self.device)

            # apply network
            pred = self.model(image, depth)

            if(not self.NYU):
                # Condense probabilities for unsupported classes
                pred = torch.tensordot(pred.squeeze(),self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)

            if((x is None) or (y is None)):
                pred = F.interpolate(pred, (pred.shape[2],pred.shape[3]),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')

            if(temperature is None):
                pred = pred/self.temperature
            else:
                pred = pred/temperature

            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy().squeeze().astype(np.uint8)
            # if(not self.NYU):
            #     pred = self.class_mapping[pred]
        return pred
    def get_pred_probs(self,img_rgb,depth,x = None,y = None,temperature = None):
        with torch.no_grad():
            # preprocess sample
            sample = self.preprocessor({'image': img_rgb, 'depth': depth})

            # add batch axis and copy to device
            image = sample['image'][None].to(self.device)
            depth = sample['depth'][None].to(self.device)
            pred = self.model(image, depth)

            if(not self.NYU):
                # Condense probabilities for unsupported classes
                pred = torch.tensordot(pred.squeeze(),self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)



            # print(pred.shape)
            if((x is None) or (y is None)):
                pred = F.interpolate(pred, (pred.shape[2],pred.shape[3]),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')

            if(temperature):
                # apply network
                # print('applying temperature scaling')
                pred = self.softmax(pred/temperature)
            else:
                pred = self.softmax(pred/self.temperature)

            # pred = torch.tensordot(pred.cpu(),self.cm,dims = ([0],[0]))
            # pred = pred.cpu().squeeze().permute(1,2,0).detach().numpy()
            # self.pred_dist[:,:,:] = 0
            # for idx,new_class in enumerate(self.class_mapping):
            #     self.pred_dist[:,:,new_class] += pred[:,:,idx]
        return pred.squeeze().detach().permute((1,2,0)).contiguous().cpu().numpy()

    def get_logits(self,img_rgb,depth,x = None,y = None,temperature = None):
        with torch.no_grad():
            sample = self.preprocessor({'image': img_rgb, 'depth': depth})

            # add batch axis and copy to device
            image = sample['image'][None].to(self.device)
            depth = sample['depth'][None].to(self.device)
            # print(pred.shape)
            pred = self.model(image, depth)
            if((x is None) or (y is None)):
                pred = F.interpolate(pred, (pred.shape[2],pred.shape[3]),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')

            if(not self.NYU):
                pred = torch.tensordot(pred.squeeze(),self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)
            
            if(temperature is None):
                pred = pred/self.temperature
            else:
                pred = pred/temperature

            if((x is None) or (y is None)):
                pred = F.interpolate(pred, (pred.shape[2],pred.shape[3]),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')
            # pred = pred.cpu().squeeze().permute(1,2,0).detach().numpy()
            # self.pred_dist[:,:,:] = 0
            # for idx,new_class in enumerate(self.class_mapping):
            #     self.pred_dist[:,:,new_class] += pred[:,:,idx]
        return pred.detach().cpu().numpy().squeeze()

    def get_logits_and_preds(self,img_rgb,depth,x = None,y = None,temperature = None):
        with torch.no_grad():
            sample = self.preprocessor({'image': img_rgb, 'depth': depth})

            # add batch axis and copy to device
            image = sample['image'][None].to(self.device)
            depth = sample['depth'][None].to(self.device)
            # print(pred.shape)
            pred = self.model(image, depth)
            if((x is None) or (y is None)):
                pred = F.interpolate(pred, (pred.shape[2],pred.shape[3]),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')

            if(not self.NYU):
                pred = torch.tensordot(pred.squeeze(),self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)

            if(temperature is None):
                pred = pred/self.temperature
            else:
                pred = pred/temperature


            if((x is None) or (y is None)):
                pred = F.interpolate(pred, (pred.shape[2],pred.shape[3]),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')

            pred_classes = torch.argmax(pred, dim=1).detach().squeeze().cpu().numpy()
        # pred = pred.cpu().squeeze().permute(1,2,0).detach().numpy()
        # self.pred_dist[:,:,:] = 0
        # for idx,new_class in enumerate(self.class_mapping):
        #     self.pred_dist[:,:,new_class] += pred[:,:,idx]
        return pred.detach().cpu().numpy().squeeze(),pred_classes

class FineTunedESANet(ESANetClassifier):
    def __init__(self,temperature = 1,checkpoint = './artifacts/Try2:v79'):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        super().__init__(temperature = temperature,NYU = True)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # decoder_head = model.decode_head
        self.model.decoder.conv_out = nn.Conv2d(
            128, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.model.decoder.conv_out.requires_grad = False
        self.model.decoder.upsample1 = Upsample(mode='nearest', channels=10)
        self.model.decoder.upsample2 = Upsample(mode='nearest', channels=10)
        self.checkpoint = checkpoint
        self.temperature = temperature
        state_dict = self.get_clean_state_dict()
        new_state_dict = self.get_clean_state_dict()
        self.model.load_state_dict(new_state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)
    
    def set_temperature(self,temperature):
        self.temperature = temperature

    def get_clean_state_dict(self):
        params_dict = torch.load(self.checkpoint)
        state = params_dict['state_dict']
        new_state_dict = OrderedDict()
        for param in state.keys():
            prefix,new_param = param.split('.',1)
            if(prefix != 'criterion'):
                new_state_dict.update({new_param:state[param]})
        return new_state_dict