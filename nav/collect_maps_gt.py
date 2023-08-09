import argparse
import os
import random
import habitat
import torch
import sys
import cv2
from arguments import get_args
from habitat.core.env import Env
from constants import hm3d_names, mpcat40_labels, raw_name_to_mpcat40
import numpy as np
import matplotlib.pyplot as plt

from agent.peanut_agent import PEANUT_Agent

categories9 = ['chair', 'sofa', 'plant', 'bed', 'toilet', 'tv_monitor',  
              'fireplace', 'bathtub', 'mirror']

def shuffle_episodes(env, shuffle_interval):
    ranges = np.arange(0, len(env.episodes), shuffle_interval)
    np.random.shuffle(ranges)
    new_episodes = []
    for r in ranges:
        new_episodes += env.episodes[r:r + shuffle_interval]
    env.episodes = new_episodes
    
def main():

    args_2 = get_args()
    args_2.only_explore = 1  ########## whether to NOT go for goal detections 
    args_2.switch_step = 999
    args_2.global_downscaling = 4
    
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    config.defrost()
    config.SEED = 100
    # config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args_2.sem_gpu_id
    config.SIMULATOR.AGENT_0.SENSORS =  ['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR']
    config.SIMULATOR.SCENE_DATASET = 'habitat-challenge-data/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'

    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 10
    config.DATASET.SPLIT = 'val'
    config.freeze()
    print(config.DATASET.SPLIT)
    
    nav_agent = PEANUT_Agent(args=args_2,task_config=config)
    args_2.use_gt_seg = 1
    nav_agent_gt = PEANUT_Agent(args=args_2,task_config=config) 
    hab_env = Env(config=config)
    
    print(len(hab_env.episodes), 'episodes in dataset')
    
    num_episodes = 200
    start = args_2.start_ep
    end = args_2.end_ep if args_2.end_ep > 0 else num_episodes
    
    save_steps = list(range(25, 525, 25))
    save_dir = './data/saved_maps/val_gt'
    try:
        os.makedirs(save_dir)
    except:
        pass
    
    count_episodes = 0
    while count_episodes < min(num_episodes, end):
        observations = hab_env.reset()
        nav_agent.reset()
        nav_agent_gt.reset()
        print(hab_env._current_episode.scene_id)
        
        annots = hab_env.sim.semantic_annotations()
        instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in annots.objects}
        instance_id_to_cat_name = {int(obj.id.split("_")[-1]): obj.category.name() for obj in annots.objects}
        
        if count_episodes >= start and count_episodes < end:

            step_i = 0
            seq_i = 0
            full_map_seq = np.zeros((len(save_steps), 4 + args_2.num_sem_categories, nav_agent.agent_states.full_w, nav_agent.agent_states.full_h), dtype=np.uint8)
            while not hab_env.episode_over:
                sys.stdout.flush()
                
                observations['new_semantic'] = np.zeros((480, 640, args_2.num_sem_categories))
                for o_i, o_id in enumerate(np.unique(observations['semantic'])):
                    o_cat = instance_id_to_cat_name[o_id]
                    if o_cat in raw_name_to_mpcat40:
                        o_cat = raw_name_to_mpcat40[o_cat]
                        if o_cat in categories9:
                            cat_id = categories9.index(o_cat)
                            observations['new_semantic'][:, :, cat_id] += observations['semantic'] == o_id
                            
                
                action = nav_agent.act(observations)
                _ = nav_agent_gt.act(observations)
                observations = hab_env.step(action)
                          
                if step_i % 100 == 0:
                    print('episode %d, step %d' % (count_episodes, step_i))
                    sys.stdout.flush()

                step_i += 1
                if step_i in save_steps:
                    full_map = nav_agent.agent_states.full_map.cpu().numpy() * 255
                    full_map_seq[seq_i] = full_map.astype(np.uint8)
                    seq_i += 1
                    
                
            full_map_gt = nav_agent_gt.agent_states.full_map.cpu().numpy() * 255
                
            np.savez_compressed(os.path.join(save_dir, 'f%05d.npz' % count_episodes), maps=full_map_seq, gt=full_map_gt.astype(np.uint8))

        count_episodes += 1
        
    

if __name__ == "__main__":
    main()
