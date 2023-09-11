import argparse
import os
import random
import habitat
import torch
import sys
import cv2
from arguments import get_args
from habitat.core.env import Env
from constants import hm3d_names
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from agent.peanut_agent import PEANUT_Agent
import pandas as pd

def main():

    args = get_args()
    args.only_explore = 0  ########## whether to NOT go for goal detections 
    
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    config.defrost()
    config.SEED = 100
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.sem_gpu_id
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 1
    config.DATASET.SPLIT = 'val'
    config.freeze()
    
    hab_env = Env(config=config)
    nav_agent = PEANUT_Agent(args=args,task_config=config)
    print(config.DATASET.SPLIT, 'split')
    print(len(hab_env.episodes), 'episodes in dataset')
    
    num_episodes = 1000
    start = args.start_ep
    end = args.end_ep if args.end_ep > 0 else num_episodes
    
    save_steps = list(range(25, 525, 25))
    sucs, spls, ep_lens = [], [], []
    distance_to_goals = []
    soft_spls = []
    target_cats  = []
    per_target_spls = {'bed':[],'toilet':[],'chair':[],'sofa':[],'tv_monitor':[],'plant':[]}
    per_target_success = {'bed':[],'toilet':[],'chair':[],'sofa':[],'tv_monitor':[],'plant':[]}
    per_target_softsplt = {'bed':[],'toilet':[],'chair':[],'sofa':[],'tv_monitor':[],'plant':[]}
    per_target_distance_to_goal = {'bed':[],'toilet':[],'chair':[],'sofa':[],'tv_monitor':[],'plant':[]}
    ep_i = 0
    failed_eps = []
    episode_ids = []
    while ep_i < min(num_episodes, end):
        observations = hab_env.reset()
        nav_agent.reset()
        torch.cuda.empty_cache()
        o3d.core.cuda.release_cache()

        print('-' * 40)
        sys.stdout.flush()
        
        if ep_i >= start and ep_i < end:
            target_category =  hm3d_names[observations['objectgoal'][0]]
            print('Episode %d | Target: %s' % (ep_i, hm3d_names[observations['objectgoal'][0]]))
            print('Scene: %s' % hab_env._current_episode.scene_id)
            
            step_i = 0
            seq_i = 0
            
            while not hab_env.episode_over:
                action = nav_agent.act(observations)
                observations = hab_env.step(action)
                          
                if step_i % 100 == 0:
                    print('step %d...' % step_i)
                    sys.stdout.flush()

                step_i += 1
                    
            if args.only_explore == 0:
                episode_ids.append(ep_i)

                print('ended at step %d' % step_i)
                
                # Navigation metrics
                metrics = hab_env.get_metrics()
                print(metrics)
                if(metrics['success']<0.5):
                    failed_eps.append(ep_i)
                # Log the metrics (save them however you want)
                sucs.append(metrics['success'])
                per_target_spls[target_category].append(metrics['spl'])
                per_target_success[target_category].append(metrics['success'])
                per_target_softsplt[target_category].append(metrics['softspl'])
                target_cats.append(target_category)
                per_target_distance_to_goal[target_category].append(metrics['distance_to_goal'])
                spls.append(metrics['spl'])
                distance_to_goals.append(metrics['distance_to_goal'])
                soft_spls.append(metrics['softspl'])
                ep_lens.append(step_i)
                print('-' * 40)
                print('Average Success: %.4f | Average SPL: %.4f | Average Dist To Goal %.4f | Average SoftSPL %.4f' % (np.mean(sucs), np.mean(spls),np.mean(distance_to_goals),np.mean(soft_spls)))
                print('-' * 40)
                results_summary = {'spl':spls,'success':sucs,'softspl':soft_spls,'distance_to_goals':distance_to_goals,'target':target_cats,'episode_id':episode_ids,'ep_length':ep_lens}
                df = pd.DataFrame(results_summary)
                df.to_csv('./logs/{}'.format(args.perf_log_name),sep = '|',index = False)
                sys.stdout.flush()
                
        ep_i += 1
    
    print('\n\n\n\n')
    print('Final Results - Aggregate')
    print('-' * 40)
    print('Average Success: %.4f | Average SPL: %.4f | Average Dist To Goal %.4f | Average SoftSPL %.4f' % (np.mean(sucs), np.mean(spls),np.mean(distance_to_goals),np.mean(soft_spls)))
    print('-' * 40)
    print('\n\n\n\n')
    for key in per_target_spls.keys():
        print('Final Results - {}'.format(key))
        print('-' * 40)
        print('Average Success: %.4f | Average SPL: %.4f | Average Dist To Goal %.4f | Average SoftSPL %.4f' % (np.mean(per_target_success[key]), np.mean(per_target_spls[key]),np.mean(per_target_distance_to_goal[key]),np.mean(per_target_softsplt[key])))
        print('-' * 40,'\n')
    print(failed_eps)
    results_summary = {'spl':spls,'success':sucs,'softspl':soft_spls,'distance_to_goals':distance_to_goals,'target':target_cats,'episode_id':episode_ids,'ep_length':ep_lens}
    df = pd.DataFrame(results_summary)
    df.to_csv('./logs/{}'.format(args.perf_log_name),sep = '|',index = False)
if __name__ == "__main__":
    main()
