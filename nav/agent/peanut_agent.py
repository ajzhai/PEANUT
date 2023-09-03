import argparse
import os
import random
import habitat
import torch
from arguments import get_args
import numpy as np
import agent.utils.pose as pu
from constants import hm3d_names, hm3d_to_coco
import copy
from agent.agent_state import Agent_State,Traditional_Agent_State
from agent.agent_helper import Agent_Helper
import pdb

class PEANUT_Agent(habitat.Agent):
    def __init__(self, args, task_config: habitat.Config):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.mapping_strategy = args.mapping_strategy
        assert self.mapping_strategy in ['neural','traditional'],"Only valid mapping strategies are ['neural','traditional'], but {} was chosen".format(self.mapping_strategy)
        if(self.mapping_strategy == 'neural'):
            self.agent_states = Agent_State(args)
        else:
            self.agent_states = Traditional_Agent_State(args)
        self.agent_helper = Agent_Helper(args, self.agent_states)
        self.agent_states.helper = self.agent_helper
        self.last_sim_location = None
        self.device = args.device
        self.first_obs = True
        self.valid_goals = 0
        self.total_episodes = 0
        self.args = args
        self.timestep = 0
        
    def reset(self):
        self.agent_helper.reset()
        self.agent_states.reset()
        self.last_sim_location = None
        self.first_obs = True
        self.step = 0
        self.timestep = 0
        self.total_episodes += 1

    def act(self, observations):
        self.timestep += 1
        
        # Always stop at episode end
        if self.timestep > self.args.timestep_limit:
            return {'action': 0}
        
        # Preprocess observations
        goal = observations['objectgoal'][0]
        info = self.get_info(observations)
        if self.args.use_gt_seg:
            info['goalseg'] = observations['goalseg']
            
        info['goal_name'] = hm3d_names[goal]
        goal = hm3d_to_coco[goal]
            
        self.agent_helper.set_goal_cat(goal)
        original_info = np.zeros((3))
        original_info[:2] = observations['gps']
        original_info[2] = observations['compass']
        # pdb.set_trace()

        obs, info = self.agent_helper.preprocess_inputs(observations['rgb'], observations['depth'], info)
        info['goal_cat_id'] = goal
        
        obs = obs[np.newaxis, :, :, :]
        obs = torch.from_numpy(obs).float().to(self.device)
        if self.first_obs:
            if(self.mapping_strategy == 'neural'):
                self.agent_states.init_with_obs(obs, info)
            elif(self.mapping_strategy == 'traditional'):
                self.agent_states.init_with_obs(obs,info,original_info)
            self.first_obs = False

        # Update state and plan action
        if(self.mapping_strategy == 'neural'):
            planner_inputs = self.agent_states.update_state(obs, info)
        elif(self.mapping_strategy == 'traditional'):
            planner_inputs = self.agent_states.update_state(obs,info,original_info)
        action = self.agent_helper.plan_act(planner_inputs)
        
        return action

    def get_info(self, obs):
        """Initialize additional info with relative pose change."""
        info = {}
        dx, dy, do = self.get_pose_change(obs)
        info['sensor_pose'] = [dx, dy, do]
        return info

    def get_sim_location(self,obs):
        """Returns x, y, o pose of the agent in the Habitat simulator."""
        x = obs['gps'][0]
        y = -obs['gps'][1]
        o = obs['compass']
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self,obs):
        curr_sim_pose = self.get_sim_location(obs)
        if self.last_sim_location is not None:
            dx, dy, do = pu.get_rel_pose_change(
                curr_sim_pose, self.last_sim_location)
            dx,dy,do = dx[0],dy[0],do[0]
        else:
            dx, dy, do = 0,0,0
        self.last_sim_location = curr_sim_pose
        return dx, dy, do





