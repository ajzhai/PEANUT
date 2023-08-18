import argparse
import os
import random
import habitat
import torch
from arguments import get_args
import numpy as np

from agent.peanut_agent import PEANUT_Agent

from habitat.core.logging import logger

def main():

    args_2 = get_args()
    args_2.sem_gpu_id = 0
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    nav_agent = PEANUT_Agent(args=args_2,task_config=config)
    if args_2.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
    else:
        challenge = habitat.Challenge(eval_remote=True)
    challenge.submit(nav_agent)
    
    # metrics = challenge.evaluate(nav_agent, num_episodes=200)
    # for k, v in metrics.items():
    #     logger.info("{}: {}".format(k, v))

if __name__ == "__main__":
    main()
