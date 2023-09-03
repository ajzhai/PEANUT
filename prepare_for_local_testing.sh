#!bin/bash
export AGENT_EVALUATION_TYPE=local
export TRACK_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml
export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH
export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE
conda init
source ~/.bashrc
conda activate habitat
pip install open3d 
pip install klampt
pip install transformers