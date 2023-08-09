#!/usr/bin/env bash

python nav/collect.py -v 0 --dump_location ./data/tmp --exp_name debugnew --print_images 1 --switch_step 0 --start_ep 1 --end_ep 2 --evaluation $AGENT_EVALUATION_TYPE $@ 
wait
