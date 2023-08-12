#!/usr/bin/env bash

python nav/collect.py -v 0 --dump_location ./data/tmp --exp_name debug --start_ep 0 --end_ep 10 --evaluation $AGENT_EVALUATION_TYPE $@ 
wait
