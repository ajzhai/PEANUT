#!/usr/bin/env bash

python nav/collect.py -v 0 --dump_location ./data/tmp --exp_name debug --start_ep 0 --end_ep -1 --mapping_strategy neural --evaluation $AGENT_EVALUATION_TYPE $@ --seg_type Mask-RCNN -fw 160 -fh 120
wait
