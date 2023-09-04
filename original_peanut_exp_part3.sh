#!/usr/bin/env bash

export OMP_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export MKL_NUM_THREADS=5
export VECLIB_MAXIMUM_THREADS=5
export NUMEXPR_NUM_THREADS=5

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet


python nav/collect.py -v 0 --dump_location ./data/tmp --exp_name debug --start_ep 334 --end_ep -1 --mapping_strategy neural --evaluation $AGENT_EVALUATION_TYPE --seg_type Mask-RCNN -fw 160 -fh 120 --perf_log_name original_part3
wait
