#!/usr/bin/env bash

export OMP_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export MKL_NUM_THREADS=5
export VECLIB_MAXIMUM_THREADS=5
export NUMEXPR_NUM_THREADS=5

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

python nav/collect.py -v 2 --dump_location ./data/tmp --exp_name debug --start_ep 167  --end_ep 334 --mapping_strategy mixed --evaluation $AGENT_EVALUATION_TYPE --seg_type Segformer --perf_log_name bayesian_nav_part2 --col_rad 3 --fusion_type 'Bayesian' 

wait
