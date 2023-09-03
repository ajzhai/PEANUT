#!/usr/bin/env bash
export OMP_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export MKL_NUM_THREADS=5
export VECLIB_MAXIMUM_THREADS=5
export NUMEXPR_NUM_THREADS=5


python nav/collect.py -v 0 --dump_location ./data/tmp --exp_name debug --start_ep 334 --end_ep -1 --mapping_strategy traditional --evaluation $AGENT_EVALUATION_TYPE --seg_type Segformer 
wait
