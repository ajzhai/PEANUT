#!/usr/bin/env bash

export OMP_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export MKL_NUM_THREADS=5
export VECLIB_MAXIMUM_THREADS=5
export NUMEXPR_NUM_THREADS=5

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

python nav/collect.py -v 2 --dump_location ./data/tmp \
    --exp_name debug\
    --mapping_strategy mixed\
    --evaluation $AGENT_EVALUATION_TYPE \
    --seg_type Segformer --col_rad 4 \
    --perf_log_name traditional_averaging_final_part1 --estimate_z 0 \
    --fusion_type Averaging --map_trad_detection_threshold 0.6

wait
