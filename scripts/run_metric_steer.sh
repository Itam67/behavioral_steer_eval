#!/bin/bash

PYTHONPATH=/home/hpcpres1/rds/hpc-work/steer_wrk
python experiments/calc_steer.py
python src/pipeline.py