#!/bin/bash
#SBATCH -J ncr
#SBATCH -p hfactest
#SBATCH -o ./out/%j.out
#SBATCH -e ./out/%j.err
#SBATCH --mail-type=END,FAIL                 # Request status by email 
#SBATCH --mail-user=cn356@cornell.edu        # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 20                                # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=12G                            # server memory requested (per node)
#SBATCH -t 2:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=ma                       # Request partition
#SBATCH --gres=gpu:1                         # Type/number of GPUs needed

python svd.py
