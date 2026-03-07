#!/bin/bash
#SBATCH --job-name=remy-cca-neural
#SBATCH --account=iris
#SBATCH --partition=sc-freecpu
#SBATCH --time=300:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/iris/u/armaana/jobs/logs/remy_%j.out
#SBATCH --error=/iris/u/armaana/jobs/logs/remy_%j.err



conda activate remy-rl
cd /iris/u/armaana/remy-rl
./src/rattrainer --cf=link-1x.cfg --of=checkpoints-$SLURM_JOB_ID/brain --save-every=1 --num-config-evals=24 --replay-buffer-size=3000000 --utd-ratio=5 --batch-size=2400000 --accumulation-steps=64 --hidden-size=64 --value-loss-coeff=0.1


