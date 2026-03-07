#!/bin/bash
#SBATCH --job-name=remy-eval
#SBATCH --account=iris
#SBATCH --partition=sc-freecpu
#SBATCH --time=300:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/iris/u/armaana/jobs/logs/remy_%j.out
#SBATCH --error=/iris/u/armaana/jobs/logs/remy_%j.err




. /iris/u/armaana/remy-rl/.venv/bin/activate
cd /iris/u/armaana/remy
python ./scripts/plot.py 20x-2src/cca.19 --link-ppt 0.237 9.49 --num-points 9 --delay 150 --nsenders 2  --mean-on 5000 --mean-off 5000 --results-dir plot-20x-2src-1e6 --no-console-output-files



