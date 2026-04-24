#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=snt-r-exp07-perc-diff
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --output=/home/valperga/supernode-tokenizer/sbatch/das/logs/exp07_rlbench18_perceiver_diffusion_resume-%j.out
#SBATCH --error=/home/valperga/supernode-tokenizer/sbatch/das/logs/exp07_rlbench18_perceiver_diffusion_resume-%j.err

set -eu
exec sh /home/valperga/supernode-tokenizer/sbatch/das/run_experiment.sh resume/exp07_rlbench18_perceiver_diffusion.sh "$@"
