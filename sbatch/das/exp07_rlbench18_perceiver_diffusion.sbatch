#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=snt-exp07-perceiver-diff
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --output=/home/valperga/supernode-tokenizer/sbatch/das/exp07_rlbench18_perceiver_diffusion-%j.out
#SBATCH --error=/home/valperga/supernode-tokenizer/sbatch/das/exp07_rlbench18_perceiver_diffusion-%j.err

set -eu
exec sh /home/valperga/supernode-tokenizer/sbatch/das/run_experiment.sh exp07_rlbench18_perceiver_diffusion.sh "$@"
