#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=snt-exp09-robust-perc
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --output=/home/valperga/supernode-tokenizer/sbatch/das/logs/exp09_rlbench18_robustness_perceiver-%j.out
#SBATCH --error=/home/valperga/supernode-tokenizer/sbatch/das/logs/exp09_rlbench18_robustness_perceiver-%j.err

set -eu
exec sh /home/valperga/supernode-tokenizer/sbatch/das/run_experiment.sh exp09_rlbench18_robustness_perceiver.sh "$@"
