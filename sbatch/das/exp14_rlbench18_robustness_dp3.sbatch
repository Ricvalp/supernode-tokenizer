#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=snt-exp14-robust-dp3
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --output=/home/valperga/supernode-tokenizer/sbatch/das/logs/exp14_rlbench18_robustness_dp3-%j.out
#SBATCH --error=/home/valperga/supernode-tokenizer/sbatch/das/logs/exp14_rlbench18_robustness_dp3-%j.err

set -eu
exec sh /home/valperga/supernode-tokenizer/sbatch/das/run_experiment.sh exp14_rlbench18_robustness_dp3.sh "$@"
