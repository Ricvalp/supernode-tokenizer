#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=snt-exp10-robust-super
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --output=/home/valperga/supernode-tokenizer/sbatch/das/exp10_rlbench18_robustness_supernode-%j.out
#SBATCH --error=/home/valperga/supernode-tokenizer/sbatch/das/exp10_rlbench18_robustness_supernode-%j.err

set -eu
exec sh /home/valperga/supernode-tokenizer/sbatch/das/run_experiment.sh exp10_rlbench18_robustness_supernode.sh "$@"
