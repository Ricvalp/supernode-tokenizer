#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=snt-exp04-supernode-match
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --output=/home/valperga/supernode-tokenizer/sbatch/das/exp04_rlbench18_supernode_matched_chunk-%j.out
#SBATCH --error=/home/valperga/supernode-tokenizer/sbatch/das/exp04_rlbench18_supernode_matched_chunk-%j.err

set -eu
exec sh /home/valperga/supernode-tokenizer/sbatch/das/run_experiment.sh exp04_rlbench18_supernode_matched_chunk.sh "$@"
