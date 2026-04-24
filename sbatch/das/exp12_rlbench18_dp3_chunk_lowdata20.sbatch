#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=snt-exp12-dp3-low20
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --output=/home/valperga/supernode-tokenizer/sbatch/das/logs/exp12_rlbench18_dp3_chunk_lowdata20-%j.out
#SBATCH --error=/home/valperga/supernode-tokenizer/sbatch/das/logs/exp12_rlbench18_dp3_chunk_lowdata20-%j.err

set -eu
exec sh /home/valperga/supernode-tokenizer/sbatch/das/run_experiment.sh exp12_rlbench18_dp3_chunk_lowdata20.sh "$@"
