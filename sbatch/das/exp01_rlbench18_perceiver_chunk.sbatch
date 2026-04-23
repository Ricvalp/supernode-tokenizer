#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=snt-exp01-perceiver-chunk
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --output=/home/valperga/supernode-tokenizer/sbatch/das/exp01_rlbench18_perceiver_chunk-%j.out
#SBATCH --error=/home/valperga/supernode-tokenizer/sbatch/das/exp01_rlbench18_perceiver_chunk-%j.err

set -eu
exec sh /home/valperga/supernode-tokenizer/sbatch/das/run_experiment.sh exp01_rlbench18_perceiver_chunk.sh "$@"
