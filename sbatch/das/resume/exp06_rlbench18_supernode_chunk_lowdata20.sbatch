#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=snt-r-exp06-super-low20
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --output=/home/valperga/supernode-tokenizer/sbatch/das/logs/exp06_rlbench18_supernode_chunk_lowdata20_resume-%j.out
#SBATCH --error=/home/valperga/supernode-tokenizer/sbatch/das/logs/exp06_rlbench18_supernode_chunk_lowdata20_resume-%j.err

set -eu
exec sh /home/valperga/supernode-tokenizer/sbatch/das/run_experiment.sh resume/exp06_rlbench18_supernode_chunk_lowdata20.sh "$@"
