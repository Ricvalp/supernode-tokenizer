#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=snt-ivi-exp02-super-chunk
#SBATCH --partition=cees6000
#SBATCH --account=ceesusers
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=40
#SBATCH --time=2-00:00
#SBATCH --output=/home/rvalper/supernode-tokenizer/sbatch/ivi/logs/exp02_rlbench18_supernode_chunk-%j.out
#SBATCH --error=/home/rvalper/supernode-tokenizer/sbatch/ivi/logs/exp02_rlbench18_supernode_chunk-%j.err
#SBATCH --export=ALL

set -eu
exec sh /home/rvalper/supernode-tokenizer/sbatch/ivi/run_experiment.sh exp02_rlbench18_supernode_chunk.sh "$@"
