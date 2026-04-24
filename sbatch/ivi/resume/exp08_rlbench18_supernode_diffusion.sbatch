#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=snt-ivi-r-exp08-diff
#SBATCH --partition=cees6000
#SBATCH --account=ceesusers
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=40
#SBATCH --time=2-00:00
#SBATCH --output=/home/rvalper/supernode-tokenizer/sbatch/ivi/logs/exp08_rlbench18_supernode_diffusion_resume-%j.out
#SBATCH --error=/home/rvalper/supernode-tokenizer/sbatch/ivi/logs/exp08_rlbench18_supernode_diffusion_resume-%j.err
#SBATCH --export=ALL

set -eu
exec sh /home/rvalper/supernode-tokenizer/sbatch/ivi/run_experiment.sh resume/exp08_rlbench18_supernode_diffusion.sh "$@"
