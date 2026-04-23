# DAS Sbatch Launchers

These launchers default to the `defq` partition.
Stdout and stderr are written under `sbatch/das/logs/` with filenames based on the experiment name and Slurm job id.

Source the DAS environment directly in interactive shells with:

```bash
source experiments/env_das.sh
```

Submit training jobs with:

```bash
sbatch sbatch/das/exp01_rlbench18_perceiver_chunk.sbatch
sbatch sbatch/das/exp02_rlbench18_supernode_chunk.sbatch
```

The robustness jobs require a checkpoint path argument:

```bash
sbatch sbatch/das/exp09_rlbench18_robustness_perceiver.sbatch /path/to/checkpoint.pt
sbatch sbatch/das/exp10_rlbench18_robustness_supernode.sbatch /path/to/checkpoint.pt
```
