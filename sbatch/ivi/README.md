# IvI Sbatch Launchers

These launchers default to:
- partition `cees6000`
- account `ceesusers`
- `1` GPU
- `40` CPUs per task
- `60G` memory
- `2-00:00` wall time

Stdout and stderr are written under `sbatch/ivi/logs/` with filenames based on the experiment name and Slurm job id.

Source the IvI environment in interactive shells with:

```bash
source experiments/env_ivi.sh
```

The IvI environment script only sets repo/cache/output/checkpoint roots.
If IvI needs a specific Python environment, activate it before sourcing
`experiments/env_ivi.sh`, or export `SUPERNODE_TOKENIZER_CONDA_PREFIX`
before submission.

Submit training jobs with:

```bash
sbatch sbatch/ivi/exp01_rlbench18_perceiver_chunk.sbatch
sbatch sbatch/ivi/exp11_rlbench18_dp3_chunk.sbatch
```

Submit resume jobs with:

```bash
sbatch sbatch/ivi/resume/exp01_rlbench18_perceiver_chunk.sbatch
sbatch sbatch/ivi/resume/exp11_rlbench18_dp3_chunk.sbatch
```
