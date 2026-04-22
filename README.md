# supernode-tokenizer

`supernode-tokenizer` is a standalone standard imitation learning codepath for RLBench point-cloud policies.
It is intentionally separate from the ICIL code in the parent repository:

- no in-context support/query episodes
- no MAML / FOMAML / test-time training
- no ICIL-specific losses or evaluation logic
- class-conditioned multitask imitation learning with point clouds + proprio

## What Is Implemented

- deterministic RLBench cache builder copied locally into this directory
- deterministic per-variation train/val/test episode splits: `100 / 25 / 25`
- standard-IL dataset and collator
- task-id conditioning at two levels:
  - encoder-side identity-init FiLM/AdaLN
  - decoder-side appended task tokens
- Perceiver observation encoder baseline
- supernode observation encoder
- supernode no-message-passing ablation
- chunk-regression decoder head
- diffusion decoder head
- DDP-capable train entrypoints
- rollout evaluation and robustness evaluation entrypoints
- experiment shell scripts for the main paper table and ablations

## Important Difference From The Original Prompt

The original prompt specified several RLBench-18 task names that do not exist verbatim in this repository.
The agreed task mapping is:

- `slide_block_to_color_target -> slide_block_to_target`
- `sweep_to_dustpan_of_size -> sweep_to_dustpan`
- `place_wine_at_rack_location -> stack_wine`

The geometry-sensitive subset is reported using those resolved task names.

## Train

Chunk-regression baseline:

```bash
PYTHONPATH=supernode-tokenizer python supernode-tokenizer/scripts/train_chunk_policy.py \
  --config=supernode-tokenizer/configs/train_chunk_policy.py
```

Supernode chunk model:

```bash
PYTHONPATH=supernode-tokenizer python supernode-tokenizer/scripts/train_chunk_policy.py \
  --config=supernode-tokenizer/configs/train_chunk_policy.py \
  --config.model.encoder_name=supernode
```

Diffusion model:

```bash
PYTHONPATH=supernode-tokenizer python supernode-tokenizer/scripts/train_diffusion_policy.py \
  --config=supernode-tokenizer/configs/train_diffusion_policy.py
```

DDP:

```bash
PYTHONPATH=supernode-tokenizer torchrun --standalone --nproc_per_node=4 \
  supernode-tokenizer/scripts/train_chunk_policy.py \
  --config=supernode-tokenizer/configs/train_chunk_policy.py
```

## Evaluate

Policy rollout evaluation:

```bash
PYTHONPATH=supernode-tokenizer python supernode-tokenizer/scripts/eval_policy.py \
  --config=supernode-tokenizer/configs/eval_policy.py \
  --config.checkpoint_path=/path/to/checkpoint.pt
```

Robustness sweep:

```bash
PYTHONPATH=supernode-tokenizer python supernode-tokenizer/scripts/eval_robustness.py \
  --config=supernode-tokenizer/configs/eval_robustness.py \
  --config.checkpoint_path=/path/to/checkpoint.pt
```

## Smoke / Debug Configs

- `configs/train_chunk_smoke.py`
- `configs/train_chunk_overfit_one_task.py`
- `configs/train_diffusion_smoke.py`
- `configs/eval_policy_smoke.py`

## Notes

- RGB hooks exist but default to `False` everywhere.
- Low-data mode currently uses deterministic truncation per variation via `data.low_data_train_demos_per_variation`.
- Rollout eval cycles over all RLBench task variations when `eval.variation_ids` is left empty.
- All reused logic was copied locally into this directory so the codepath can be moved out of `icil-rlbench` later.
