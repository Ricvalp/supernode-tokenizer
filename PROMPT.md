# PROMPT.md

You are working inside the standalone `supernode-tokenizer` repository.

This repository is a standard imitation learning codebase for RLBench point-cloud policies.
It was originally created inside `icil-rlbench`, but it must now be treated as an independent repo and codepath.
The intent is to use it for a NeurIPS-style paper on point-cloud tokenizers for robot imitation learning.

Everything in this prompt should be treated as the current source of truth for the repo.
If the code and this file diverge, update this file together with the code.

## 1. Core Goal

The paper story is:

> A locality-preserving supernode point-cloud tokenizer improves robot imitation learning, especially on geometry-sensitive manipulation tasks, under matched downstream policy heads and comparable compute.

This repo is focused on **controlled tokenizer comparisons**.
The main comparison principle is:

- same task conditioning
- same downstream policy head
- same action horizon
- same training budget
- same preprocessing
- same RLBench benchmark/eval protocol
- only the tokenizer/encoder should change in the main comparison

## 2. Hard Non-Goals

This repo is **not** for:

- ICIL / in-context imitation learning
- support/query episode conditioning
- MAML / FOMAML / test-time training
- language-conditioned policies in v1
- PerAct / DP3 / Act3D full baseline reimplementations
- generic policy-architecture exploration outside the tokenizer story

Do not reintroduce ICIL logic here.
Do not add dependencies on `icil.*` modules.
This repo should remain portable as a self-contained project.

## 3. Standalone Boundary

Treat this directory as a separate repository.

Practical rule:
- do not import new code from the parent `icil-rlbench` codepath
- if you need reusable logic from the old repo, copy it locally and adapt it here
- keep local copies correct and readable

Several modules were already copied locally for this reason:
- raw RLBench generation/export utilities
- cache/store code
- mask utilities
- attention / perceiver / embeddings
- supernode tokenizer primitives

## 4. Benchmark Definition

Primary benchmark: RLBench-18 multitask imitation learning.

Resolved task set used in this repo:
- `open_drawer`
- `slide_block_to_target`
- `sweep_to_dustpan`
- `meat_off_grill`
- `turn_tap`
- `put_item_in_drawer`
- `close_jar`
- `reach_and_drag`
- `stack_blocks`
- `light_bulb_in`
- `put_money_in_safe`
- `stack_wine`
- `put_groceries_in_cupboard`
- `place_shape_in_shape_sorter`
- `push_buttons`
- `insert_onto_square_peg`
- `stack_cups`
- `place_cups`

This differs from the original design prompt because some exact RLBench task names did not exist in the local RLBench package.
The agreed mapping is:
- `slide_block_to_color_target -> slide_block_to_target`
- `sweep_to_dustpan_of_size -> sweep_to_dustpan`
- `place_wine_at_rack_location -> stack_wine`

The mapping lives in [splits.py](./supernode_tokenizer/data/splits.py).

## 5. Geometry-Sensitive Subset

The subset that should be reported separately is:
- `open_drawer`
- `put_item_in_drawer`
- `close_jar`
- `turn_tap`
- `light_bulb_in`
- `stack_wine`
- `place_shape_in_shape_sorter`
- `insert_onto_square_peg`

This is also defined in [splits.py](./supernode_tokenizer/data/splits.py).

## 6. Data Format and Splits

The dataset is built from cached RLBench variation HDF5 files.
The cache format was copied from the old repo and remains variation-based.

Each variation file stores:
- `episode_ids`
- `episodes/<episode_id>/xyz`
- `episodes/<episode_id>/valid`
- optionally `rgb`
- optionally `mask_id`
- `state`
- `action`

Current deterministic split policy:
- train: first 100 episodes by sorted episode id
- val: next 25 episodes by sorted episode id
- test: next 25 episodes by sorted episode id

This is implemented in [splits.py](./supernode_tokenizer/data/splits.py) and used by [rlbench_standard_il_dataset.py](./supernode_tokenizer/data/rlbench_standard_il_dataset.py).

Low-data regime:
- deterministic truncation of the train split per variation
- controlled by `data.low_data_train_demos_per_variation`
- `-1` means full train split
- `20` gives the low-data regime used in the paper plan

## 6.1 Dataset Generation And Caching Status

Current status of this repo:
- **Raw RLBench dataset generation/export code is present locally inside `supernode-tokenizer`.**
- **Caching code is also present locally inside `supernode-tokenizer`.**

This repo supports the full two-stage dataset workflow:
1. generate a raw RLBench export with merged point clouds, low-dimensional observations, and segmentation label maps
2. convert that raw export into the standard-IL per-variation HDF5 cache used for training and evaluation

The local generation entrypoints are:
- [generate_rlbench_dataset.py](./scripts/generate_rlbench_dataset.py)
- [generate_rlbench_raw_dataset.py](./supernode_tokenizer/data/generate_rlbench_raw_dataset.py)

The local caching entrypoints are:
- [build_rlbench_cache.py](./scripts/build_rlbench_cache.py)
- [build_dense_cache_per_variation.py](./supernode_tokenizer/data/build_dense_cache_per_variation.py)

## 6.2 How To Generate The Raw RLBench Export In This Repo

The local raw-generation script is:
- [generate_rlbench_dataset.py](./scripts/generate_rlbench_dataset.py)

Recommended command for the repo RLBench-18 benchmark task set:

```bash
PYTHONUNBUFFERED=1 \
COPPELIASIM_ROOT="$HOME/CoppeliaSim" \
LD_LIBRARY_PATH="$HOME/CoppeliaSim:${LD_LIBRARY_PATH:-}" \
QT_QPA_PLATFORM_PLUGIN_PATH="$HOME/CoppeliaSim" \
QT_QPA_PLATFORM=xcb \
DISPLAY=:99 \
PYTHONPATH=supernode-tokenizer \
python -u supernode-tokenizer/scripts/generate_rlbench_dataset.py \
  --raw-root /path/to/raw_rlbench_export \
  --episodes-per-variation 150 \
  --variations 1 \
  --image-size 128 128 \
  --renderer opengl \
  --processes 4
```

Important defaults:
- if `--tasks` is omitted, the script uses the repo RLBench-18 task set from [splits.py](./supernode_tokenizer/data/splits.py)
- if `--all-tasks` is passed, it generates all locally available RLBench tasks instead
- default `--episodes-per-variation` is `150`
- default `--variations` is `1`

Why `150` episodes per variation:
- the deterministic split policy is `100 / 25 / 25`
- so `150` episodes per variation is the minimum that exactly supports the default train/val/test split

Task-subset example:

```bash
PYTHONUNBUFFERED=1 \
COPPELIASIM_ROOT="$HOME/CoppeliaSim" \
LD_LIBRARY_PATH="$HOME/CoppeliaSim:${LD_LIBRARY_PATH:-}" \
QT_QPA_PLATFORM_PLUGIN_PATH="$HOME/CoppeliaSim" \
QT_QPA_PLATFORM=xcb \
DISPLAY=:99 \
PYTHONPATH=supernode-tokenizer \
python -u supernode-tokenizer/scripts/generate_rlbench_dataset.py \
  --raw-root /path/to/raw_rlbench_export \
  --episodes-per-variation 150 \
  --variations 1 \
  --image-size 128 128 \
  --renderer opengl \
  --processes 4 \
  --tasks open_drawer slide_block_to_color_target place_wine_at_rack_location
```

Notes:
- task aliases passed to `--tasks` are resolved through [splits.py](./supernode_tokenizer/data/splits.py)
- so prompt aliases like `slide_block_to_color_target` are mapped automatically to the actual local RLBench task name `slide_block_to_target`
- the raw generator writes RGB, depth, mask images, merged point clouds, `low_dim_obs.pkl`, variation descriptions, and `mask_to_label.json`
- the raw generator depends on RLBench, PyRep, and a working CoppeliaSim install
- the raw generator shows a global `tqdm` progress bar named `raw-generate`
- the progress bar advances when an episode slot is resolved, i.e. when an episode is saved or when remaining slots in an aborted variation are marked skipped
- when `--variations >= 0`, the bar total is a safe estimate `len(tasks) * variations * episodes_per_variation`
- when `--variations = -1`, the generator cannot know the exact total without launching RLBench/CoppeliaSim in the parent process, so the progress bar should be treated as open-ended if this mode is ever enabled
- this is intentional: the parent process must not launch CoppeliaSim just to count totals, because that can trigger Qt/PyRep deadlocks before worker startup

## 6.3 Expected Raw Dataset Layout For Caching

The local cache builder expects a raw RLBench directory with this structure:

```text
<raw_root>/
  <task_name>/
    variation0/
      mask_to_label.json
      variation_descriptions.pkl
      episodes/
        episode0/
          low_dim_obs.pkl
          merged_point_cloud/
            0.npz
            1.npz
            ...
          left_shoulder_rgb/
          left_shoulder_depth/
          left_shoulder_mask/
          right_shoulder_rgb/
          right_shoulder_depth/
          right_shoulder_mask/
          overhead_rgb/
          overhead_depth/
          overhead_mask/
          wrist_rgb/
          wrist_depth/
          wrist_mask/
          front_rgb/
          front_depth/
          front_mask/
        episode1/
          ...
    variation1/
      ...
```

Each `merged_point_cloud/<t>.npz` contains:
- `points`: `[M, 3]` float32
- `colors`: `[M, 3]` uint8
- `masks`: `[M]` int32

Each episode also contains:
- `low_dim_obs.pkl`

Each variation directory also contains:
- `mask_to_label.json`
- `variation_descriptions.pkl`

This is exactly what the local cache builder reads in:
- [cache_variation_h5.py](./supernode_tokenizer/data/cache_variation_h5.py)
- [build_dense_cache_per_variation.py](./supernode_tokenizer/data/build_dense_cache_per_variation.py)

## 6.4 How To Build The Cache In This Repo

The local caching script is:
- [build_rlbench_cache.py](./scripts/build_rlbench_cache.py)

Example command:

```bash
PYTHONPATH=supernode-tokenizer python supernode-tokenizer/scripts/build_rlbench_cache.py \
  --raw-root /path/to/raw_rlbench_export \
  --cache-root /path/to/cache_root \
  --num-points 4096 \
  --num-workers 16
```

Optional task subset:

```bash
PYTHONPATH=supernode-tokenizer python supernode-tokenizer/scripts/build_rlbench_cache.py \
  --raw-root /path/to/raw_rlbench_export \
  --cache-root /path/to/cache_root \
  --tasks open_drawer,slide_block_to_color_target,place_wine_at_rack_location \
  --num-points 4096 \
  --num-workers 16
```

The cache builder resolves aliases through [splits.py](./supernode_tokenizer/data/splits.py), applies semantic ignore-mask filtering, workspace cropping, and fixed-size point sampling, then writes per-variation HDF5 files.

## 6.5 Cache Semantics

The local cache builder performs:
- semantic ignore-mask filtering using `mask_to_label.json`
- conservative background filtering
- workspace crop
- fixed-size point subsampling or resampling to `N` points
- storage of `xyz`, `valid`, optional `rgb`, optional `mask_id`, `state`, and `action`

The cached format is per-variation HDF5, not per-episode files.

## 6.6 End-To-End Dataset Workflow

The intended dataset workflow in this repo is:
1. generate the raw RLBench export with [generate_rlbench_dataset.py](./scripts/generate_rlbench_dataset.py)
2. build the cache with [build_rlbench_cache.py](./scripts/build_rlbench_cache.py)
3. train from the cached HDF5 dataset using the local training scripts

Practical rule:
- if you want the default deterministic `100 / 25 / 25` split, generate at least `150` episodes per variation
- if you generate fewer than `150`, the dataset constructor will raise when it cannot satisfy the deterministic split requirement

## 7. Input/Output Convention

This repo is standard IL with:
- point cloud observations
- proprio state
- task-id conditioning
- action chunk prediction

Default settings:
- `T_obs = 2`
- `stride = 2`
- `H = 16`
- `num_points = 4096`
- `RGB default = off`
- `mask_id default = on`

Collated training batch keys are:
- `task_id`: `[B]`
- `obs_xyz`: `[B, T_obs, N, 3]`
- `obs_state`: `[B, T_obs, S]`
- `obs_valid`: `[B, T_obs, N]`
- optional `obs_rgb`: `[B, T_obs, N, 3]`
- optional `obs_mask_id`: `[B, T_obs, N]`
- `target_action`: `[B, H, A]`
- `meta`: list of metadata dicts

These come from:
- [rlbench_standard_il_dataset.py](./supernode_tokenizer/data/rlbench_standard_il_dataset.py)
- [collate_standard_il.py](./supernode_tokenizer/data/collate_standard_il.py)

## 8. Task Conditioning Design

Task conditioning is class-conditioned, not language-conditioned.
Current conditioning uses `nn.Embedding(num_tasks, d_model)`.

Two levels are used simultaneously.

### 8.1 Encoder-side conditioning

Implemented with identity-init FiLM/AdaLN style conditioning in local blocks.
Modules:
- [blocks.py](./supernode_tokenizer/models/common/blocks.py)
- `IdentityTaskFiLM`
- `IdentityTaskAdaLN`
- `TaskConditionedSelfAttentionBlock`
- `TaskConditionedCrossAttentionBlock`
- `TaskConditionedFramePerceiverTokenizer`

Identity-init behavior:
- modulation projection weights initialized to zero
- modulation starts as a no-op
- task conditioning can grow during training without changing the unconditioned network at init

### 8.2 Decoder-side conditioning

Implemented with:
- `task_emb: [B, d_model]`
- `task_tokens: [B, n_task_tokens, d_model]`

Task tokens are appended to observation memory tokens before decoding.
This is handled by [task_conditioner.py](./supernode_tokenizer/models/condition/task_conditioner.py) and used in both policy heads.

Default:
- `n_task_tokens = 4`

## 9. Encoder Families

All encoders follow the same interface:
- input: observation history only
- output: `ObservationEncoderOutput(tokens, token_mask, debug)`

Defined in [base.py](./supernode_tokenizer/models/encoders/base.py).

### 9.1 Perceiver baseline

File:
- [observation_encoder_perceiver.py](./supernode_tokenizer/models/encoders/observation_encoder_perceiver.py)

Structure:
- per point feature projection from xyz
- optional rgb projection with learnable alpha
- optional gripper-relative point features
- task-conditioned frame Perceiver tokenizer
- append projected state token
- add frame-time embedding
- flatten across `T_obs`
- post-tokenizer task-conditioned self-attention refinement

Default key values:
- `m_frame_tokens = 128`
- state token appended per frame
- effective tokens per frame = `128 + 1 = 129`
- `frame_tokenizer_layers = 2`
- `post_self_attn_layers = 2`

### 9.2 Supernode main model

File:
- [observation_encoder_supernode.py](./supernode_tokenizer/models/encoders/observation_encoder_supernode.py)

Structure:
- point feature projection from xyz
- optional rgb projection with learnable alpha
- optional gripper-relative point features
- optional mask embedding
- quota-based supernode sampling
- KNN neighborhood construction
- point-to-supernode message passing with task FiLM
- supernode self-attention refinement with task conditioning
- optional Perceiver pooling from supernodes to fixed frame token count
- append projected state token
- add frame-time embedding
- flatten across `T_obs`
- post-tokenizer task-conditioned self-attention refinement

Default key values:
- `num_supernodes = 192`
- `frame_tokens_out = 128`
- `neighbors_per_supernode = 32`
- `supernode_refine_layers = 2`
- `compress_supernodes = True`
- `supernode_pool_layers = 1`
- `post_self_attn_layers = 2`
- `supernode_sampling_mode = fast_random`
- `use_mask_id = True`
- `use_mask_embedding = False`
- `use_mask_instance_quota = True`

Interpretation:
- the main supernode model uses more raw local structure than the Perceiver baseline before compressing to a matched frame-token count

### 9.3 Supernode no-message-passing ablation

Files:
- [observation_encoder_supernode_nomsg.py](./supernode_tokenizer/models/encoders/observation_encoder_supernode_nomsg.py)
- [observation_encoder_supernode.py](./supernode_tokenizer/models/encoders/observation_encoder_supernode.py)

This ablation:
- keeps the same supernode sampling
- keeps the same neighborhood construction
- removes point-to-supernode message passing
- replaces it with simple neighborhood pooling into sampled supernodes

This is controlled by `use_message_passing = False`.

### 9.4 Supernode matched-budget ablation

There is no separate encoder class.
The matched-budget ablation is represented by a config/launch variant:
- same downstream token count
- reduced raw supernode count to stay closer to the Perceiver baseline

Current experiment wrapper sets:
- `num_supernodes = 128`

See:
- [exp04_rlbench18_supernode_matched_chunk.sh](./experiments/exp04_rlbench18_supernode_matched_chunk.sh)

## 10. Shared Policy Heads

### 10.1 Chunk-regression head

File:
- [chunk_decoder_policy.py](./supernode_tokenizer/models/policies/chunk_decoder_policy.py)

This is the primary paper head.

Structure:
- learned action query tokens of length `H`
- learned action-slot embeddings
- stack of task-conditioned transformer decoder blocks
- self-attention over action queries
- cross-attention to `[obs_memory_tokens ; task_tokens]`
- parallel prediction of all `H` action slots
- linear output projection to action dimension

Default:
- `d_model = 512`
- `n_heads = 8`
- `n_layers = 8`
- `mlp_mult = 4`
- `dropout = 0.0`
- `loss_type = l1`

Loss outputs:
- `loss`
- `l1`
- `mse`
- `pred_action`

### 10.2 Diffusion head

File:
- [diffusion_policy.py](./supernode_tokenizer/models/policies/diffusion_policy.py)

This is the secondary paper head.

Structure:
- same encoder memory contract as the chunk head
- DDIM scheduler via `diffusers`
- task-conditioned denoiser blocks
- sinusoidal diffusion time embedding + MLP
- parallel denoising over action chunk tokens

Default:
- `d_model = 512`
- `n_heads = 8`
- `denoiser_layers = 10`
- `denoiser_mlp_mult = 4`
- `num_train_timesteps = 1000`
- `prediction_type = v_prediction`
- `num_inference_steps = 50`

## 11. Builder System

The main model entrypoint is [builders.py](./supernode_tokenizer/models/builders.py).

Key dataclass:
- `ModelConfig`

Main builder functions:
- `build_encoder`
- `build_policy`
- `validate_model_config`

Supported encoder names:
- `perceiver`
- `supernode`
- `supernode_nomsg`

Supported policy heads:
- `chunk`
- `diffusion`

Validation rule:
- encoder `d_model`
- task conditioner `d_model`
- policy-head `d_model`
must all match

## 12. Training Stack

Chunk trainer:
- [train_chunk.py](./supernode_tokenizer/trainers/train_chunk.py)

Diffusion trainer:
- [train_diffusion.py](./supernode_tokenizer/trainers/train_diffusion.py)

Common behavior:
- DDP through `torch.distributed`
- optional AMP
- AdamW optimizer
- gradient clipping
- checkpoint save/load
- wandb hooks
- periodic validation on the deterministic val split

Utility modules:
- [ddp_utils.py](./supernode_tokenizer/utils/ddp_utils.py)
- [checkpointing.py](./supernode_tokenizer/utils/checkpointing.py)
- [wandb_utils.py](./supernode_tokenizer/utils/wandb_utils.py)
- [metrics.py](./supernode_tokenizer/utils/metrics.py)

Default optimizer values:
- `lr = 1e-4`
- `betas = (0.9, 0.95)`
- `weight_decay = 1e-4`
- `grad_clip_norm = 1.0`

Default chunk training values:
- `num_steps = 300000`
- `batch_size = 8`
- `grad_accum_steps = 1`
- `eval_every = 5000`
- `ckpt_every = 10000`

Default diffusion training values:
- `num_steps = 400000`
- `batch_size = 8`
- `grad_accum_steps = 1`
- `eval_every = 5000`
- `ckpt_every = 10000`

Current validation behavior:
- validation is loss-based only
- rollout eval is a separate script

## 13. Evaluation Stack

Main eval file:
- [eval_rlbench.py](./supernode_tokenizer/eval/eval_rlbench.py)

Entry scripts:
- [eval_policy.py](./scripts/eval_policy.py)
- [eval_robustness.py](./scripts/eval_robustness.py)

### 13.1 Rollout evaluation

Current rollout evaluation:
- loads a checkpoint
- reconstructs model from saved config
- loads `task_name_to_id` from checkpoint
- builds a live RLBench environment
- converts live observations to the same point-cloud representation used in training
- predicts action chunks and executes `execute_actions_per_plan` actions per plan
- writes per-task and summary JSON
- optionally writes videos

Default eval settings:
- `episodes_per_task = 25`
- `max_env_steps = 200`
- `execute_actions_per_plan = 8`
- `num_points = 4096`
- quaternion normalization enabled
- gripper discretization enabled

Important current behavior:
- if `eval.variation_ids` is empty, rollout eval cycles over all RLBench variations for that task
- if provided, it cycles through the listed variation ids

### 13.2 Robustness evaluation

Current robustness sweep varies at least:
- point count: `[4096, 2048, 1024, 512]`
- random point dropout: `[0.0, 0.25, 0.5]`

The implementation reuses the live eval pipeline and changes observation subsampling/dropout at eval time.

## 14. Config Files

Main configs:
- [train_chunk_policy.py](./configs/train_chunk_policy.py)
- [train_diffusion_policy.py](./configs/train_diffusion_policy.py)
- [eval_policy.py](./configs/eval_policy.py)
- [eval_robustness.py](./configs/eval_robustness.py)

Smoke/debug configs:
- [train_chunk_smoke.py](./configs/train_chunk_smoke.py)
- [train_chunk_overfit_one_task.py](./configs/train_chunk_overfit_one_task.py)
- [train_diffusion_smoke.py](./configs/train_diffusion_smoke.py)
- [eval_policy_smoke.py](./configs/eval_policy_smoke.py)

Config package note:
- `configs/` is now an importable Python package
- smoke configs use package-style imports from `configs.*`

## 15. Experiment Scripts

Thin wrappers live in [experiments](./experiments):
- `exp01_rlbench18_perceiver_chunk.sh`
- `exp02_rlbench18_supernode_chunk.sh`
- `exp03_rlbench18_supernode_nomsg_chunk.sh`
- `exp04_rlbench18_supernode_matched_chunk.sh`
- `exp05_rlbench18_perceiver_chunk_lowdata20.sh`
- `exp06_rlbench18_supernode_chunk_lowdata20.sh`
- `exp07_rlbench18_perceiver_diffusion.sh`
- `exp08_rlbench18_supernode_diffusion.sh`
- `exp09_rlbench18_robustness_perceiver.sh`
- `exp10_rlbench18_robustness_supernode.sh`

Important implementation detail:
- these scripts set explicit `output.run_name` overrides so runs are labeled correctly even when the encoder is overridden from the command line

## 16. Environment Variables

The main configs use these environment variables when available:
- `SUPERNODE_TOKENIZER_CACHE_ROOT`
- `SUPERNODE_TOKENIZER_OUTPUT_ROOT`
- `SUPERNODE_TOKENIZER_CHECKPOINT_ROOT`
- `SUPERNODE_TOKENIZER_EVAL_ROOT`
- `WANDB_PROJECT`
- `WANDB_ENTITY`
- `WANDB_MODE`

There is also a fallback to `ICIL_CACHE_ROOT` for cache location if needed.

## 17. Current File Map

Top-level structure:
- `supernode_tokenizer/data/`
- `supernode_tokenizer/models/`
- `supernode_tokenizer/trainers/`
- `supernode_tokenizer/eval/`
- `supernode_tokenizer/utils/`
- `configs/`
- `scripts/`
- `experiments/`
- `README.md`
- `setup.py`
- `PROMPT.md`

Important source files:
- dataset/cache:
  - [data/rlbench_standard_il_dataset.py](./supernode_tokenizer/data/rlbench_standard_il_dataset.py)
  - [data/splits.py](./supernode_tokenizer/data/splits.py)
- task conditioning:
  - [models/condition/task_conditioner.py](./supernode_tokenizer/models/condition/task_conditioner.py)
- encoders:
  - [models/encoders/observation_encoder_perceiver.py](./supernode_tokenizer/models/encoders/observation_encoder_perceiver.py)
  - [models/encoders/observation_encoder_supernode.py](./supernode_tokenizer/models/encoders/observation_encoder_supernode.py)
  - [models/encoders/observation_encoder_supernode_nomsg.py](./supernode_tokenizer/models/encoders/observation_encoder_supernode_nomsg.py)
- policy heads:
  - [models/policies/chunk_decoder_policy.py](./supernode_tokenizer/models/policies/chunk_decoder_policy.py)
  - [models/policies/diffusion_policy.py](./supernode_tokenizer/models/policies/diffusion_policy.py)
- training:
  - [trainers/train_chunk.py](./supernode_tokenizer/trainers/train_chunk.py)
  - [trainers/train_diffusion.py](./supernode_tokenizer/trainers/train_diffusion.py)
- eval:
  - [eval/eval_rlbench.py](./supernode_tokenizer/eval/eval_rlbench.py)

## 18. Coding Rules for Future Work

When editing this repo, follow these principles.

### 18.1 Preserve the tokenizer-paper scope

Do not turn this into a broad robotics-policy playground.
If adding a new experiment, ask whether it isolates tokenizer effects or dilutes the paper.

### 18.2 Keep the comparison fair

For main-model comparisons:
- keep decoder fixed
- keep task conditioning fixed
- keep preprocessing fixed
- keep training budget fixed
- change only encoder/tokenizer-side choices

### 18.3 Keep it standalone

Do not add imports from the parent repo.
If you need a shared utility from the old codebase, copy it here and adapt it locally.

### 18.4 Keep config surface clean

Prefer a small number of explicit, readable knobs.
Do not explode the config space unless the paper needs it.

### 18.5 Be explicit about defaults

If you change a default that affects experiment fairness or benchmark numbers, update:
- config files
- README
- this `PROMPT.md`
- experiment scripts if needed

## 19. Known Current Limitations

Be aware of these current limitations before extending the repo.

- The code compiles and the model builders instantiate, but the repo has not yet been runtime-validated end-to-end on a full real training run in this state.
- Rollout eval is implemented, but should be smoke-tested before large cluster usage.
- Robustness eval currently covers point-count and random-dropout perturbations only.
- The matched-budget ablation is currently a config/launch variant, not a distinct module.
- There is no language-conditioning backend yet; only task-id embeddings are implemented.
- There is no separate paper-table export utility yet beyond saved summary JSON.

## 20. Recommended Next Validation Order

If you are continuing work on this repo, validate in this order:

1. Build the cache with [build_rlbench_cache.py](./scripts/build_rlbench_cache.py)
2. Run [train_chunk_smoke.py](./configs/train_chunk_smoke.py)
3. Run [train_chunk_overfit_one_task.py](./configs/train_chunk_overfit_one_task.py)
4. Run [eval_policy_smoke.py](./configs/eval_policy_smoke.py) on a smoke checkpoint
5. Run `exp01` and `exp02`
6. Only after that, trust larger ablations or diffusion runs

## 21. Short Operational Summary

If you need the shortest possible mental model of the repo, it is this:

- The repo trains class-conditioned RLBench imitation policies from point clouds and proprio.
- The main paper comparison is Perceiver tokenizer vs supernode tokenizer.
- All main models share the same downstream decoder family.
- Data splits are deterministic `100/25/25` per variation.
- Training is standard supervised imitation learning.
- Evaluation is rollout-based on RLBench plus robustness sweeps.
- The repo is intentionally independent from ICIL and should stay that way.
