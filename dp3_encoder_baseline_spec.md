# DP3 Encoder Baseline Spec for `supernode-tokenizer`

This document specifies the **DP3-style point-cloud encoder baseline** to include in the `supernode-tokenizer` experiments, and explains exactly how to attach it to the **existing chunk decoder** and **existing DiT diffusion decoder** already implemented in the repo.

The goal is to provide a **faithful, strong, external point-cloud baseline** that can be compared directly against:

- the current **Perceiver-based frame tokenizer**, and
- the new **supernode frame tokenizer**.

This is **standard imitation learning**, **not** in-context imitation learning:
- no support/query split,
- no test-time training,
- no MAML/FOMAML,
- no adaptation at inference.

The comparison should isolate the **point-cloud encoder / tokenizer** while reusing as much of the existing repo as possible.

---

## 1. Source of truth

Implement the DP3 encoder as faithfully as possible to:

- **3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations**  
  Yanjie Ze et al., 2024  
  arXiv:2403.03954  
  HTML: https://arxiv.org/html/2403.03954v4  
  PDF: https://arxiv.org/pdf/2403.03954

Key details from the paper / appendix:

1. **Point cloud preprocessing**
   - crop raw point cloud with a workspace-aligned bounding box;
   - downsample using **farthest point sampling (FPS)**;
   - use **512 or 1024 points**;
   - original DP3 uses **xyz only** in the main paper, but the encoder also supports `xyzrgb` in experiments.

2. **DP3 Encoder**
   - per-point **3-layer MLP**:
     - `Linear(channels, 64) -> LayerNorm(64) -> ReLU`
     - `Linear(64, 128) -> LayerNorm(128) -> ReLU`
     - `Linear(128, 256) -> LayerNorm(256) -> ReLU`
   - global **max-pooling** over points
   - projection head:
     - `Linear(256, 64) -> LayerNorm(64)`
   - output point-cloud feature dimension: **64**

3. **Robot state encoder**
   - `Linear(state_dim, 64) -> ReLU -> Linear(64, 64)`
   - concatenate point feature and state feature
   - final per-observation embedding dimension: **128**

DP3 in the paper then feeds this compact observation representation into a convolutional diffusion policy.  
In this repo, we will keep the **existing transformer decoders** and adapt the DP3 observation representation into **memory tokens** for decoder cross-attention.

---

## 2. Design principle for this repo

The point of the DP3 baseline is **not** to replicate the original DP3 action head exactly.  
The point is to provide a **faithful DP3-style point-cloud encoder** under the **same downstream decoder family** used by the other models.

Therefore:

- **Do not** reimplement the original DP3 1D U-Net decision backbone as the main comparison.
- **Do** keep the existing repo’s **chunk decoder** and **DiT diffusion decoder**.
- **Only change the encoder / tokenizer path**.

This isolates the contribution of the tokenizer / encoder.

---

## 3. Required preprocessing

### 3.1 Workspace crop
Implement a point-cloud crop stage before the DP3 encoder.

Use the same fixed crop bounds already used in the repo for all models, if such bounds exist.  
Otherwise, add a common workspace crop config and apply it identically to:

- Perceiver encoder
- supernode tokenizer
- DP3 encoder baseline

This is important for fairness.

### 3.2 Downsampling
Apply **farthest point sampling (FPS)** after cropping.

Recommended default for RLBench:
- `n_points = 1024`

Optional low-compute variant:
- `n_points = 512`

Use the **same input point count for all models** in the main experiments.

### 3.3 Color channels
DP3’s paper uses **xyz only** in the main setting for better appearance generalization, but explicitly notes the encoder also works with `xyzrgb`.

For this repo:

- if the main experimental setting for all models uses point clouds **without RGB**, set `channels=3`;
- if the main setting for all models uses **xyzrgb**, set `channels=6`.

**Important:** keep the **same input modality across all compared models**.

Do **not** give RGB to one encoder and not the others in the main table.

---

## 4. Exact DP3 encoder module

Implement a module called, for example:

```python
DP3ObservationEncoder
```

This module encodes one observation timestep (one frame + robot state) into a compact vector.

### 4.1 Inputs
For one timestep:
- `xyz`: `[B, N, 3]`
- `rgb`: `[B, N, 3]` or `None`
- `state`: `[B, state_dim]`

### 4.2 Point encoder
If using `xyz` only:
- point input shape = `[B, N, 3]`

If using `xyzrgb`:
- concatenate channels and use `[B, N, 6]`

Exact architecture:

```python
self.mlp = nn.Sequential(
    nn.Linear(channels, 64),
    nn.LayerNorm(64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.LayerNorm(128),
    nn.ReLU(),
    nn.Linear(128, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
)

self.projection = nn.Sequential(
    nn.Linear(256, 64),
    nn.LayerNorm(64),
)
```

Forward:
1. `x = self.mlp(points)` -> `[B, N, 256]`
2. `x = torch.max(x, dim=1)[0]` -> `[B, 256]`
3. `x = self.projection(x)` -> `[B, 64]`

### 4.3 State encoder
Exact architecture:

```python
self.state_mlp = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
)
```

Forward:
- `s = self.state_mlp(state)` -> `[B, 64]`

### 4.4 Final observation feature
Concatenate:

```python
obs_feat = torch.cat([pc_feat, state_feat], dim=-1)  # [B, 128]
```

No extra nonlinear layer after concatenation in the faithful baseline.

---

## 5. Multi-frame history handling (important)

The repo already uses short observation history (`T_obs`) for policies.  
DP3 in the paper works over observation sequences, but its encoder itself is **per timestep**.

For this repo, handle history as follows:

### 5.1 Encode each timestep independently
For `T_obs` observation frames:
- run the same `DP3ObservationEncoder` on each frame independently.

Input:
- `query_xyz`: `[B, T_obs, N, 3]`
- `query_rgb`: `[B, T_obs, N, 3]` or `None`
- `query_state`: `[B, T_obs, state_dim]`

Implementation:
- flatten batch/time:
  - `[B * T_obs, N, C]`
  - `[B * T_obs, state_dim]`
- encode each timestep
- reshape back to:
  - `obs_feat_seq: [B, T_obs, 128]`

### 5.2 Time embedding
Project the per-timestep 128-D feature to `d_model` and add a time embedding over `T_obs`.

Recommended module:

```python
self.obs_proj = nn.Sequential(
    nn.Linear(128, d_model),
    nn.LayerNorm(d_model),
)
```

Then:
- `obs_tokens = self.obs_proj(obs_feat_seq)` -> `[B, T_obs, d_model]`
- add a learned or sinusoidal time embedding over `T_obs`

Recommended default:
- use the same **query-time embedding** mechanism already used in the repo for query tokens, if possible.

### 5.3 Optional temporal refiner
To keep downstream interfaces more comparable across encoders, add **0 or 1 lightweight temporal self-attention blocks** on top of these `T_obs` tokens.

Recommended default:
- **0 blocks** for the most faithful DP3 baseline
- **1 block** only as an optional ablation if needed

Do **not** make this branch deep; otherwise the comparison stops being “DP3-style compact encoder”.

---

## 6. How to attach DP3 encoder to the existing decoders

This is the key integration point.

Your existing decoders expect **memory tokens** for cross-attention.  
The DP3 encoder produces a **compact per-timestep observation feature**.

The clean solution is:

- convert each timestep’s compact DP3 feature into **one memory token**
- sequence length seen by the decoder is therefore:
  - `T_obs` visual tokens
  - plus task tokens (see below)

This is the recommended **main** DP3 baseline.

### 6.1 Main memory interface
After Section 5:
- `obs_tokens`: `[B, T_obs, d_model]`

This becomes the decoder memory.

If task conditioning is enabled (recommended), append task tokens:
- see Section 8.

Final memory:
- `memory = concat([obs_tokens, task_tokens], dim=1)`  
- shape: `[B, T_obs + M_task, d_model]`

This memory is then consumed by:

- the **chunk decoder**
- the **diffusion DiT decoder**

using the exact same cross-attention mechanism already present in the repo.

### 6.2 Why one token per frame
This is the most faithful adaptation of DP3’s design:
- DP3 intentionally compresses each point cloud into a **compact global representation**
- using one memory token per observation frame preserves that property

Do **not** artificially inflate DP3 into dozens of tokens per frame in the main comparison.

### 6.3 Optional supplementary variant
If needed for a supplementary fairness check, add a second variant:
- project each 128-D per-frame feature to `M_dp3` tokens per frame (e.g. 4 tokens/frame) through a small learned MLP

But this should **not** be the primary baseline, because it is no longer the standard DP3 encoder design.

---

## 7. Chunk policy attachment

The chunk policy should be the **main policy head** for the paper because it is simpler and isolates the tokenizer better than diffusion.

### 7.1 Decoder input
Reuse the repo’s standard chunk decoder implementation if present.

If a chunk decoder is not already implemented, use the existing transformer decoder stack with the following interface:

- learned action query tokens of length `H`
- action-position embedding over the `H` slots
- cross-attend to `memory` from the encoder

Input to decoder:
- `action_queries`: `[B, H, d_model]`
- `memory`: `[B, T_obs + M_task, d_model]`

Output:
- `pred_action`: `[B, H, action_dim]`

Loss:
- primary: `L1`
- secondary ablation: `MSE`

### 7.2 Decoder architecture
The chunk decoder should be **shared across all encoders**:
- Perceiver encoder
- supernode tokenizer
- DP3 encoder

Only the encoder changes.

---

## 8. Diffusion policy attachment

Use the **existing DiT diffusion decoder** in the repo with minimal changes.

### 8.1 Encoder output
Use the same `memory` tokens from Section 6:
- `obs_tokens` from the DP3 encoder
- optional task tokens appended

### 8.2 DiT input
Exactly as in the existing repo:
- noisy action chunk `x_t`
- action-slot positional embeddings
- diffusion timestep conditioning
- cross-attention to `memory`

### 8.3 What changes
Only the encoder changes.
The diffusion decoder remains identical.

This is important to isolate the encoder/tokenizer.

---

## 9. Task conditioning (class-conditioned IL)

This repo is for **class-conditioned standard imitation learning**, not language-conditioned VLA and not in-context learning.

Therefore task conditioning should be handled uniformly across all encoders.

Recommended approach:

### 9.1 Task embedding
Use a learned embedding:
```python
task_emb = nn.Embedding(num_tasks, d_model)
```

### 9.2 Level-1 conditioning (encoder-side)
Inject task information into the encoder-side representation.

For DP3:
- after projecting per-frame observation features to `d_model`, add the task embedding:
  - `obs_tokens = obs_tokens + task_emb[:, None, :]`

This is the simplest version.

Optional stronger version:
- FiLM/AdaLN on the projected `obs_tokens`

### 9.3 Level-2 conditioning (decoder-side)
Project the task embedding into 1–4 decoder memory tokens:
```python
task_tokens = task_token_mlp(task_emb)  # [B, M_task * d_model]
task_tokens = task_tokens.view(B, M_task, d_model)
```

Recommended default:
- `M_task = 1`

Append to memory:
```python
memory = torch.cat([obs_tokens, task_tokens], dim=1)
```

### 9.4 Fairness rule
Use **exactly the same task-conditioning scheme for all encoders**.

Do not special-case DP3.

---

## 10. Suggested module structure in the repo

Implement the following modules.

### 10.1 `models/encoders/dp3_observation_encoder.py`
Contains:
- `DP3ObservationEncoder`
- per-timestep point+state encoder
- no temporal modeling inside this module

### 10.2 `models/encoders/dp3_sequence_encoder.py`
Contains:
- loops over `T_obs`
- applies `DP3ObservationEncoder`
- projects to `d_model`
- adds time embedding
- appends task tokens (or returns visual tokens + task tokens separately)

Output interface should match the rest of the repo’s encoder outputs as closely as possible.

Recommended output object fields:
- `query_tokens`: `[B, T_obs, d_model]` or final memory tokens
- `tokens`: `[B, T_obs + M_task, d_model]`
- masks if used

For standard IL, there is no support/query split in the ICIL sense, so the simplest is:
- return `tokens` only
- or return `query_tokens == tokens`

### 10.3 Config entry
Add a config option such as:
```yaml
model.encoder_name: dp3
```

Possible encoder names:
- `perceiver`
- `supernode`
- `dp3`

---

## 11. Exact defaults to use

These should be the defaults for the main DP3 baseline.

### Input / preprocessing
- point cloud crop: same as all methods
- FPS points: `1024`
- channels: same as other encoders in the experiment (`3` or `6`)

### DP3 encoder
- MLP dims: `64 -> 128 -> 256`
- projection dim: `64`
- state MLP output dim: `64`
- final per-frame obs feature: `128`

### Sequence adapter
- project `128 -> d_model`
- one token per frame
- learned or sinusoidal time embedding
- no temporal self-attn by default

### Task conditioning
- `M_task = 1`
- add task embedding to obs tokens
- append 1 task token to memory

### Decoder
- identical chunk decoder / DiT decoder across all methods

---

## 12. Experiment fairness rules

These must be enforced.

1. Same RLBench split, same demos, same seeds
2. Same point-cloud crop
3. Same FPS point count
4. Same RGB usage / no-RGB usage
5. Same task conditioning
6. Same decoder and action head for the main comparison
7. Same training budget / number of steps
8. Report parameter counts, throughput, and peak GPU memory

---

## 13. Main experiment usage

The DP3 encoder should be used in the following comparisons:

### Main table
- Perceiver encoder + chunk decoder
- DP3 encoder + chunk decoder
- supernode encoder + chunk decoder

### Secondary table
- Perceiver encoder + diffusion decoder
- DP3 encoder + diffusion decoder
- supernode encoder + diffusion decoder

### Low-data regime
Repeat chunk-decoder comparison with fewer demos/task.

### Robustness
Evaluate encoder robustness under:
- fewer input points
- stronger point dropout
- noise / jitter

---

## 14. Notes for Codex

### Reuse existing repo components aggressively
Codex must reuse:
- existing RLBench dataset / cache / DDP / logging
- existing action chunk decoder if present
- existing DiT decoder
- existing task-conditioning plumbing where possible
- existing config system and training script structure

The DP3 baseline should be implemented as a **drop-in encoder option**, not as a separate codebase.

### Keep repo structure similar to `icil-rlbench`
Even though this repo is not for ICIL, the module organization should remain close enough that:
- data code,
- logging,
- DDP launcher,
- checkpointing,
- and evaluation
are easy to reuse.

---

## 15. Minimal pseudocode

```python
class DP3ObservationEncoder(nn.Module):
    def __init__(self, point_channels: int, state_dim: int):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(point_channels, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.point_proj = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

    def forward(self, points, state):
        # points: [B, N, C]
        x = self.point_mlp(points)       # [B, N, 256]
        x = x.max(dim=1).values          # [B, 256]
        pc_feat = self.point_proj(x)     # [B, 64]
        st_feat = self.state_mlp(state)  # [B, 64]
        return torch.cat([pc_feat, st_feat], dim=-1)  # [B, 128]
```

```python
class DP3SequenceEncoder(nn.Module):
    def __init__(self, obs_encoder, d_model, num_tasks, n_task_tokens=1):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.obs_proj = nn.Sequential(
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
        )
        self.task_embed = nn.Embedding(num_tasks, d_model)
        self.task_token_proj = nn.Linear(d_model, n_task_tokens * d_model)
        # reuse repo time embedding if possible

    def forward(self, xyz, rgb, state, task_id):
        # xyz: [B, T_obs, N, 3], rgb optional, state: [B, T_obs, state_dim]
        B, T_obs = xyz.shape[:2]
        points = xyz if rgb is None else torch.cat([xyz, rgb], dim=-1)
        points = points.view(B * T_obs, points.shape[2], points.shape[3])
        st = state.view(B * T_obs, state.shape[-1])

        obs_feat = self.obs_encoder(points, st)       # [B*T_obs, 128]
        obs_feat = obs_feat.view(B, T_obs, 128)
        obs_tokens = self.obs_proj(obs_feat)          # [B, T_obs, d_model]

        t = self.task_embed(task_id)                  # [B, d_model]
        obs_tokens = obs_tokens + t[:, None, :]

        task_tokens = self.task_token_proj(t).view(B, -1, obs_tokens.shape[-1])
        memory = torch.cat([obs_tokens, task_tokens], dim=1)
        return memory
```

The output `memory` should then plug directly into:
- the chunk decoder cross-attention, or
- the DiT diffusion decoder cross-attention.

---

## 16. Final recommendation

The primary external baseline should be:

> **DP3-style compact point-cloud encoder + the same decoder used by the other models**

This gives a strong, credible baseline that is:
- faithful to a known SOTA 3D manipulation policy encoder,
- easy to implement,
- and directly comparable to the supernode tokenizer.

If the supernode tokenizer beats this baseline under matched downstream heads and compute, that is a meaningful result.
