from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ..data import RLBenchStandardILDataset, StandardILCollator, StandardILConfig, build_store, infer_state_action_dims
from ..models import ModelConfig, TaskConditionerConfig, build_policy, validate_model_config
from ..utils import (
    all_reduce_dict_mean,
    barrier,
    count_parameters,
    finish_wandb,
    grad_norm,
    init_distributed,
    init_wandb,
    load_checkpoint,
    log_wandb,
    save_checkpoint,
    write_json,
)


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _build_loader(dataset: RLBenchStandardILDataset, *, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=StandardILCollator(),
        persistent_workers=int(num_workers) > 0,
    )


def train_diffusion(cfg: Any) -> None:
    ddp = init_distributed()
    cache_root = Path(str(cfg.data.cache_root)).expanduser().resolve()
    store, selected_tasks = build_store(cache_root, tasks=cfg.data.tasks)
    task_name_to_id = {task: idx for idx, task in enumerate(selected_tasks)}
    state_dim, action_dim = infer_state_action_dims(store)

    model_cfg = ModelConfig()
    model_cfg.encoder_name = str(cfg.model.encoder_name)
    model_cfg.policy_head = "diffusion"
    model_cfg.task_conditioner = TaskConditionerConfig(
        num_tasks=len(selected_tasks),
        d_model=int(cfg.model.d_model),
        n_task_tokens=int(cfg.model.n_task_tokens),
        dropout=float(model_cfg.task_conditioner.dropout),
    )
    for section_name, section in (
        ("perceiver_encoder", cfg.model.perceiver_encoder),
        ("supernode_encoder", cfg.model.supernode_encoder),
        ("supernode_nomsg_encoder", cfg.model.supernode_nomsg_encoder),
        ("diffusion_policy", cfg.model.diffusion_policy),
    ):
        target = getattr(model_cfg, section_name)
        for key, value in section.items():
            setattr(target, key, value)
    validate_model_config(model_cfg)

    train_cfg = StandardILConfig(
        T_obs=int(cfg.dataset.T_obs),
        H=int(cfg.dataset.H),
        stride=int(cfg.dataset.stride),
        split="train",
        task_sampling=str(cfg.data.task_sampling),
        task_sampling_alpha=float(cfg.data.task_sampling_alpha),
        max_train_episodes_per_variation=int(cfg.data.low_data_train_demos_per_variation),
        use_rgb=bool(cfg.data.use_rgb),
        use_mask_id=bool(cfg.data.use_mask_id),
    )
    val_cfg = StandardILConfig(
        T_obs=int(cfg.dataset.T_obs),
        H=int(cfg.dataset.H),
        stride=int(cfg.dataset.stride),
        split="val",
        task_sampling="variation_uniform",
        task_sampling_alpha=1.0,
        use_rgb=bool(cfg.data.use_rgb),
        use_mask_id=bool(cfg.data.use_mask_id),
    )
    train_dataset = RLBenchStandardILDataset(store, cfg=train_cfg, task_name_to_id=task_name_to_id, seed=int(cfg.seed) + 1000 * int(ddp.rank), num_samples=None)
    val_dataset = RLBenchStandardILDataset(store, cfg=val_cfg, task_name_to_id=task_name_to_id, seed=int(cfg.seed) + 50000, num_samples=int(cfg.train.val_num_samples))
    train_loader = _build_loader(train_dataset, batch_size=int(cfg.train.batch_size), num_workers=int(cfg.train.num_workers))
    val_loader = _build_loader(val_dataset, batch_size=int(cfg.train.val_batch_size), num_workers=int(cfg.train.num_workers))

    model = build_policy(model_cfg, state_dim=state_dim, action_dim=action_dim).to(ddp.device)
    if ddp.distributed:
        model = DDP(model, device_ids=[ddp.local_rank], output_device=ddp.local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optimizer.lr),
        betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp and ddp.device.type == "cuda"))

    run_dir = Path(str(cfg.output.root_dir)).expanduser().resolve() / str(cfg.output.run_name)
    ckpt_dir = Path(str(cfg.output.checkpoint_dir)).expanduser().resolve() / str(cfg.output.run_name)
    if ddp.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "resolved_config.json", cfg.to_dict())
    barrier()

    start_step = 0
    resume_path = str(getattr(cfg.train, "resume_path", "")).strip()
    if resume_path:
        ckpt = load_checkpoint(Path(resume_path).expanduser(), ddp.device)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))

    run = None
    try:
        total_params, trainable_params = count_parameters(model.module if isinstance(model, DDP) else model)
        if ddp.is_main_process:
            run = init_wandb(cfg.wandb, run_dir=run_dir, config_dict=cfg.to_dict())
            print(f"Model params: total={total_params:,} | trainable={trainable_params:,}")

        grad_accum = max(1, int(cfg.train.grad_accum_steps))
        train_iter = iter(train_loader)
        last_log_time = time.time()
        for step in range(start_step, int(cfg.train.num_steps)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            for _ in range(grad_accum):
                batch = _move_batch_to_device(next(train_iter), ddp.device)
                with torch.cuda.amp.autocast(enabled=bool(cfg.train.amp and ddp.device.type == "cuda")):
                    out = model.module.forward_loss(batch) if isinstance(model, DDP) else model.forward_loss(batch)
                    loss = out["loss"] / float(grad_accum)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                running_loss += float(out["loss"].item())
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad = grad_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.optimizer.grad_clip_norm))
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            metrics = {
                "train/loss": running_loss / float(grad_accum),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train/grad_norm": float(grad),
                "system/steps_per_sec": 1.0 / max(1e-6, time.time() - last_log_time),
            }
            if ddp.device.type == "cuda":
                metrics["system/gpu_mem_gb"] = float(torch.cuda.max_memory_allocated(ddp.device) / (1024 ** 3))
            metrics = all_reduce_dict_mean(metrics, ddp.device)
            last_log_time = time.time()
            if ddp.is_main_process and ((step + 1) % int(cfg.train.log_every) == 0 or step == 0):
                print(
                    f"step {step + 1}/{int(cfg.train.num_steps)} | "
                    f"loss {metrics['train/loss']:.6f} | grad {metrics['train/grad_norm']:.6f} | lr {metrics['train/lr']:.3e}"
                )
                log_wandb(run, metrics, step + 1)

            if (step + 1) % int(cfg.train.eval_every) == 0:
                model.eval()
                total = 0.0
                batches = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = _move_batch_to_device(val_batch, ddp.device)
                        out = model.module.forward_loss(val_batch) if isinstance(model, DDP) else model.forward_loss(val_batch)
                        total += float(out["loss"].item())
                        batches += 1
                        if batches >= int(cfg.train.val_num_batches):
                            break
                val_metrics = {"val/loss": total / float(max(1, batches))}
                val_metrics = all_reduce_dict_mean(val_metrics, ddp.device)
                if ddp.is_main_process:
                    print(f"val step {step + 1} | loss {val_metrics['val/loss']:.6f}")
                    log_wandb(run, val_metrics, step + 1)

            if ddp.is_main_process and (step + 1) % int(cfg.train.ckpt_every) == 0:
                save_checkpoint(
                    ckpt_dir / f"step_{step + 1:07d}.pt",
                    step=step + 1,
                    model=model.module if isinstance(model, DDP) else model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=cfg.to_dict(),
                    extra={
                        "task_name_to_id": task_name_to_id,
                        "state_dim": state_dim,
                        "action_dim": action_dim,
                    },
                )
    finally:
        store.close()
        finish_wandb(run)
