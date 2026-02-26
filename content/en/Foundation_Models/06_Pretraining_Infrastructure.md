# 06. Pre-training Infrastructure

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the four main parallelism strategies (data, tensor, pipeline, sequence) and explain how they are combined in 3D parallelism to train models with hundreds of billions of parameters.
2. Estimate GPU memory requirements for a given model size and batch configuration, accounting for model parameters, gradients, optimizer states, and activations.
3. Apply memory optimization techniques including gradient checkpointing, ZeRO optimizer stages, mixed-precision (fp16/bf16) training, and activation offloading.
4. Explain how training stability techniques such as gradient clipping, loss scaling, and learning rate warm-up prevent common training failures (loss spikes, gradient explosions).
5. Configure a distributed training setup using frameworks such as Megatron-LM, DeepSpeed, or PyTorch FSDP, and justify parallelism strategy choices for a given model and hardware configuration.
6. Analyze throughput metrics (MFU, GPU utilization) and identify bottlenecks in large-scale training runs caused by communication overhead, load imbalance, or memory pressure.

---

## Overview

Training large-scale Foundation Models runs on thousands of GPUs for weeks to months. This lesson covers distributed training strategies, memory optimization, and training stability techniques.

---

## 1. Distributed Training Paradigms

### 1.1 Parallelization Strategy Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     Distributed Training Paradigms                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Data Parallelism (DP)         Tensor Parallelism (TP)          │
│  ┌─────┐ ┌─────┐               ┌──────────────────┐              │
│  │GPU 0│ │GPU 1│               │   W = [W1 | W2]  │              │
│  │Model│ │Model│               │GPU0    GPU1      │              │
│  │Data1│ │Data2│               │ W1      W2       │              │
│  └─────┘ └─────┘               └──────────────────┘              │
│  Same model, different data    Split layers across GPUs         │
│                                                                  │
│  Pipeline Parallelism (PP)     Sequence Parallelism (SP)        │
│  ┌─────┐ ┌─────┐               ┌────┬────┬────┬────┐             │
│  │GPU 0│ │GPU 1│               │ S1 │ S2 │ S3 │ S4 │             │
│  │L1-L6│→│L7-12│               │GPU0│GPU1│GPU2│GPU3│             │
│  └─────┘ └─────┘               └────┴────┴────┴────┘             │
│  Sequential layer split        Split sequence across GPUs       │
│                                                                  │
│  3D Parallelism: DP + TP + PP combination                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Memory Analysis

```python
def estimate_training_memory(
    num_params: int,  # Number of parameters
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    dtype_bytes: int = 2,  # fp16/bf16 = 2, fp32 = 4
    optimizer: str = 'adam'
) -> dict:
    """
    Estimate GPU memory during training

    Memory components:
    1. Model Parameters
    2. Gradients
    3. Optimizer States
    4. Activations (forward pass)
    """

    # 1. Model parameters
    param_memory = num_params * dtype_bytes

    # 2. Gradients (same as parameters)
    grad_memory = num_params * dtype_bytes

    # 3. Optimizer States
    if optimizer == 'adam':
        # Adam: momentum(fp32) + variance(fp32)
        optimizer_memory = num_params * 4 * 2  # 8 bytes per param
    elif optimizer == 'sgd':
        optimizer_memory = num_params * 4  # momentum only
    else:
        optimizer_memory = 0

    # 4. Activations (approximation)
    # Per layer: attention + FFN activations
    bytes_per_token = hidden_dim * dtype_bytes * 10  # approximation
    activation_memory = batch_size * seq_len * bytes_per_token * num_layers

    # Activation checkpointing reduces to 1/sqrt(L)

    total = param_memory + grad_memory + optimizer_memory + activation_memory

    return {
        'parameters_gb': param_memory / 1e9,
        'gradients_gb': grad_memory / 1e9,
        'optimizer_gb': optimizer_memory / 1e9,
        'activations_gb': activation_memory / 1e9,
        'total_gb': total / 1e9
    }


# Example: 7B model
memory = estimate_training_memory(
    num_params=7e9,
    batch_size=4,
    seq_len=2048,
    hidden_dim=4096,
    num_layers=32
)

print("7B model memory estimate:")
for key, value in memory.items():
    print(f"  {key}: {value:.1f} GB")

# Output:
# parameters_gb: 14.0 GB
# gradients_gb: 14.0 GB
# optimizer_gb: 56.0 GB
# activations_gb: ~21.5 GB (batch_size=4)
# total_gb: ~105.5 GB
```

---

## 2. FSDP (Fully Sharded Data Parallel)

### 2.1 FSDP Concept

```
┌─────────────────────────────────────────────────────────────┐
│                      FSDP Operating Principle                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Traditional DDP:                                           │
│  GPU 0: [Full Model] + [Data 0]                            │
│  GPU 1: [Full Model] + [Data 1]                            │
│  → Full model replicated on each GPU (inefficient)         │
│                                                             │
│  FSDP (Zero Stage 3):                                       │
│  GPU 0: [Shard 0] + [Data 0]                               │
│  GPU 1: [Shard 1] + [Data 1]                               │
│                                                             │
│  Forward: All-Gather to collect full parameters            │
│  Backward: Reduce-Scatter to distribute gradients          │
│                                                             │
│  Memory: (Params + Grads + Optim) / N + Activations        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 PyTorch FSDP Implementation

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import functools

def setup_fsdp_training():
    """Setup FSDP training"""

    # Initialize distributed
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Create model
    model = MyTransformerModel(config)

    # Mixed Precision settings
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,      # Parameters
        reduce_dtype=torch.bfloat16,     # Gradient reduction
        buffer_dtype=torch.bfloat16,     # Buffers
    )

    # Auto Wrap Policy: Shard at Transformer layer level
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # FSDP wrapping
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Zero-3
        mixed_precision=mixed_precision,
        auto_wrap_policy=wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=False),
        device_id=local_rank,
    )

    return model


def train_step_fsdp(model, batch, optimizer, scaler=None):
    """FSDP training step"""
    model.train()

    # Forward
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**batch)
        loss = outputs.loss

    # Backward
    loss.backward()

    # Gradient clipping (requires care with FSDP)
    model.clip_grad_norm_(max_norm=1.0)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


# Checkpoint save/load
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)

def save_fsdp_checkpoint(model, optimizer, path):
    """Save FSDP checkpoint"""

    # Full State Dict config
    full_state_dict_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        full_state_dict_config,
    ):
        state_dict = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)

        if dist.get_rank() == 0:
            torch.save({
                'model': state_dict,
                'optimizer': optim_state,
            }, path)

    dist.barrier()
```

---

## 3. DeepSpeed ZeRO

### 3.1 ZeRO Stage Comparison

```
┌────────────────────────────────────────────────────────────┐
│                     DeepSpeed ZeRO Stages                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Stage 1: Optimizer State Partitioning                    │
│  - Only optimizer states (Adam m, v) partitioned          │
│  - Memory savings: ~4x                                     │
│                                                            │
│  Stage 2: + Gradient Partitioning                         │
│  - Gradients also partitioned                             │
│  - Memory savings: ~8x                                     │
│                                                            │
│  Stage 3: + Parameter Partitioning                        │
│  - Parameters also partitioned (similar to FSDP)          │
│  - Memory savings: ~N (proportional to GPU count)         │
│                                                            │
│  ZeRO-Offload: Offload to CPU/NVMe                        │
│  ZeRO-Infinity: Support for infinite model size           │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 3.2 DeepSpeed Configuration

```python
# ds_config.json
ds_config = {
    "train_batch_size": 256,
    "gradient_accumulation_steps": 8,
    "train_micro_batch_size_per_gpu": 4,

    # FP16 settings
    "fp16": {
        "enabled": True,
        "loss_scale": 0,  # dynamic
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    # BF16 settings (alternative)
    "bf16": {
        "enabled": False
    },

    # ZeRO Stage 3
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",  # or "nvme"
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },

    # Gradient Checkpointing
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": True,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": False
    },

    # Optimizer
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },

    # Scheduler
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-4,
            "warmup_num_steps": 1000,
            "total_num_steps": 100000
        }
    }
}
```

### 3.3 DeepSpeed Training Code

```python
import deepspeed
import torch

def train_with_deepspeed():
    """DeepSpeed training loop"""

    # Model and data
    model = MyTransformerModel(config)
    train_dataloader = create_dataloader(...)

    # DeepSpeed initialization
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    # Training loop
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            # Forward
            outputs = model_engine(**batch)
            loss = outputs.loss

            # Backward (DeepSpeed handles gradient scaling/accumulation)
            model_engine.backward(loss)

            # Step
            model_engine.step()

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")

    # Save checkpoint
    model_engine.save_checkpoint("checkpoint_dir")


# Execute
# deepspeed --num_gpus=8 train.py --deepspeed_config ds_config.json
```

---

## 4. Activation Checkpointing (Gradient Checkpointing)

### 4.1 Concept

```
Normal Forward:
Layer 1 → [Save Act1] → Layer 2 → [Save Act2] → ... → Loss

Use Act1, Act2 during backward to compute gradients
→ Memory: O(L) - proportional to layer count

Activation Checkpointing:
Layer 1 → [Checkpoint] → Layer 2 → Layer 3 → [Checkpoint] → ... → Loss

Recompute from checkpoints during backward
→ Memory: O(√L) - square root of layer count
→ Computation: ~33% increase (recomputation cost)
```

### 4.2 Implementation

```python
import torch
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class TransformerBlockWithCheckpoint(nn.Module):
    """Transformer block with checkpointing"""

    def __init__(self, config, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, attention_mask=None):
        if self.use_checkpoint and self.training:
            # Use checkpointing
            return checkpoint(
                self._forward_impl,
                x, attention_mask,
                use_reentrant=False,  # PyTorch 2.0+ recommended
            )
        else:
            return self._forward_impl(x, attention_mask)

    def _forward_impl(self, x, attention_mask):
        # Attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask)
        x = residual + x

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class TransformerWithSelectiveCheckpoint(nn.Module):
    """Selective Checkpointing"""

    def __init__(self, config, checkpoint_ratio=0.5):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlockWithCheckpoint(
                config,
                # Only checkpoint some layers
                use_checkpoint=(i % int(1/checkpoint_ratio) == 0)
            )
            for i in range(config.num_layers)
        ])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x
```

---

## 5. Training Stability

### 5.1 Loss Spike Response

```python
class TrainingStabilizer:
    """Training stability management"""

    def __init__(
        self,
        loss_spike_threshold: float = 5.0,  # 5x compared to previous
        grad_norm_threshold: float = 10.0,
        window_size: int = 100
    ):
        self.loss_spike_threshold = loss_spike_threshold
        self.grad_norm_threshold = grad_norm_threshold
        self.window_size = window_size

        self.loss_history = []
        self.grad_norm_history = []
        self.skipped_steps = 0

    def check_loss_spike(self, loss: float) -> bool:
        """Detect loss spike"""
        if len(self.loss_history) < self.window_size:
            self.loss_history.append(loss)
            return False

        avg_loss = sum(self.loss_history[-self.window_size:]) / self.window_size

        if loss > avg_loss * self.loss_spike_threshold:
            print(f"⚠️ Loss spike detected: {loss:.4f} (avg: {avg_loss:.4f})")
            return True

        self.loss_history.append(loss)
        return False

    def check_grad_norm(self, model: nn.Module) -> tuple[float, bool]:
        """Check gradient norm"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        is_spike = total_norm > self.grad_norm_threshold

        if is_spike:
            print(f"⚠️ Gradient spike: {total_norm:.4f}")

        self.grad_norm_history.append(total_norm)
        return total_norm, is_spike

    def should_skip_step(self, loss: float, model: nn.Module) -> bool:
        """Decide whether to skip this step"""
        loss_spike = self.check_loss_spike(loss)
        _, grad_spike = self.check_grad_norm(model)

        if loss_spike or grad_spike:
            self.skipped_steps += 1
            return True

        return False


def stable_training_step(
    model, batch, optimizer, stabilizer, scaler=None
):
    """Stable training step"""

    # Forward
    with torch.cuda.amp.autocast():
        outputs = model(**batch)
        loss = outputs.loss

    # Check loss spike
    if stabilizer.should_skip_step(loss.item(), model):
        optimizer.zero_grad()
        print(f"Skipping step (total skipped: {stabilizer.skipped_steps})")
        return None

    # Backward
    if scaler:
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    optimizer.zero_grad()

    return loss.item()
```

### 5.2 Checkpoint Strategy

```python
import os
import shutil
from datetime import datetime

class CheckpointManager:
    """Checkpoint management"""

    def __init__(
        self,
        save_dir: str,
        max_checkpoints: int = 5,
        save_interval_steps: int = 1000,
        save_interval_hours: float = 1.0
    ):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.save_interval_steps = save_interval_steps
        self.save_interval_hours = save_interval_hours

        self.last_save_time = datetime.now()
        self.checkpoints = []

        os.makedirs(save_dir, exist_ok=True)

    def should_save(self, step: int) -> bool:
        """Decide whether to save checkpoint"""
        # Step-based
        if step % self.save_interval_steps == 0:
            return True

        # Time-based
        elapsed = (datetime.now() - self.last_save_time).total_seconds() / 3600
        if elapsed >= self.save_interval_hours:
            return True

        return False

    def save(
        self,
        model,
        optimizer,
        scheduler,
        step: int,
        loss: float,
        **extra
    ):
        """Save checkpoint"""
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

        # Save
        state = {
            'step': step,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            **extra
        }

        torch.save(state, checkpoint_path + ".pt")

        # Metadata
        self.checkpoints.append({
            'path': checkpoint_path,
            'step': step,
            'loss': loss,
            'time': datetime.now().isoformat()
        })

        self.last_save_time = datetime.now()

        # Remove old checkpoints
        self._cleanup()

        print(f"💾 Saved checkpoint: {checkpoint_name}")

    def _cleanup(self):
        """Clean up old checkpoints"""
        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            if os.path.exists(oldest['path'] + ".pt"):
                os.remove(oldest['path'] + ".pt")
                print(f"🗑️ Removed old checkpoint: {oldest['path']}")

    def load_latest(self) -> dict:
        """Load latest checkpoint"""
        if not self.checkpoints:
            # Find in directory
            files = sorted([
                f for f in os.listdir(self.save_dir)
                if f.startswith("checkpoint-") and f.endswith(".pt")
            ])

            if not files:
                return None

            latest = files[-1]
            return torch.load(os.path.join(self.save_dir, latest))

        return torch.load(self.checkpoints[-1]['path'] + ".pt")
```

---

## 6. Learning Rate Scheduling

### 6.1 Warmup + Cosine Decay

```python
import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    num_cycles: float = 0.5
):
    """
    Warmup + Cosine Decay scheduler

    Early training: Linear warmup (0 → max_lr)
    After: Cosine decay (max_lr → min_lr)
    """

    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))

        # Decay only to min_lr
        decayed = (1 - min_lr_ratio) * cosine_decay + min_lr_ratio

        return decayed

    return LambdaLR(optimizer, lr_lambda)


# WSD (Warmup-Stable-Decay) scheduler (Llama 2)
def get_wsd_schedule(
    optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float = 0.1
):
    """
    Warmup-Stable-Decay scheduler

    1. Warmup: 0 → max_lr
    2. Stable: maintain max_lr
    3. Decay: max_lr → min_lr (cosine)
    """
    total_steps = num_warmup_steps + num_stable_steps + num_decay_steps

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))

        elif current_step < num_warmup_steps + num_stable_steps:
            # Stable phase
            return 1.0

        else:
            # Decay phase
            decay_step = current_step - num_warmup_steps - num_stable_steps
            progress = float(decay_step) / float(max(1, num_decay_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (1 - min_lr_ratio) * cosine_decay + min_lr_ratio

    return LambdaLR(optimizer, lr_lambda)
```

---

## 7. Practice: Complete Training Script

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb

def main():
    """Complete distributed training script"""

    # 1. Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # Only rank 0 logs
    is_main = local_rank == 0

    if is_main:
        wandb.init(project="foundation-model-training")

    # 2. Configuration
    config = {
        'hidden_size': 4096,
        'num_layers': 32,
        'num_heads': 32,
        'vocab_size': 50257,
        'max_seq_len': 2048,
        'batch_size': 4,  # per GPU
        'gradient_accumulation': 8,
        'learning_rate': 3e-4,
        'warmup_steps': 2000,
        'total_steps': 100000,
        'weight_decay': 0.1,
        'max_grad_norm': 1.0,
    }

    effective_batch = config['batch_size'] * config['gradient_accumulation'] * world_size
    print(f"Effective batch size: {effective_batch}")

    # 3. Model
    model = TransformerModel(config).cuda()

    # Activation checkpointing
    model.gradient_checkpointing_enable()

    # DDP or FSDP
    model = DDP(model, device_ids=[local_rank])

    # 4. Data
    dataset = PretrainingDataset(config)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95),
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=config['total_steps'],
    )

    # 6. Utilities
    scaler = torch.cuda.amp.GradScaler()
    stabilizer = TrainingStabilizer()
    checkpoint_mgr = CheckpointManager("checkpoints")

    # Resume from checkpoint
    checkpoint = checkpoint_mgr.load_latest()
    start_step = 0
    if checkpoint:
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step']
        if is_main:
            print(f"Resumed from step {start_step}")

    # 7. Training loop
    model.train()
    global_step = start_step
    accumulated_loss = 0.0

    for epoch in range(100):  # Sufficiently large number
        sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}

            # Forward (Mixed Precision)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / config['gradient_accumulation']

            # Backward
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

            # Gradient Accumulation
            if (batch_idx + 1) % config['gradient_accumulation'] == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['max_grad_norm']
                )

                # Step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging
                if is_main and global_step % 10 == 0:
                    lr = scheduler.get_last_lr()[0]
                    wandb.log({
                        'loss': accumulated_loss,
                        'learning_rate': lr,
                        'grad_norm': grad_norm.item(),
                        'step': global_step,
                    })
                    print(f"Step {global_step}: loss={accumulated_loss:.4f}, lr={lr:.2e}")

                accumulated_loss = 0.0

                # Checkpoint
                if checkpoint_mgr.should_save(global_step):
                    if is_main:
                        checkpoint_mgr.save(
                            model.module, optimizer, scheduler,
                            global_step, accumulated_loss
                        )

                # Termination condition
                if global_step >= config['total_steps']:
                    break

        if global_step >= config['total_steps']:
            break

    # Cleanup
    dist.destroy_process_group()
    if is_main:
        wandb.finish()


if __name__ == "__main__":
    main()

# Execute:
# torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 \
#          --master_addr="master" --master_port=29500 train.py
```

---

## References

### Documentation
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSpeed](https://www.deepspeed.ai/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

### Papers
- Rajbhandari et al. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
- Narayanan et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters"

### Related Lessons
- [../Deep_Learning/11_Model_Deployment.md](../Deep_Learning/11_Model_Deployment.md)
- [../MLOps/08_Model_Serving_Basics.md](../MLOps/08_Model_Serving_Basics.md)

---

## Exercises

### Exercise 1: Memory Budget Estimation

A transformer model has the following configuration:
- Parameters: 7B (float16, 2 bytes each)
- Batch size: 4, sequence length: 2048
- Hidden dim: 4096, 32 layers
- Using AdamW optimizer (fp32 optimizer states)

Estimate the approximate GPU memory required for:
1. Model parameters
2. Optimizer states (AdamW: parameter copy + 1st moment + 2nd moment, all in fp32)
3. Gradients (fp32)

Which component dominates memory usage?

<details>
<summary>Show Answer</summary>

**1. Model parameters (fp16):**
```
7B parameters × 2 bytes = 14 GB
```

**2. Optimizer states (AdamW in fp32):**
AdamW maintains three fp32 copies per parameter:
- fp32 parameter master copy: 7B × 4 bytes = 28 GB
- 1st moment (momentum): 7B × 4 bytes = 28 GB
- 2nd moment (variance): 7B × 4 bytes = 28 GB
Total: **84 GB**

**3. Gradients (fp32):**
```
7B × 4 bytes = 28 GB
```

**Total (excluding activations):** 14 + 84 + 28 = **~126 GB**

**Which dominates:** The optimizer states (84 GB) are by far the largest component — 6× larger than the model parameters themselves. This is why ZeRO Stage 1 targets optimizer state sharding first: it immediately cuts the dominant cost.

Note: Activations add additional memory proportional to batch_size × seq_len × hidden_dim × num_layers, which for this config adds roughly 4 × 2048 × 4096 × 32 × 2 bytes ≈ 2 GB — relatively small compared to optimizer states.

</details>

---

### Exercise 2: Parallelism Strategy Selection

For each training scenario, recommend the most appropriate parallelism strategy (or combination) and justify your choice.

1. **Scenario A**: 7B parameter model, 8× A100 80GB GPUs on a single node. The model fits on a single GPU in fp16.
2. **Scenario B**: 70B parameter model, 64× A100 80GB GPUs across 8 nodes. The model does NOT fit on a single GPU in fp16.
3. **Scenario C**: 7B model, 1024-token sequences, but training is bottlenecked by attention computation with 8K context length.

<details>
<summary>Show Answer</summary>

**Scenario A: 7B, 8 GPUs, single node, fits on one GPU**
- **Recommendation: Data Parallelism (DDP or FSDP)**
- Since the model fits on a single GPU, no need to split the model itself. Data parallelism simply replicates the model and splits batches across 8 GPUs.
- FSDP (ZeRO-3) can be used to shard optimizer states even when the model fits, reducing per-GPU memory by ~4× and allowing larger batch sizes.
- Intra-node NVLink bandwidth makes all-reduce very fast.

**Scenario B: 70B, 64 GPUs, 8 nodes, doesn't fit on one GPU**
- **Recommendation: 3D Parallelism (TP + PP + DP)**
- **Tensor Parallelism (TP=4)**: Split each transformer layer across 4 GPUs within a node (leverages fast NVLink).
- **Pipeline Parallelism (PP=2)**: Split the 32+ layers across 2 pipeline stages across nodes (tolerates slower inter-node bandwidth).
- **Data Parallelism (DP=8)**: Replicate the resulting TP×PP "model replica" across 8 copies.
- Total GPUs: 4 × 2 × 8 = 64 ✓

**Scenario C: 7B, 8K context, attention bottleneck**
- **Recommendation: Sequence Parallelism (SP) + FlashAttention**
- With 8K context, standard attention is O(8K²) = 64M operations per layer — both slow and memory-hungry.
- **Sequence Parallelism** splits the sequence dimension across GPUs, dividing attention computation cost.
- **FlashAttention** reduces attention's memory footprint from O(n²) to O(n) by computing attention in tiles without materializing the full attention matrix.
- These two techniques can be combined.

</details>

---

### Exercise 3: Gradient Clipping Analysis

The lesson shows gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

1. Explain what global gradient norm clipping does mathematically.
2. Why is gradient clipping important for training stability, particularly in the early phases of large model training?
3. What would happen if you set `max_norm` too small (e.g., 0.001) vs too large (e.g., 1000)?

<details>
<summary>Show Answer</summary>

**1. What gradient norm clipping does:**

It computes the global L2 norm of all gradients concatenated into a single vector:
```
global_norm = sqrt(sum(g_i² for all parameter gradients g_i))
```
If `global_norm > max_norm`, it scales ALL gradients by `max_norm / global_norm`:
```
g_i ← g_i × (max_norm / global_norm)
```
This preserves the direction of the gradient update while bounding its magnitude.

**2. Why it matters for large model training stability:**

- Large models have many layers, and gradients can compound multiplicatively through layers (exploding gradients).
- In early training, parameter initialization can produce very large gradient magnitudes before the loss stabilizes.
- A single large gradient step can "destroy" previously learned representations, requiring many steps to recover.
- Loss spikes — sudden upward jumps in loss — are often caused by such large updates. Gradient clipping provides a hard ceiling on update magnitude.

**3. max_norm too small vs too large:**

- **max_norm = 0.001 (too small)**: Nearly every gradient update is severely scaled down. Training becomes extremely slow — the model barely moves in parameter space. Eventually the model may converge but requires far more steps. This is equivalent to using an extremely small effective learning rate.
- **max_norm = 1000 (too large)**: Clipping essentially never activates (most gradient norms are well below 1000). The model trains as if clipping doesn't exist — fine during normal training, but offers no protection against gradient explosions during unstable periods.

In practice, max_norm = 1.0 is a common default for large language models, with the understanding that it should activate rarely (only during instabilities).

</details>

---

### Exercise 4: ZeRO Stage Comparison

DeepSpeed's ZeRO optimizer has three stages. Complete the table below:

| Stage | What is sharded | Memory saving (approx.) | Communication overhead |
|-------|----------------|------------------------|----------------------|
| ZeRO-1 | ? | ? | ? |
| ZeRO-2 | ? | ? | ? |
| ZeRO-3 | ? | ? | ? |

<details>
<summary>Show Answer</summary>

| Stage | What is sharded | Memory saving (approx.) | Communication overhead |
|-------|----------------|------------------------|----------------------|
| **ZeRO-1** | Optimizer states (momentum + variance) across N GPUs | ~4× reduction in optimizer state memory | Minimal: only all-reduce after parameter update (same as DDP) |
| **ZeRO-2** | Optimizer states + gradients across N GPUs | ~8× reduction in optimizer + gradient memory | Moderate: reduce-scatter gradients instead of all-reduce (slightly more efficient than DDP) |
| **ZeRO-3** | Optimizer states + gradients + model parameters across N GPUs | ~N× reduction (linear with number of GPUs) | Significant: all-gather parameters before each forward/backward pass; adds communication latency per layer |

**Key insight:** ZeRO-3 achieves the best memory efficiency (can train models that are N× larger than a single GPU can hold), but at the cost of 2× more communication operations per training step compared to standard DDP. The communication overhead is usually worth it when model size exceeds single-GPU capacity, but for models that fit in memory, ZeRO-1 or ZeRO-2 is preferred for lower latency.

</details>
