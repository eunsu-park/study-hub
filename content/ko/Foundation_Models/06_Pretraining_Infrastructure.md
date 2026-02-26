# 06. Pre-training 인프라

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 네 가지 주요 병렬화 전략(데이터, 텐서, 파이프라인, 시퀀스)을 설명하고, 수천억 개 파라미터의 모델 학습을 위해 3D 병렬 처리(3D Parallelism)에서 이들이 어떻게 결합되는지 설명할 수 있습니다.
2. 모델 파라미터, 그래디언트(Gradient), 옵티마이저 상태(Optimizer States), 활성화(Activation)를 고려하여 주어진 모델 크기와 배치 구성에 대한 GPU 메모리 요구사항을 추정할 수 있습니다.
3. 그래디언트 체크포인팅(Gradient Checkpointing), ZeRO 옵티마이저 단계, 혼합 정밀도(fp16/bf16) 학습, 활성화 오프로딩(Activation Offloading)을 포함한 메모리 최적화 기법을 적용할 수 있습니다.
4. 그래디언트 클리핑(Gradient Clipping), 손실 스케일링(Loss Scaling), 학습률 워밍업(Learning Rate Warm-up) 등의 학습 안정성 기법이 일반적인 학습 실패(손실 급증, 그래디언트 폭발)를 방지하는 방법을 설명할 수 있습니다.
5. Megatron-LM, DeepSpeed, PyTorch FSDP 등의 프레임워크를 사용하여 분산 학습 환경을 구성하고, 주어진 모델 및 하드웨어 구성에 맞는 병렬화 전략 선택을 정당화할 수 있습니다.
6. 처리량 지표(MFU, GPU 활용률)를 분석하고, 통신 오버헤드, 부하 불균형, 메모리 압박으로 인한 대규모 학습 실행의 병목 현상을 식별할 수 있습니다.

---

## 개요

대규모 Foundation Model 학습은 수천 개의 GPU에서 수주에서 수개월간 진행됩니다. 이 레슨에서는 분산 학습 전략, 메모리 최적화, 학습 안정성 기법을 다룹니다.

---

## 1. 분산 학습 패러다임

### 1.1 병렬화 전략 개요

```
┌──────────────────────────────────────────────────────────────────┐
│                     분산 학습 패러다임                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Data Parallelism (DP)         Tensor Parallelism (TP)          │
│  ┌─────┐ ┌─────┐               ┌──────────────────┐              │
│  │GPU 0│ │GPU 1│               │   W = [W1 | W2]  │              │
│  │Model│ │Model│               │GPU0    GPU1      │              │
│  │Data1│ │Data2│               │ W1      W2       │              │
│  └─────┘ └─────┘               └──────────────────┘              │
│  동일 모델, 다른 데이터         레이어를 GPU간 분할               │
│                                                                  │
│  Pipeline Parallelism (PP)     Sequence Parallelism (SP)        │
│  ┌─────┐ ┌─────┐               ┌────┬────┬────┬────┐             │
│  │GPU 0│ │GPU 1│               │ S1 │ S2 │ S3 │ S4 │             │
│  │L1-L6│→│L7-12│               │GPU0│GPU1│GPU2│GPU3│             │
│  └─────┘ └─────┘               └────┴────┴────┴────┘             │
│  레이어를 순차 분할             시퀀스를 GPU간 분할               │
│                                                                  │
│  3D Parallelism: DP + TP + PP 조합                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 메모리 분석

```python
def estimate_training_memory(
    num_params: int,  # 파라미터 수
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    dtype_bytes: int = 2,  # fp16/bf16 = 2, fp32 = 4
    optimizer: str = 'adam'
) -> dict:
    """
    학습 시 GPU 메모리 추정

    메모리 구성:
    1. Model Parameters
    2. Gradients
    3. Optimizer States
    4. Activations (forward pass)
    """

    # 1. 모델 파라미터
    param_memory = num_params * dtype_bytes

    # 2. Gradients (파라미터와 동일)
    grad_memory = num_params * dtype_bytes

    # 3. Optimizer States
    if optimizer == 'adam':
        # Adam: momentum(fp32) + variance(fp32)
        optimizer_memory = num_params * 4 * 2  # 8 bytes per param
    elif optimizer == 'sgd':
        optimizer_memory = num_params * 4  # momentum only
    else:
        optimizer_memory = 0

    # 4. Activations (근사치)
    # 각 레이어: attention + FFN activations
    bytes_per_token = hidden_dim * dtype_bytes * 10  # 근사
    activation_memory = batch_size * seq_len * bytes_per_token * num_layers

    # Activation checkpointing 시 1/sqrt(L) 로 감소

    total = param_memory + grad_memory + optimizer_memory + activation_memory

    return {
        'parameters_gb': param_memory / 1e9,
        'gradients_gb': grad_memory / 1e9,
        'optimizer_gb': optimizer_memory / 1e9,
        'activations_gb': activation_memory / 1e9,
        'total_gb': total / 1e9
    }


# 예시: 7B 모델
memory = estimate_training_memory(
    num_params=7e9,
    batch_size=4,
    seq_len=2048,
    hidden_dim=4096,
    num_layers=32
)

print("7B 모델 메모리 추정:")
for key, value in memory.items():
    print(f"  {key}: {value:.1f} GB")

# 출력:
# parameters_gb: 14.0 GB
# gradients_gb: 14.0 GB
# optimizer_gb: 56.0 GB
# activations_gb: ~21.5 GB (batch_size=4)
# total_gb: ~105.5 GB
```

---

## 2. FSDP (Fully Sharded Data Parallel)

### 2.1 FSDP 개념

```
┌─────────────────────────────────────────────────────────────┐
│                      FSDP 동작 원리                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  기존 DDP:                                                  │
│  GPU 0: [Full Model] + [Data 0]                            │
│  GPU 1: [Full Model] + [Data 1]                            │
│  → 각 GPU에 전체 모델 복제 (비효율)                          │
│                                                             │
│  FSDP (Zero Stage 3):                                       │
│  GPU 0: [Shard 0] + [Data 0]                               │
│  GPU 1: [Shard 1] + [Data 1]                               │
│                                                             │
│  Forward 시: All-Gather로 전체 파라미터 수집                │
│  Backward 시: Reduce-Scatter로 gradient 분산               │
│                                                             │
│  메모리: (Params + Grads + Optim) / N + Activations         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 PyTorch FSDP 구현

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
    """FSDP 학습 설정"""

    # 분산 초기화
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 모델 생성
    model = MyTransformerModel(config)

    # Mixed Precision 설정
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,      # 파라미터
        reduce_dtype=torch.bfloat16,     # gradient reduction
        buffer_dtype=torch.bfloat16,     # 버퍼
    )

    # Auto Wrap Policy: Transformer 레이어 단위로 샤딩
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # FSDP 래핑
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
    """FSDP 학습 스텝"""
    model.train()

    # Forward
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**batch)
        loss = outputs.loss

    # Backward
    loss.backward()

    # Gradient clipping (FSDP에서는 주의 필요)
    model.clip_grad_norm_(max_norm=1.0)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


# 체크포인트 저장/로드
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)

def save_fsdp_checkpoint(model, optimizer, path):
    """FSDP 체크포인트 저장"""

    # Full State Dict 설정
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

### 3.1 ZeRO 단계별 비교

```
┌────────────────────────────────────────────────────────────┐
│                     DeepSpeed ZeRO 단계                    │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Stage 1: Optimizer State Partitioning                    │
│  - Optimizer states (Adam m, v)만 분할                    │
│  - 메모리 절감: ~4x                                        │
│                                                            │
│  Stage 2: + Gradient Partitioning                         │
│  - Gradients도 분할                                        │
│  - 메모리 절감: ~8x                                        │
│                                                            │
│  Stage 3: + Parameter Partitioning                        │
│  - Parameters도 분할 (FSDP와 유사)                         │
│  - 메모리 절감: ~N (GPU 수에 비례)                         │
│                                                            │
│  ZeRO-Offload: CPU/NVMe로 오프로드                         │
│  ZeRO-Infinity: 무한 모델 크기 지원                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 3.2 DeepSpeed 설정

```python
# ds_config.json
ds_config = {
    "train_batch_size": 256,
    "gradient_accumulation_steps": 8,
    "train_micro_batch_size_per_gpu": 4,

    # FP16 설정
    "fp16": {
        "enabled": True,
        "loss_scale": 0,  # dynamic
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    # BF16 설정 (대안)
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

### 3.3 DeepSpeed 학습 코드

```python
import deepspeed
import torch

def train_with_deepspeed():
    """DeepSpeed 학습 루프"""

    # 모델 및 데이터
    model = MyTransformerModel(config)
    train_dataloader = create_dataloader(...)

    # DeepSpeed 초기화
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    # 학습 루프
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            # Forward
            outputs = model_engine(**batch)
            loss = outputs.loss

            # Backward (DeepSpeed가 gradient scaling/accumulation 처리)
            model_engine.backward(loss)

            # Step
            model_engine.step()

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")

    # 체크포인트 저장
    model_engine.save_checkpoint("checkpoint_dir")


# 실행
# deepspeed --num_gpus=8 train.py --deepspeed_config ds_config.json
```

---

## 4. Activation Checkpointing (Gradient Checkpointing)

### 4.1 개념

```
일반 Forward:
Layer 1 → [Act1 저장] → Layer 2 → [Act2 저장] → ... → Loss

Backward 시 Act1, Act2 등을 사용하여 gradient 계산
→ 메모리: O(L) - 레이어 수에 비례

Activation Checkpointing:
Layer 1 → [체크포인트] → Layer 2 → Layer 3 → [체크포인트] → ... → Loss

Backward 시 체크포인트에서 재계산
→ 메모리: O(√L) - 루트 레이어 수
→ 계산: ~33% 증가 (재계산 비용)
```

### 4.2 구현

```python
import torch
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class TransformerBlockWithCheckpoint(nn.Module):
    """Checkpointing이 적용된 Transformer 블록"""

    def __init__(self, config, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, attention_mask=None):
        if self.use_checkpoint and self.training:
            # Checkpointing 사용
            return checkpoint(
                self._forward_impl,
                x, attention_mask,
                use_reentrant=False,  # PyTorch 2.0+ 권장
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
    """선택적 Checkpointing"""

    def __init__(self, config, checkpoint_ratio=0.5):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlockWithCheckpoint(
                config,
                # 일부 레이어만 checkpoint
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

## 5. 학습 안정성

### 5.1 Loss Spike 대응

```python
class TrainingStabilizer:
    """학습 안정성 관리"""

    def __init__(
        self,
        loss_spike_threshold: float = 5.0,  # 이전 대비 5배
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
        """Loss spike 감지"""
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
        """Gradient norm 체크"""
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
        """해당 step을 건너뛸지 결정"""
        loss_spike = self.check_loss_spike(loss)
        _, grad_spike = self.check_grad_norm(model)

        if loss_spike or grad_spike:
            self.skipped_steps += 1
            return True

        return False


def stable_training_step(
    model, batch, optimizer, stabilizer, scaler=None
):
    """안정적인 학습 스텝"""

    # Forward
    with torch.cuda.amp.autocast():
        outputs = model(**batch)
        loss = outputs.loss

    # Loss spike 체크
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

### 5.2 체크포인트 전략

```python
import os
import shutil
from datetime import datetime

class CheckpointManager:
    """체크포인트 관리"""

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
        """체크포인트 저장 여부 결정"""
        # 스텝 기반
        if step % self.save_interval_steps == 0:
            return True

        # 시간 기반
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
        """체크포인트 저장"""
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

        # 저장
        state = {
            'step': step,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            **extra
        }

        torch.save(state, checkpoint_path + ".pt")

        # 메타데이터
        self.checkpoints.append({
            'path': checkpoint_path,
            'step': step,
            'loss': loss,
            'time': datetime.now().isoformat()
        })

        self.last_save_time = datetime.now()

        # 오래된 체크포인트 삭제
        self._cleanup()

        print(f"💾 Saved checkpoint: {checkpoint_name}")

    def _cleanup(self):
        """오래된 체크포인트 정리"""
        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            if os.path.exists(oldest['path'] + ".pt"):
                os.remove(oldest['path'] + ".pt")
                print(f"🗑️ Removed old checkpoint: {oldest['path']}")

    def load_latest(self) -> dict:
        """최신 체크포인트 로드"""
        if not self.checkpoints:
            # 디렉토리에서 찾기
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

## 6. 학습률 스케줄링

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
    Warmup + Cosine Decay 스케줄러

    학습 초기: Linear warmup (0 → max_lr)
    이후: Cosine decay (max_lr → min_lr)
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

        # min_lr까지만 감소
        decayed = (1 - min_lr_ratio) * cosine_decay + min_lr_ratio

        return decayed

    return LambdaLR(optimizer, lr_lambda)


# WSD (Warmup-Stable-Decay) 스케줄러 (Llama 2)
def get_wsd_schedule(
    optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float = 0.1
):
    """
    Warmup-Stable-Decay 스케줄러

    1. Warmup: 0 → max_lr
    2. Stable: max_lr 유지
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

## 7. 실습: 완전한 학습 스크립트

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb

def main():
    """완전한 분산 학습 스크립트"""

    # 1. 분산 초기화
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # Rank 0만 로깅
    is_main = local_rank == 0

    if is_main:
        wandb.init(project="foundation-model-training")

    # 2. 설정
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

    # 3. 모델
    model = TransformerModel(config).cuda()

    # Activation checkpointing
    model.gradient_checkpointing_enable()

    # DDP 또는 FSDP
    model = DDP(model, device_ids=[local_rank])

    # 4. 데이터
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

    # 6. 유틸리티
    scaler = torch.cuda.amp.GradScaler()
    stabilizer = TrainingStabilizer()
    checkpoint_mgr = CheckpointManager("checkpoints")

    # 체크포인트 복원
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

    # 7. 학습 루프
    model.train()
    global_step = start_step
    accumulated_loss = 0.0

    for epoch in range(100):  # 충분히 큰 수
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

                # 로깅
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

                # 체크포인트
                if checkpoint_mgr.should_save(global_step):
                    if is_main:
                        checkpoint_mgr.save(
                            model.module, optimizer, scheduler,
                            global_step, accumulated_loss
                        )

                # 종료 조건
                if global_step >= config['total_steps']:
                    break

        if global_step >= config['total_steps']:
            break

    # 정리
    dist.destroy_process_group()
    if is_main:
        wandb.finish()


if __name__ == "__main__":
    main()

# 실행:
# torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 \
#          --master_addr="master" --master_port=29500 train.py
```

---

## 참고 자료

### 문서
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSpeed](https://www.deepspeed.ai/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

### 논문
- Rajbhandari et al. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
- Narayanan et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters"

### 관련 레슨
- [../Deep_Learning/11_Model_Deployment.md](../Deep_Learning/11_Model_Deployment.md)
- [../MLOps/08_Model_Serving_Basics.md](../MLOps/08_Model_Serving_Basics.md)

---

## 연습 문제

### 연습 문제 1: 메모리 예산 추정

트랜스포머 모델의 구성이 다음과 같습니다:
- 파라미터: 7B (float16, 각 2 바이트)
- 배치 크기: 4, 시퀀스 길이: 2048
- 히든 차원: 4096, 32 레이어
- AdamW 옵티마이저 사용 (fp32 옵티마이저 상태)

다음 각 항목에 필요한 GPU 메모리를 추정하세요:
1. 모델 파라미터
2. 옵티마이저 상태 (AdamW: 파라미터 복사본 + 1차 모멘트 + 2차 모멘트, 모두 fp32)
3. 그래디언트 (fp32)

어느 항목이 메모리 사용량을 지배하나요?

<details>
<summary>정답 보기</summary>

**1. 모델 파라미터 (fp16):**
```
7B 파라미터 × 2 바이트 = 14 GB
```

**2. 옵티마이저 상태 (AdamW, fp32):**
AdamW는 파라미터마다 세 가지 fp32 복사본을 유지합니다:
- fp32 마스터 파라미터 복사본: 7B × 4 바이트 = 28 GB
- 1차 모멘트: 7B × 4 바이트 = 28 GB
- 2차 모멘트: 7B × 4 바이트 = 28 GB
합계: **84 GB**

**3. 그래디언트 (fp32):**
```
7B × 4 바이트 = 28 GB
```

**합계 (활성화 제외):** 14 + 84 + 28 = **~126 GB**

**지배적인 항목:** 옵티마이저 상태(84 GB)가 단연 가장 큰 항목입니다 — 모델 파라미터 자체보다 6배 더 큽니다. 이것이 ZeRO Stage 1이 옵티마이저 상태 샤딩을 먼저 타겟으로 하는 이유입니다: 지배적인 비용을 즉시 절감합니다.

참고: 활성화(Activation)는 batch_size × seq_len × hidden_dim × num_layers에 비례하는 추가 메모리를 사용하며, 이 구성에서는 약 4 × 2048 × 4096 × 32 × 2 바이트 ≈ 2 GB — 옵티마이저 상태에 비해 상대적으로 작습니다.

</details>

---

### 연습 문제 2: 병렬화 전략 선택

각 학습 시나리오에 대해 가장 적합한 병렬화 전략(또는 조합)을 추천하고 이유를 설명하세요.

1. **시나리오 A**: 7B 파라미터 모델, 단일 노드의 8× A100 80GB GPU. 모델이 fp16에서 단일 GPU에 맞음.
2. **시나리오 B**: 70B 파라미터 모델, 8개 노드에 걸쳐 64× A100 80GB GPU. 모델이 fp16에서 단일 GPU에 맞지 않음.
3. **시나리오 C**: 7B 모델, 1024 토큰 시퀀스지만, 8K 컨텍스트 길이에서 어텐션 계산이 병목.

<details>
<summary>정답 보기</summary>

**시나리오 A: 7B, 8 GPU, 단일 노드, 단일 GPU에 맞음**
- **권장: 데이터 병렬화(DDP 또는 FSDP)**
- 모델이 단일 GPU에 맞으므로 모델 자체를 분할할 필요가 없습니다. 데이터 병렬화는 단순히 모델을 복제하고 배치를 8개 GPU에 분산합니다.
- FSDP(ZeRO-3)를 사용하면 모델이 메모리에 맞더라도 옵티마이저 상태를 샤딩하여 GPU당 메모리를 ~4배 줄이고 더 큰 배치 크기를 허용합니다.
- 노드 내 NVLink 대역폭으로 all-reduce가 매우 빠릅니다.

**시나리오 B: 70B, 64 GPU, 8 노드, 단일 GPU에 맞지 않음**
- **권장: 3D 병렬화 (TP + PP + DP)**
- **텐서 병렬화(TP=4)**: 노드 내에서 각 트랜스포머 레이어를 4개 GPU에 분할(빠른 NVLink 활용).
- **파이프라인 병렬화(PP=2)**: 32+ 레이어를 2개 파이프라인 스테이지로 분할하여 노드 간 배분(더 느린 노드 간 대역폭 허용).
- **데이터 병렬화(DP=8)**: TP×PP "모델 복제본"을 8개 복사본으로 복제.
- 총 GPU: 4 × 2 × 8 = 64 ✓

**시나리오 C: 7B, 8K 컨텍스트, 어텐션 병목**
- **권장: 시퀀스 병렬화(SP) + FlashAttention**
- 8K 컨텍스트에서 표준 어텐션은 레이어당 O(8K²) = 6400만 연산 — 느리고 메모리 집약적입니다.
- **시퀀스 병렬화(Sequence Parallelism)**는 시퀀스 차원을 GPU에 분산하여 어텐션 계산 비용을 분산합니다.
- **FlashAttention**은 전체 어텐션 행렬을 구체화하지 않고 타일(tile) 단위로 계산하여 어텐션의 메모리 공간을 O(n²)에서 O(n)으로 줄입니다.
- 이 두 기법을 결합할 수 있습니다.

</details>

---

### 연습 문제 3: 그래디언트 클리핑(Gradient Clipping) 분석

수업에서 그래디언트 클리핑을 보여줍니다: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

1. 전역 그래디언트 노름(global gradient norm) 클리핑이 수학적으로 무엇을 하는지 설명하세요.
2. 왜 그래디언트 클리핑이 학습 안정성에 중요한가요, 특히 대형 모델 학습의 초기 단계에서?
3. `max_norm`을 너무 작게 설정하면(예: 0.001) 또는 너무 크게 설정하면(예: 1000) 어떻게 되나요?

<details>
<summary>정답 보기</summary>

**1. 그래디언트 노름 클리핑의 수행 내용:**

단일 벡터로 연결된 모든 그래디언트의 전역 L2 노름을 계산합니다:
```
global_norm = sqrt(sum(g_i² for all parameter gradients g_i))
```
`global_norm > max_norm`이면, 모든 그래디언트를 `max_norm / global_norm`으로 스케일링합니다:
```
g_i ← g_i × (max_norm / global_norm)
```
이를 통해 그래디언트 업데이트의 방향은 유지하면서 크기를 제한합니다.

**2. 대형 모델 학습 안정성에 중요한 이유:**

- 대형 모델은 많은 레이어를 가지며, 그래디언트는 레이어를 통해 곱셈적으로 증폭될 수 있습니다(그래디언트 폭발).
- 학습 초기에는 파라미터 초기화가 손실이 안정되기 전에 매우 큰 그래디언트 크기를 생성할 수 있습니다.
- 단일 대형 그래디언트 스텝이 이전에 학습된 표현을 "파괴"하여 많은 스텝이 복구에 필요하게 됩니다.
- 손실 스파이크(Loss spike) — 손실의 갑작스러운 상승 — 은 종종 이런 대형 업데이트로 인해 발생합니다. 그래디언트 클리핑은 업데이트 크기에 상한선을 제공합니다.

**3. max_norm 너무 작음 vs 너무 큼:**

- **max_norm = 0.001 (너무 작음)**: 거의 모든 그래디언트 업데이트가 심하게 스케일 다운됩니다. 학습이 극도로 느려집니다 — 모델이 파라미터 공간에서 거의 이동하지 않습니다. 결국 모델이 수렴할 수 있지만 훨씬 더 많은 스텝이 필요합니다. 이는 매우 작은 유효 학습률을 사용하는 것과 동일합니다.
- **max_norm = 1000 (너무 큼)**: 클리핑이 사실상 활성화되지 않습니다(대부분의 그래디언트 노름은 1000 이하). 모델은 클리핑이 없는 것처럼 학습됩니다 — 정상 학습 중에는 괜찮지만 불안정한 기간 동안 그래디언트 폭발로부터 보호받지 못합니다.

실제로 max_norm = 1.0은 대형 언어 모델의 일반적인 기본값이며, 이는 불안정성 발생 시에만 드물게 활성화되어야 합니다.

</details>

---

### 연습 문제 4: ZeRO 스테이지 비교

DeepSpeed의 ZeRO 옵티마이저에는 세 가지 스테이지가 있습니다. 아래 표를 완성하세요:

| 스테이지 | 샤딩 대상 | 메모리 절감 (근사치) | 통신 오버헤드 |
|---------|---------|-----------------|------------|
| ZeRO-1 | ? | ? | ? |
| ZeRO-2 | ? | ? | ? |
| ZeRO-3 | ? | ? | ? |

<details>
<summary>정답 보기</summary>

| 스테이지 | 샤딩 대상 | 메모리 절감 (근사치) | 통신 오버헤드 |
|---------|---------|-----------------|------------|
| **ZeRO-1** | N개 GPU에 옵티마이저 상태(모멘텀 + 분산) 분산 | 옵티마이저 상태 메모리 약 4배 감소 | 최소: 파라미터 업데이트 후 all-reduce만 (DDP와 동일) |
| **ZeRO-2** | N개 GPU에 옵티마이저 상태 + 그래디언트 분산 | 옵티마이저 + 그래디언트 메모리 약 8배 감소 | 보통: all-reduce 대신 reduce-scatter 그래디언트 (DDP보다 약간 더 효율적) |
| **ZeRO-3** | N개 GPU에 옵티마이저 상태 + 그래디언트 + 모델 파라미터 분산 | ~N배 감소 (GPU 수에 선형 비례) | 상당함: 각 포워드/백워드 패스 전에 파라미터 all-gather 필요; 레이어당 통신 지연 추가 |

**핵심 인사이트:** ZeRO-3는 최고의 메모리 효율성을 달성합니다(단일 GPU 용량의 N배 더 큰 모델 학습 가능), 하지만 표준 DDP 대비 학습 스텝당 2배 더 많은 통신 연산의 비용이 발생합니다. 모델 크기가 단일 GPU 용량을 초과할 때 통신 오버헤드는 가치가 있지만, 메모리에 맞는 모델의 경우 낮은 지연 시간을 위해 ZeRO-1 또는 ZeRO-2가 선호됩니다.

</details>
