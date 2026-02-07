# 24. TensorBoard 시각화

## 학습 목표

- TensorBoard의 핵심 기능과 활용 사례 이해
- PyTorch에서 TensorBoard 연동 방법 습득
- 학습 메트릭, 모델 그래프, 임베딩 시각화
- 하이퍼파라미터 튜닝 결과 비교 분석

---

## 1. TensorBoard 소개

### 1.1 TensorBoard란?

TensorBoard는 머신러닝 실험을 시각화하고 분석하기 위한 도구입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                      TensorBoard                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Scalars │  │ Images  │  │ Graphs  │  │Histograms│        │
│  │ (손실,  │  │ (샘플,  │  │ (모델   │  │ (가중치 │        │
│  │ 정확도) │  │ 생성물) │  │ 구조)   │  │ 분포)   │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│                                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │Embeddings│ │  Text   │  │ Audio   │  │ HParams │        │
│  │(t-SNE,  │  │ (로그,  │  │ (음성   │  │(하이퍼  │        │
│  │ PCA)    │  │ 샘플)   │  │ 샘플)   │  │파라미터)│        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 설치 및 실행

```bash
# 설치
pip install tensorboard

# 실행
tensorboard --logdir=runs --port=6006

# 브라우저에서 접속: http://localhost:6006
```

---

## 2. PyTorch와 TensorBoard 연동

### 2.1 SummaryWriter 기본 사용법

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

# SummaryWriter 생성
writer = SummaryWriter('runs/experiment_1')

# 스칼라 값 기록
for step in range(100):
    loss = 1.0 / (step + 1)  # 예시 손실값
    accuracy = step / 100.0   # 예시 정확도

    writer.add_scalar('Loss/train', loss, step)
    writer.add_scalar('Accuracy/train', accuracy, step)

# 종료
writer.close()
```

### 2.2 실험별 로그 디렉토리 구성

```python
from datetime import datetime
import os

def create_writer(experiment_name: str, extra: str = None) -> SummaryWriter:
    """실험별 고유 로그 디렉토리 생성"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if extra:
        log_dir = f'runs/{experiment_name}/{extra}/{timestamp}'
    else:
        log_dir = f'runs/{experiment_name}/{timestamp}'

    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)

# 사용 예시
writer = create_writer('mnist_cnn', 'lr_0.001_batch_32')
```

---

## 3. 스칼라 로깅 (Scalars)

### 3.1 학습/검증 메트릭 기록

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.writer = SummaryWriter()
        self.global_step = 0

    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # 배치별 로깅 (선택적)
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)
            self.global_step += 1

        # 에폭별 로깅
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        self.writer.add_scalar('Loss/train', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def validate(self, epoch: int):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total

        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)

        return val_loss, val_acc
```

### 3.2 여러 스칼라를 한 그래프에 표시

```python
# 방법 1: add_scalars 사용
writer.add_scalars('Loss', {
    'train': train_loss,
    'val': val_loss
}, epoch)

writer.add_scalars('Accuracy', {
    'train': train_acc,
    'val': val_acc
}, epoch)

# 방법 2: 같은 태그 경로 사용
# Loss/train과 Loss/val은 TensorBoard에서 자동으로 그룹화됨
```

### 3.3 학습률 스케줄러 로깅

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    train_one_epoch()
    scheduler.step()

    # 현재 학습률 기록
    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar('Learning_Rate', current_lr, epoch)
```

---

## 4. 이미지 로깅 (Images)

### 4.1 입력 이미지 시각화

```python
import torchvision
from torchvision import transforms

def log_images(writer, images, tag, step, normalize=True):
    """이미지 배치를 그리드로 시각화"""
    # 정규화된 이미지를 원래 범위로 복원 (선택적)
    if normalize:
        # ImageNet 정규화 역변환
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)

    # 그리드 생성
    grid = torchvision.utils.make_grid(images, nrow=8, padding=2)
    writer.add_image(tag, grid, step)

# 사용 예시
for batch_idx, (images, labels) in enumerate(train_loader):
    if batch_idx == 0:  # 첫 배치만 로깅
        log_images(writer, images[:32], 'Input/samples', epoch)
        break
```

### 4.2 생성 모델 출력 시각화

```python
class GANTrainer:
    def __init__(self, generator, discriminator, writer):
        self.G = generator
        self.D = discriminator
        self.writer = writer
        self.fixed_noise = torch.randn(64, 100, 1, 1)  # 고정 노이즈

    def log_generated_images(self, epoch):
        """생성된 이미지 시각화 (학습 진행 확인용)"""
        self.G.eval()
        with torch.no_grad():
            fake_images = self.G(self.fixed_noise.to(self.G.device))
            fake_images = (fake_images + 1) / 2  # [-1, 1] -> [0, 1]

        grid = torchvision.utils.make_grid(fake_images, nrow=8)
        self.writer.add_image('Generated/samples', grid, epoch)
        self.G.train()
```

### 4.3 특징 맵 시각화

```python
def visualize_feature_maps(model, image, writer, layer_name, step):
    """CNN 중간 레이어의 특징 맵 시각화"""
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # 훅 등록
    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(get_activation(layer_name))

    # 순전파
    model.eval()
    with torch.no_grad():
        _ = model(image.unsqueeze(0))

    # 특징 맵 추출
    feat = activation[layer_name].squeeze(0)  # [C, H, W]

    # 채널별로 시각화 (처음 16개)
    feat = feat[:16].unsqueeze(1)  # [16, 1, H, W]
    grid = torchvision.utils.make_grid(feat, nrow=4, normalize=True)
    writer.add_image(f'Features/{layer_name}', grid, step)

    handle.remove()
```

### 4.4 Grad-CAM 시각화

```python
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 훅 등록
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        # Grad-CAM 계산
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()

def log_gradcam(writer, model, image, target_layer, step):
    """Grad-CAM 결과를 TensorBoard에 로깅"""
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(image.unsqueeze(0))

    # 컬러맵 적용
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 원본 이미지
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0].imshow(img_np)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Grad-CAM
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')

    # 오버레이
    axes[2].imshow(img_np)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    writer.add_figure('GradCAM', fig, step)
    plt.close(fig)
```

---

## 5. 히스토그램 (Histograms)

### 5.1 가중치 분포 시각화

```python
def log_weights_histograms(writer, model, epoch):
    """모델 가중치 분포를 히스토그램으로 시각화"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 가중치 값
            writer.add_histogram(f'Weights/{name}', param.data, epoch)

            # 그래디언트 값 (있는 경우)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

# 학습 루프에서 사용
for epoch in range(num_epochs):
    train_one_epoch()

    # 매 10 에폭마다 히스토그램 로깅
    if epoch % 10 == 0:
        log_weights_histograms(writer, model, epoch)
```

### 5.2 활성화 값 분포 추적

```python
class ActivationLogger:
    """레이어별 활성화 값 분포 추적"""

    def __init__(self, model, writer):
        self.writer = writer
        self.activations = {}
        self.hooks = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def log(self, step):
        for name, activation in self.activations.items():
            self.writer.add_histogram(f'Activations/{name}', activation, step)
        self.activations.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
```

---

## 6. 모델 그래프 (Graphs)

### 6.1 모델 구조 시각화

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 모델 그래프 로깅
model = SimpleCNN()
dummy_input = torch.randn(1, 3, 32, 32)

writer = SummaryWriter('runs/model_graph')
writer.add_graph(model, dummy_input)
writer.close()
```

### 6.2 복잡한 모델 그래프

```python
# Transformer 모델 그래프
from torchvision.models import vit_b_16

model = vit_b_16(pretrained=False)
dummy_input = torch.randn(1, 3, 224, 224)

writer.add_graph(model, dummy_input)
```

---

## 7. 임베딩 시각화 (Embeddings)

### 7.1 t-SNE/PCA로 임베딩 시각화

```python
import torch
import torchvision
from torchvision import datasets, transforms

def extract_embeddings(model, dataloader, device):
    """모델의 마지막 레이어 전 임베딩 추출"""
    model.eval()
    embeddings = []
    labels = []
    images = []

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)

            # 마지막 FC 레이어 전까지 순전파
            # 모델 구조에 따라 수정 필요
            x = model.features(data)
            x = model.avgpool(x)
            emb = x.view(x.size(0), -1)

            embeddings.append(emb.cpu())
            labels.append(target)
            images.append(data.cpu())

    return (
        torch.cat(embeddings),
        torch.cat(labels),
        torch.cat(images)
    )

# 사용 예시
embeddings, labels, images = extract_embeddings(model, test_loader, device)

# TensorBoard에 임베딩 로깅
writer.add_embedding(
    embeddings,
    metadata=labels,
    label_img=images,
    global_step=epoch,
    tag='Embeddings/test_set'
)
```

### 7.2 단어 임베딩 시각화 (NLP)

```python
import torch.nn as nn

# 단어 임베딩 예시
vocab = ['king', 'queen', 'man', 'woman', 'prince', 'princess',
         'dog', 'cat', 'puppy', 'kitten']
embedding_dim = 128

embedding_layer = nn.Embedding(len(vocab), embedding_dim)

# 임베딩 벡터 추출
indices = torch.arange(len(vocab))
embeddings = embedding_layer(indices)

# TensorBoard에 로깅
writer.add_embedding(
    embeddings,
    metadata=vocab,
    tag='Word_Embeddings'
)
```

---

## 8. 하이퍼파라미터 튜닝 (HParams)

### 8.1 하이퍼파라미터 실험 로깅

```python
from torch.utils.tensorboard.summary import hparams

def train_with_hparams(lr, batch_size, optimizer_name, epochs=10):
    """하이퍼파라미터별 실험 실행"""

    # 고유 실험 디렉토리
    run_name = f'lr_{lr}_bs_{batch_size}_{optimizer_name}'
    writer = SummaryWriter(f'runs/hparam_search/{run_name}')

    # 모델 및 데이터 설정
    model = SimpleCNN().to(device)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 학습
    best_accuracy = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = validate(model, val_loader)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        best_accuracy = max(best_accuracy, val_acc)

    # 하이퍼파라미터와 최종 메트릭 기록
    hparam_dict = {
        'lr': lr,
        'batch_size': batch_size,
        'optimizer': optimizer_name
    }
    metric_dict = {
        'hparam/best_accuracy': best_accuracy,
        'hparam/final_loss': val_loss
    }

    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()

    return best_accuracy

# 그리드 서치 실행
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
optimizers = ['adam', 'sgd']

for lr in learning_rates:
    for bs in batch_sizes:
        for opt in optimizers:
            acc = train_with_hparams(lr, bs, opt)
            print(f'LR={lr}, BS={bs}, OPT={opt} -> Acc={acc:.2f}%')
```

### 8.2 Optuna와 TensorBoard 연동

```python
import optuna
from optuna.integration import TensorBoardCallback

def objective(trial):
    # 하이퍼파라미터 샘플링
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=64)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # 모델 생성
    model = create_model(n_layers, hidden_dim, dropout)

    # 학습 및 평가
    accuracy = train_and_evaluate(model, lr, batch_size)

    return accuracy

# TensorBoard 콜백과 함께 최적화
study = optuna.create_study(direction='maximize')
tensorboard_callback = TensorBoardCallback('runs/optuna/', metric_name='accuracy')

study.optimize(
    objective,
    n_trials=100,
    callbacks=[tensorboard_callback]
)

print(f'Best trial: {study.best_trial.params}')
print(f'Best accuracy: {study.best_value:.2f}%')
```

---

## 9. 커스텀 스칼라 레이아웃

### 9.1 대시보드 레이아웃 정의

```python
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import custom_scalars

# 커스텀 레이아웃 정의
layout = {
    'Training Metrics': {
        'loss': ['Multiline', ['Loss/train', 'Loss/val']],
        'accuracy': ['Multiline', ['Accuracy/train', 'Accuracy/val']],
    },
    'Learning Rate': {
        'lr': ['Multiline', ['Learning_Rate']],
    },
    'Per-Class Accuracy': {
        'classes': ['Multiline', [f'Accuracy/class_{i}' for i in range(10)]],
    },
}

writer = SummaryWriter('runs/custom_layout')
writer.add_custom_scalars(layout)

# 이후 일반적인 로깅
for epoch in range(100):
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Learning_Rate', lr, epoch)

    for i in range(10):
        writer.add_scalar(f'Accuracy/class_{i}', class_acc[i], epoch)
```

---

## 10. 텍스트 및 오디오 로깅

### 10.1 텍스트 로깅

```python
# 학습 로그 기록
writer.add_text('Hyperparameters', f'''
- Learning Rate: {lr}
- Batch Size: {batch_size}
- Optimizer: {optimizer_name}
- Epochs: {num_epochs}
''', 0)

# 모델 요약 기록
from torchinfo import summary

model_summary = str(summary(model, input_size=(1, 3, 224, 224), verbose=0))
writer.add_text('Model/summary', f'```\n{model_summary}\n```', 0)

# NLP 샘플 로깅
writer.add_text('Samples/input', 'The quick brown fox jumps over the lazy dog', 0)
writer.add_text('Samples/prediction', 'The fast brown fox jumps over the lazy dog', 0)
```

### 10.2 오디오 로깅

```python
import torchaudio

# 오디오 파일 로깅
waveform, sample_rate = torchaudio.load('audio.wav')
writer.add_audio('Audio/input', waveform, 0, sample_rate=sample_rate)

# 생성된 오디오 로깅 (예: TTS, 음악 생성)
generated_audio = model.generate(text_input)
writer.add_audio('Audio/generated', generated_audio, step, sample_rate=22050)
```

---

## 11. 프로파일링 (Profiler)

### 11.1 PyTorch Profiler와 TensorBoard

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# 프로파일링 설정
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,      # 워밍업
        warmup=1,    # 프로파일 준비
        active=3,    # 실제 프로파일링
        repeat=2     # 반복
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('runs/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, (data, target) in enumerate(train_loader):
        if step >= (1 + 1 + 3) * 2:
            break

        with record_function("data_loading"):
            data, target = data.to(device), target.to(device)

        with record_function("forward"):
            output = model(data)
            loss = criterion(output, target)

        with record_function("backward"):
            optimizer.zero_grad()
            loss.backward()

        with record_function("optimizer_step"):
            optimizer.step()

        prof.step()

# TensorBoard에서 PYTORCH_PROFILER 탭 확인
```

### 11.2 메모리 프로파일링

```python
def profile_memory(model, input_size, device='cuda'):
    """GPU 메모리 사용량 분석"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = model.to(device)
    x = torch.randn(input_size).to(device)

    # 순전파
    torch.cuda.synchronize()
    output = model(x)
    forward_memory = torch.cuda.max_memory_allocated() / 1e9

    # 역전파
    loss = output.sum()
    loss.backward()
    torch.cuda.synchronize()
    total_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f'Forward memory: {forward_memory:.2f} GB')
    print(f'Total memory (forward + backward): {total_memory:.2f} GB')

    return forward_memory, total_memory

# 로깅
fwd_mem, total_mem = profile_memory(model, (32, 3, 224, 224))
writer.add_scalar('Memory/forward_GB', fwd_mem, 0)
writer.add_scalar('Memory/total_GB', total_mem, 0)
```

---

## 12. 분산 학습에서 TensorBoard

### 12.1 DDP 환경에서 로깅

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def train_ddp():
    rank, world_size = setup_distributed()

    # Rank 0에서만 TensorBoard 로깅
    writer = SummaryWriter() if rank == 0 else None

    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    for epoch in range(num_epochs):
        # 로컬 메트릭 계산
        local_loss = train_one_epoch(model, train_loader)

        # 모든 프로세스의 손실 평균
        loss_tensor = torch.tensor([local_loss]).to(rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size

        # Rank 0에서만 로깅
        if writer is not None:
            writer.add_scalar('Loss/train', avg_loss, epoch)

    if writer is not None:
        writer.close()

    dist.destroy_process_group()
```

---

## 13. 실전 예제: 완전한 학습 파이프라인

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from datetime import datetime
import os

class TensorBoardTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler=None,
        device: str = 'cuda',
        experiment_name: str = 'default'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        # TensorBoard 설정
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f'runs/{experiment_name}/{timestamp}'
        self.writer = SummaryWriter(log_dir)

        # 모델 그래프 로깅
        dummy_input = next(iter(train_loader))[0][:1].to(device)
        self.writer.add_graph(model, dummy_input)

        # 하이퍼파라미터 로깅
        self._log_hyperparameters()

        self.global_step = 0
        self.best_val_acc = 0

    def _log_hyperparameters(self):
        hparams = {
            'lr': self.optimizer.param_groups[0]['lr'],
            'batch_size': self.train_loader.batch_size,
            'optimizer': self.optimizer.__class__.__name__,
            'model': self.model.__class__.__name__,
        }
        self.writer.add_text('Hyperparameters', str(hparams), 0)

    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # 배치별 손실 로깅
            self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)
            self.global_step += 1

            # 첫 배치 이미지 로깅
            if batch_idx == 0 and epoch % 10 == 0:
                grid = torchvision.utils.make_grid(data[:16])
                self.writer.add_image('Input/train_samples', grid, epoch)

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        self.writer.add_scalar('Loss/train', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        # 가중치 히스토그램 (매 10 에폭)
        if epoch % 10 == 0:
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f'Weights/{name}', param, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        return epoch_loss, epoch_acc

    def validate(self, epoch: int):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total

        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)

        # 최고 성능 갱신
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.writer.add_scalar('Best/val_accuracy', val_acc, epoch)

        return val_loss, val_acc

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            if self.scheduler:
                self.scheduler.step()
                self.writer.add_scalar(
                    'Learning_Rate',
                    self.scheduler.get_last_lr()[0],
                    epoch
                )

            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')

        # 최종 메트릭 로깅
        self.writer.add_hparams(
            {'lr': self.optimizer.param_groups[0]['lr']},
            {'hparam/best_accuracy': self.best_val_acc}
        )

        self.writer.close()
        print(f'\nTraining complete. Best Val Accuracy: {self.best_val_acc:.2f}%')


# 사용 예시
if __name__ == '__main__':
    # 데이터 준비
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 모델 설정
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 학습
    trainer = TensorBoardTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device='cuda',
        experiment_name='cifar10_resnet18'
    )

    trainer.train(num_epochs=50)
```

---

## 14. 팁과 Best Practices

### 14.1 로깅 주기 최적화

```python
# 배치별 로깅은 너무 자주하면 성능 저하
# 권장: 배치 손실은 100~500 스텝마다, 에폭 메트릭은 매 에폭

LOG_INTERVAL = 100

for batch_idx, (data, target) in enumerate(train_loader):
    # ... 학습 코드 ...

    if batch_idx % LOG_INTERVAL == 0:
        writer.add_scalar('Loss/train_step', loss.item(), global_step)
```

### 14.2 로그 파일 관리

```bash
# 오래된 로그 정리
find runs/ -type d -mtime +30 -exec rm -rf {} +

# 특정 실험만 유지
tensorboard --logdir=runs/experiment_final --port=6006
```

### 14.3 원격 TensorBoard

```bash
# 서버에서 TensorBoard 실행
tensorboard --logdir=runs --host=0.0.0.0 --port=6006

# 로컬에서 SSH 터널링
ssh -L 6006:localhost:6006 user@server

# 또는 ngrok 사용
ngrok http 6006
```

---

## 연습 문제

### 연습 1: 기본 로깅 구현
MNIST 분류 모델을 학습하면서 다음을 TensorBoard에 로깅하세요:
- 학습/검증 손실 및 정확도
- 학습률 변화
- 샘플 입력 이미지

### 연습 2: 모델 분석
학습된 CNN 모델에 대해:
- 가중치 히스토그램 시각화
- 특징 맵 시각화
- Grad-CAM 적용

### 연습 3: 하이퍼파라미터 튜닝
학습률, 배치 크기, 드롭아웃 비율에 대해:
- 그리드 서치 실행
- HParams 대시보드에서 결과 비교
- 최적 조합 찾기

---

## 참고 자료

- [TensorBoard 공식 문서](https://www.tensorflow.org/tensorboard)
- [PyTorch TensorBoard 튜토리얼](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
- [torch.utils.tensorboard API](https://pytorch.org/docs/stable/tensorboard.html)
