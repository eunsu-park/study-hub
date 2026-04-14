# PyTorch 생태계 (PyTorch Ecosystem)

**이전**: [TorchScript와 배포](./13_TorchScript_and_Deployment.md)

---

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 컴퓨터 비전 작업에 `torchvision`을 사용할 수 있습니다 (모델, 데이터셋, 변환)
2. 오디오 처리에 `torchaudio`를 사용할 수 있습니다
3. NLP 데이터 처리에 `torchtext`를 사용할 수 있습니다
4. PyTorch Lightning으로 연구를 가속화할 수 있습니다
5. HuggingFace Transformers에서 모델을 로드하고 파인튜닝할 수 있습니다
6. 작업에 적합한 생태계 라이브러리를 선택할 수 있습니다
7. PyTorch와 생태계 라이브러리 간의 관계를 이해할 수 있습니다

---

## 1. torchvision

### 1.1 사전 학습 모델

```python
import torchvision.models as models

# 사전 학습 모델 로드
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.eval()

# 파인튜닝: 최종 레이어 교체
resnet.fc = torch.nn.Linear(512, 5)  # 1000 대신 5 클래스
```

### 1.2 데이터셋

```python
from torchvision import datasets, transforms

mnist = datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.ToTensor())

# ImageFolder: 커스텀 데이터
# 구조: root/클래스명/이미지.jpg
custom = datasets.ImageFolder('./my_data/train', transform=transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
]))
print(custom.classes)       # ['cat', 'dog']
```

### 1.3 유틸리티 함수

```python
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

images = torch.randn(16, 3, 64, 64)
grid = make_grid(images, nrow=4, normalize=True)
plt.imshow(grid.permute(1, 2, 0))
plt.savefig('grid.png')
```

---

## 2. torchaudio

```python
import torchaudio

# 오디오 파일 로드
waveform, sample_rate = torchaudio.load('audio.wav')

# 리샘플링
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform_16k = resampler(waveform)

# 멜 스펙트로그램
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, n_mels=128,
)
mel_spec = mel_transform(waveform_16k)
```

---

## 3. PyTorch Lightning

### 3.1 LightningModule

```python
import lightning as L
import torch.nn.functional as F

class LitModel(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.lr = lr

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
```

### 3.2 Lightning으로 학습

```python
trainer = L.Trainer(
    max_epochs=20,
    accelerator='auto',
    devices='auto',
    callbacks=[
        L.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        L.callbacks.ModelCheckpoint(monitor='val_loss'),
    ],
)
trainer.fit(model, train_loader, val_loader)
```

### 3.3 Lightning vs 바닐라 PyTorch

| 기능 | 바닐라 PyTorch | Lightning |
|------|---------------|-----------|
| 학습 루프 | 직접 작성 | 자동화 |
| 멀티 GPU | DDP 수동 래핑 | `accelerator='gpu', devices=4` |
| 혼합 정밀도 | 수동 GradScaler | `precision='16-mixed'` |
| 체크포인팅 | 수동 | `ModelCheckpoint` 콜백 |
| 유연성 | 최대 | 매우 높음 (모든 메서드 오버라이드 가능) |

---

## 4. HuggingFace 생태계

### 4.1 Transformers 라이브러리

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

inputs = tokenizer("이 영화는 환상적이었어!", return_tensors="pt",
                    padding=True, truncation=True)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
```

### 4.2 HuggingFace Trainer로 파인튜닝

```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("imdb")

args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    learning_rate=2e-5,
)

trainer = Trainer(model=model, args=args,
                  train_dataset=tokenized["train"],
                  eval_dataset=tokenized["test"])
trainer.train()
```

---

## 5. 기타 주목할 라이브러리

| 라이브러리 | 도메인 | 핵심 기능 |
|-----------|--------|----------|
| **timm** | 비전 | 700+ 사전 학습 이미지 모델 |
| **Weights & Biases** | 실험 추적 | 대시보드, 스윕, 아티팩트 |
| **Optuna** | 하이퍼파라미터 튜닝 | 베이지안 최적화 |
| **DeepSpeed** | 분산 학습 | ZeRO 옵티마이저, 모델 병렬화 |
| **torchmetrics** | 메트릭 | 100+ 메트릭 (정확도, F1, BLEU 등) |

---

## 6. 적합한 도구 선택

```
시작:
    │
    ├── 사전 학습 비전 모델이 필요? → torchvision.models 또는 timm
    │
    ├── 사전 학습 NLP 모델이 필요? → HuggingFace Transformers
    │
    ├── 학습 보일러플레이트를 줄이고 싶으면? → PyTorch Lightning
    │
    ├── 모델 배포가 필요? → ONNX + ONNX Runtime 또는 TorchServe
    │
    └── 멀티 GPU 학습이 필요? → DistributedDataParallel 또는 DeepSpeed
```

---

## 요약

| 라이브러리 | 용도 | 사용 시기 |
|-----------|------|----------|
| torchvision | 비전 모델, 데이터셋, 변환 | 이미지 분류, 검출, 세그멘테이션 |
| torchaudio | 오디오 처리, 음성 모델 | 음성 인식, 오디오 분류 |
| Lightning | 학습 프레임워크 추상화 | 보일러플레이트 감소, 멀티 GPU, 로깅 |
| HuggingFace | NLP 모델, 데이터셋, 토크나이저 | 텍스트 분류, 생성, 파인튜닝 |
| timm | 광범위한 비전 모델 동물원 | torchvision에 없는 모델이 필요할 때 |

---

## 다음 단계

PyTorch Fundamentals를 완료한 것을 축하합니다! 이제 PyTorch로 신경망을 구축, 학습, 디버깅, 배포할 수 있는 실용적 기술을 갖추었습니다. 권장되는 다음 단계는:

- **[Deep_Learning](../Deep_Learning/00_Overview.md)**: CNN, Transformer, GAN, Diffusion 모델 구축
- **[Machine_Learning](../Machine_Learning/00_Overview.md)**: ML 이론 기초 강화
- **[CUDA](../CUDA/00_Overview.md)**: 저수준 GPU 프로그래밍 이해
- **[MLOps](../MLOps/00_Overview.md)**: 프로덕션 ML 파이프라인, 실험 추적, CI/CD
