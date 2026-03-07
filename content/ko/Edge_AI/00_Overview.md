# Edge AI 학습 가이드

## 소개

이 폴더는 **Edge AI** — 스마트폰, 마이크로컨트롤러, 임베디드 시스템, IoT 센서 등의 엣지 디바이스에서 머신러닝 모델을 직접 배포하는 기술에 대한 종합적인 가이드를 제공합니다. Edge AI는 추론을 위해 데이터를 클라우드로 전송할 필요 없이 낮은 지연 시간, 향상된 프라이버시, 감소된 대역폭 비용으로 실시간 의사결정을 가능하게 합니다.

교육 과정은 모델 압축 및 최적화부터 하드웨어별 배포까지의 전체 파이프라인을 다루며, 이론적 기초와 PyTorch, ONNX, TensorFlow Lite 및 벤더별 도구 체인을 사용한 실습 구현을 모두 포함합니다.

## 대상 학습자

- **Deep_Learning** 폴더(또는 CNN, Transformer, 학습 워크플로우에 대한 동등한 지식)를 완료한 학습자
- 리소스 제한 디바이스에 모델을 배포하는 데 관심 있는 엔지니어
- 효율적인 모델 설계 및 하드웨어 인식 최적화를 탐구하는 연구자
- 실시간 AI 애플리케이션(로봇, 자율주행, 스마트 카메라, 웨어러블)을 구축하는 모든 분

## 선행 학습

- **Deep_Learning**: CNN, 학습 루프, 손실 함수, PyTorch에 대한 탄탄한 이해
- **Computer_Architecture**: CPU/GPU 파이프라인, 메모리 계층 구조, 명령어 수준 병렬성에 대한 친숙함
- **IoT_Embedded**: 임베디드 시스템, 마이크로컨트롤러, 센서 인터페이스에 대한 기본 지식

## 학습 로드맵

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   기초            │────▶│   압축           │────▶│   최적화          │
│     L01-L02       │     │     L03-L05      │     │     L06-L07      │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
                                                          ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    실전           │◀────│    배포           │◀────│     내보내기 &    │
│     L14-L16       │     │     L10-L13      │     │    런타임         │
└──────────────────┘     └──────────────────┘     │     L08-L09      │
                                                  └──────────────────┘
```

**권장 학습 경로**:
1. 기초(L01-L02)부터 시작하여 Edge AI 제약 조건과 압축 분류 체계를 이해합니다
2. 압축 기법(L03-L05)을 숙달합니다 — quantization, pruning, knowledge distillation
3. 효율적인 아키텍처 설계(L06-L07)를 학습합니다 — MobileNet, EfficientNet, NAS
4. 내보내기 및 런타임(L08-L09)을 배웁니다 — ONNX, TensorFlow Lite, 추론 엔진
5. 하드웨어 배포(L10-L13)를 탐구합니다 — TensorRT, 모바일/MCU 타겟, 벤치마킹
6. 실전 프로젝트(L14-L16)로 지식을 적용합니다 — 엔드투엔드 Edge AI 애플리케이션

## 파일 목록

| 레슨 | 파일명 | 난이도 | 설명 |
|--------|----------|------------|-------------|
| **블록 1: 기초** |
| L01 | `01_Edge_AI_Fundamentals.md` | ⭐ | Edge vs cloud 추론, 지연 시간/프라이버시 트레이드오프, 엣지 컴퓨팅 스펙트럼 |
| L02 | `02_Model_Compression_Overview.md` | ⭐⭐ | 압축 분류 체계: pruning, quantization, distillation, NAS |
| **블록 2: 압축 기법** |
| L03 | `03_Quantization.md` | ⭐⭐⭐ | PTQ vs QAT, INT8/INT4, 대칭 vs 비대칭, 혼합 정밀도 |
| L04 | `04_Pruning.md` | ⭐⭐⭐ | 구조적 vs 비구조적, 크기 기반, lottery ticket 가설 |
| L05 | `05_Knowledge_Distillation.md` | ⭐⭐⭐ | Teacher-student 프레임워크, soft target, attention transfer |
| **블록 3: 효율적 아키텍처 설계** |
| L06 | `06_Efficient_Architectures.md` | ⭐⭐⭐ | MobileNet, EfficientNet, ShuffleNet, SqueezeNet, 설계 원칙 |
| L07 | `07_Neural_Architecture_Search.md` | ⭐⭐⭐⭐ | NAS 기초, 검색 전략, 하드웨어 인식 NAS |
| **블록 4: 내보내기 및 런타임** |
| L08 | `08_ONNX_and_Model_Export.md` | ⭐⭐⭐ | ONNX 형식, 그래프 최적화, 크로스 프레임워크 변환 |
| L09 | `09_TFLite_and_Mobile_Runtimes.md` | ⭐⭐⭐ | TensorFlow Lite, CoreML, NNAPI, delegate 시스템 |
| **블록 5: 하드웨어 배포** |
| L10 | `10_TensorRT_Optimization.md` | ⭐⭐⭐⭐ | NVIDIA TensorRT, 레이어 퓨전, INT8 캘리브레이션, 엔진 빌드 |
| L11 | `11_Mobile_Deployment.md` | ⭐⭐⭐ | Android (NNAPI), iOS (CoreML), 온디바이스 추론 파이프라인 |
| L12 | `12_MCU_Deployment.md` | ⭐⭐⭐⭐ | TinyML, TFLite Micro, CMSIS-NN, 메모리 제한 추론 |
| L13 | `13_Edge_Hardware_Landscape.md` | ⭐⭐⭐ | NPU, TPU Edge, Jetson, Coral, Hailo, 하드웨어 비교 |
| **블록 6: 실전 응용** |
| L14 | `14_Benchmarking_and_Profiling.md` | ⭐⭐⭐ | 지연 시간/처리량 측정, 전력 프로파일링, 루프라인 분석 |
| L15 | `15_Practical_Edge_Vision.md` | ⭐⭐⭐⭐ | 엔드투엔드: Raspberry Pi / Jetson Nano에서 객체 검출 |
| L16 | `16_Practical_Edge_NLP.md` | ⭐⭐⭐⭐ | 엔드투엔드: 온디바이스 텍스트 분류 및 키워드 검출 |

**총 16개 레슨** (개념 13개 + 실습/구현 3개)

## 환경 설정

### 핵심 설치

```bash
# PyTorch (모델 학습 및 내보내기용)
pip install torch torchvision

# ONNX 에코시스템
pip install onnx onnxruntime onnxoptimizer

# TensorFlow Lite (모바일/MCU 배포용)
pip install tensorflow tflite-runtime

# 프로파일링 및 벤치마킹
pip install thop fvcore
```

### 선택 도구

```bash
# NVIDIA TensorRT (NVIDIA GPU + CUDA 필요)
pip install tensorrt

# Edge TPU 컴파일러 (Google Coral용)
# 참조: https://coral.ai/docs/edgetpu/compiler/

# ARM NN SDK (ARM 기반 배포용)
# 참조: https://developer.arm.com/Tools%20and%20Software/Arm%20NN
```

### 설치 확인

```python
import torch
import onnx
import onnxruntime as ort

print(f"PyTorch: {torch.__version__}")
print(f"ONNX: {onnx.__version__}")
print(f"ONNX Runtime: {ort.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 관련 토픽

- **[Deep_Learning](../Deep_Learning/00_Overview.md)**: 선행 학습 — CNN 아키텍처, 학습, PyTorch 기초
- **[Computer_Vision](../Computer_Vision/00_Overview.md)**: 엣지에서 일반적으로 배포되는 비전 태스크(검출, 분할)
- **[IoT_Embedded](../IoT_Embedded/00_Overview.md)**: 임베디드 시스템, 센서, 마이크로컨트롤러 프로그래밍
- **[Foundation_Models](../Foundation_Models/00_Overview.md)**: 대규모 모델 압축 — LoRA, LLM용 quantization
- **[Computer_Architecture](../Computer_Architecture/00_Overview.md)**: 하드웨어 기초 — 메모리 계층 구조, 병렬성, 가속기

## 학습 팁

1. **프로파일링부터 시작하십시오**: 모델을 압축하기 전에 타겟 디바이스에서 기준 지연 시간과 메모리를 측정하십시오
2. **압축은 반복적입니다**: 한 번에 하나의 기법을 적용하고(quantize → prune → distill) 영향을 측정하십시오
3. **하드웨어가 중요합니다**: GPU에서 잘 작동하는 기법(예: 비구조적 pruning)이 모바일 NPU에서는 도움이 되지 않을 수 있습니다
4. **실제 디바이스에서 테스트하십시오**: 에뮬레이터와 시뮬레이터는 실제 지연 시간을 포착할 수 없습니다 — 항상 하드웨어에서 검증하십시오
5. **벤더 문서를 읽으십시오**: 각 하드웨어 플랫폼(TensorRT, CoreML, Edge TPU)은 특정 연산자 지원 및 제약 사항이 있습니다
6. **정확도 vs 효율성을 추적하십시오**: Pareto 곡선(정확도 vs 지연 시간/크기)을 만들어 최적점을 찾으십시오

## 학습 성과

이 폴더를 완료하면 다음을 할 수 있습니다:

- 클라우드 추론과 엣지 배포 간의 트레이드오프를 설명할 수 있습니다
- Quantization(PTQ, QAT)을 적용하여 최소한의 정확도 손실로 모델 크기를 2~4배 줄일 수 있습니다
- 구조적 및 비구조적 방법을 사용하여 신경망을 pruning할 수 있습니다
- Knowledge distillation을 통해 소형 student 모델을 학습시킬 수 있습니다
- MobileNet, EfficientNet, NAS 원칙을 사용하여 효율적인 아키텍처를 설계할 수 있습니다
- 모델을 ONNX 및 TensorFlow Lite 형식으로 내보낼 수 있습니다
- GPU(TensorRT), 모바일 디바이스, 마이크로컨트롤러에 최적화된 모델을 배포할 수 있습니다
- 지연 시간, 처리량, 전력에 대한 엣지 추론 파이프라인을 벤치마킹하고 프로파일링할 수 있습니다

## 다음 단계

- **LLM 압축**: 대규모 언어 모델의 quantization 및 LoRA에 대해서는 `Foundation_Models`를 참조하십시오
- **비전 응용**: 엣지에서의 검출, 추적, SLAM에 대해서는 `Computer_Vision`을 참조하십시오
- **프로덕션 파이프라인**: 모델 서빙, 모니터링, CI/CD에 대해서는 `MLOps`를 참조하십시오
- **하드웨어 설계**: 커스텀 가속기 설계에 대해서는 `Computer_Architecture`를 참조하십시오

---

**License**: CC BY-NC 4.0

[01_Edge_AI_Fundamentals.md](./01_Edge_AI_Fundamentals.md)에서 Edge AI 학습을 시작하십시오.
