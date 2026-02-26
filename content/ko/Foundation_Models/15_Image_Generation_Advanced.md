# 15. Image Generation 심화

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Stable Diffusion 1.5 대비 SDXL의 아키텍처 개선 사항을 설명하고 이중 텍스트 인코더(dual text encoder)가 이미지 생성 품질을 어떻게 향상시키는지 서술할 수 있다
2. ControlNet을 구현하여 깊이 맵(depth map), 엣지 맵(edge map), 포즈 스켈레톤(pose skeleton)을 이용해 이미지 생성의 공간적 제어를 달성할 수 있다
3. IP-Adapter를 적용하여 프롬프트 기반 콘텐츠 생성을 유지하면서 이미지 스타일과 아이덴티티를 전이할 수 있다
4. Latent Consistency Models(LCM)을 표준 확산 샘플링(diffusion sampling)과 비교하고 속도-품질 트레이드오프를 설명할 수 있다
5. 여러 조건화(conditioning) 기법을 결합한 엔드투엔드(end-to-end) 이미지 생성 파이프라인을 설계하여 세밀한 제어를 달성할 수 있다

---

## 개요

이 레슨에서는 Stable Diffusion 이후의 최신 이미지 생성 기술을 다룹니다. SDXL, ControlNet, IP-Adapter, Latent Consistency Models 등 실용적인 기법을 학습합니다.

---

## 1. SDXL (Stable Diffusion XL)

### 1.1 아키텍처 개선

```
┌──────────────────────────────────────────────────────────────────┐
│                    SDXL vs SD 1.5 비교                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SD 1.5:                                                         │
│  - UNet: 860M params                                            │
│  - Text Encoder: CLIP ViT-L/14 (77 토큰)                        │
│  - 해상도: 512×512                                              │
│  - VAE: 4× downscale                                            │
│                                                                  │
│  SDXL:                                                           │
│  - UNet: 2.6B params (3배 증가)                                 │
│  - Text Encoder: CLIP ViT-L + OpenCLIP ViT-bigG (이중)          │
│  - 해상도: 1024×1024                                            │
│  - VAE: 개선된 VAE-FT                                           │
│  - Refiner 모델 (선택적)                                         │
│                                                                  │
│  주요 개선:                                                      │
│  - 더 풍부한 텍스트 이해 (이중 인코더)                           │
│  - 고해상도 생성 (4배 픽셀)                                     │
│  - Micro-conditioning (크기, 종횡비)                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 SDXL 사용

```python
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

def sdxl_generation():
    """SDXL 이미지 생성"""

    # Base 모델 로드
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")

    # 메모리 최적화
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    # 생성
    prompt = "A majestic lion in a savanna at sunset, photorealistic, 8k"
    negative_prompt = "blurry, low quality, distorted"

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        height=1024,
        width=1024,
    ).images[0]

    return image


def sdxl_with_refiner():
    """SDXL Base + Refiner 파이프라인"""

    # Base
    base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    ).to("cuda")

    # Refiner
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16
    ).to("cuda")

    prompt = "A cyberpunk city at night, neon lights, rain"

    # Stage 1: Base (80% denoising)
    high_noise_frac = 0.8
    base_output = base(
        prompt=prompt,
        num_inference_steps=40,
        denoising_end=high_noise_frac,
        output_type="latent"
    ).images

    # Stage 2: Refiner (20% denoising)
    refined_image = refiner(
        prompt=prompt,
        image=base_output,
        num_inference_steps=40,
        denoising_start=high_noise_frac
    ).images[0]

    return refined_image
```

### 1.3 Micro-Conditioning

```python
def sdxl_micro_conditioning():
    """SDXL Micro-Conditioning 사용"""

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    ).to("cuda")

    prompt = "A portrait of a woman"

    # 다양한 종횡비로 생성
    aspect_ratios = [
        (1024, 1024),  # 1:1
        (1152, 896),   # 4:3
        (896, 1152),   # 3:4
        (1216, 832),   # 약 3:2
        (832, 1216),   # 약 2:3
    ]

    images = []
    for width, height in aspect_ratios:
        # Micro-conditioning: 원본 해상도 힌트
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            original_size=(height, width),  # 학습 시 원본 크기
            target_size=(height, width),    # 목표 크기
            crops_coords_top_left=(0, 0),   # 크롭 좌표
        ).images[0]
        images.append(image)

    return images
```

---

## 2. ControlNet

### 2.1 개념

```
ControlNet: 조건부 제어 추가

원본 Diffusion 모델을 수정하지 않고 추가 제어 신호 주입

지원 조건:
- Canny Edge (윤곽선)
- Depth Map (깊이)
- Pose (자세)
- Segmentation (세그멘테이션)
- Normal Map (법선)
- Scribble (낙서)
- Line Art

작동 원리:
1. 조건 이미지 → 조건 인코더
2. 인코딩된 조건 → UNet에 주입 (zero convolution)
3. 원본 모델 가중치 고정, ControlNet만 학습
```

### 2.2 구현 및 사용

```python
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from controlnet_aux import CannyDetector, OpenposeDetector
import cv2
import numpy as np

class ControlNetGenerator:
    """ControlNet 기반 이미지 생성"""

    def __init__(self, base_model: str = "runwayml/stable-diffusion-v1-5"):
        self.base_model = base_model
        self.controlnets = {}
        self.detectors = {
            'canny': CannyDetector(),
            'openpose': OpenposeDetector(),
        }

    def load_controlnet(self, control_type: str):
        """ControlNet 로드"""
        controlnet_models = {
            'canny': "lllyasviel/sd-controlnet-canny",
            'depth': "lllyasviel/sd-controlnet-depth",
            'openpose': "lllyasviel/sd-controlnet-openpose",
            'scribble': "lllyasviel/sd-controlnet-scribble",
            'seg': "lllyasviel/sd-controlnet-seg",
        }

        if control_type not in self.controlnets:
            self.controlnets[control_type] = ControlNetModel.from_pretrained(
                controlnet_models[control_type],
                torch_dtype=torch.float16
            )

        return self.controlnets[control_type]

    def generate_with_canny(
        self,
        image: np.ndarray,
        prompt: str,
        low_threshold: int = 100,
        high_threshold: int = 200
    ):
        """Canny Edge 제어"""

        # Canny edge 추출
        canny_image = cv2.Canny(image, low_threshold, high_threshold)
        canny_image = np.stack([canny_image] * 3, axis=-1)

        # ControlNet 로드
        controlnet = self.load_controlnet('canny')

        # 파이프라인
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model,
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # 생성
        output = pipe(
            prompt=prompt,
            image=canny_image,
            num_inference_steps=20,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,  # 제어 강도
        ).images[0]

        return output, canny_image

    def generate_with_pose(self, image: np.ndarray, prompt: str):
        """Pose 제어"""

        # OpenPose 추출
        pose_image = self.detectors['openpose'](image)

        controlnet = self.load_controlnet('openpose')

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model,
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

        output = pipe(
            prompt=prompt,
            image=pose_image,
            num_inference_steps=20,
        ).images[0]

        return output, pose_image

    def multi_controlnet(
        self,
        image: np.ndarray,
        prompt: str,
        control_types: list = ['canny', 'depth']
    ):
        """다중 ControlNet"""

        # 여러 ControlNet 로드
        controlnets = [self.load_controlnet(ct) for ct in control_types]

        # 조건 이미지 추출
        control_images = []
        for ct in control_types:
            if ct == 'canny':
                canny = cv2.Canny(image, 100, 200)
                control_images.append(np.stack([canny]*3, axis=-1))
            elif ct == 'depth':
                # Depth 추출 (예: MiDaS)
                depth = self.extract_depth(image)
                control_images.append(depth)

        # 다중 ControlNet 파이프라인
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model,
            controlnet=controlnets,
            torch_dtype=torch.float16
        ).to("cuda")

        output = pipe(
            prompt=prompt,
            image=control_images,
            controlnet_conditioning_scale=[1.0, 0.5],  # 각각의 강도
        ).images[0]

        return output


# 사용 예시
generator = ControlNetGenerator()

# 참조 이미지에서 구도 유지하며 스타일 변경
reference_image = cv2.imread("reference.jpg")
result, canny = generator.generate_with_canny(
    reference_image,
    "A beautiful anime girl, studio ghibli style"
)
```

---

## 3. IP-Adapter (Image Prompt Adapter)

### 3.1 개념

```
IP-Adapter: 이미지를 프롬프트로 사용

텍스트 대신/함께 이미지로 스타일/내용 지시

┌────────────────────────────────────────────────────────────┐
│                    IP-Adapter 구조                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  참조 이미지 → CLIP Image Encoder → Image Features        │
│                         ↓                                  │
│                  Projection Layer (학습)                   │
│                         ↓                                  │
│              Cross-Attention에 주입                        │
│                         ↓                                  │
│  Text Prompt + Image Features → UNet → 생성 이미지        │
│                                                            │
│  용도:                                                     │
│  - 스타일 전이 (style reference)                          │
│  - 얼굴 유사성 유지 (face reference)                      │
│  - 구도/색상 참조 (composition)                           │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 3.2 사용

```python
from diffusers import StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection
import torch

def use_ip_adapter():
    """IP-Adapter 사용"""

    # 기본 파이프라인
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    # IP-Adapter 로드
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter_sd15.bin"
    )

    # 스케일 설정 (0~1, 높을수록 참조 이미지 영향 큼)
    pipe.set_ip_adapter_scale(0.6)

    # 참조 이미지
    from PIL import Image
    style_image = Image.open("style_reference.jpg")

    # 생성
    output = pipe(
        prompt="A portrait of a woman",
        ip_adapter_image=style_image,
        num_inference_steps=30,
    ).images[0]

    return output


def ip_adapter_face():
    """IP-Adapter Face: 얼굴 유사성 유지"""

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    # Face 전용 IP-Adapter
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter-full-face_sd15.bin"
    )

    pipe.set_ip_adapter_scale(0.7)

    # 참조 얼굴
    face_image = Image.open("face_reference.jpg")

    # 다양한 스타일로 생성
    prompts = [
        "A person in a business suit, professional photo",
        "A person as a superhero, comic book style",
        "A person in ancient Rome, oil painting"
    ]

    results = []
    for prompt in prompts:
        output = pipe(
            prompt=prompt,
            ip_adapter_image=face_image,
            num_inference_steps=30,
        ).images[0]
        results.append(output)

    return results


def ip_adapter_plus():
    """IP-Adapter Plus: 더 강한 이미지 조건"""

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    # Plus 버전 (더 세밀한 제어)
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter-plus_sd15.bin"
    )

    # 다중 이미지 참조
    style_images = [
        Image.open("style1.jpg"),
        Image.open("style2.jpg")
    ]

    output = pipe(
        prompt="A landscape",
        ip_adapter_image=style_images,
        num_inference_steps=30,
    ).images[0]

    return output
```

---

## 4. Latent Consistency Models (LCM)

### 4.1 개념

```
LCM: 초고속 이미지 생성

기존 Diffusion: 20-50 스텝 필요
LCM: 2-4 스텝으로 고품질 생성

작동 원리:
1. 원본 Diffusion 모델을 consistency 목표로 증류
2. 어떤 노이즈 레벨에서도 바로 깨끗한 이미지로 매핑
3. 단일 또는 소수 스텝으로 생성

장점:
- 실시간 생성 가능 (< 1초)
- 인터랙티브 응용
- 저전력 디바이스 가능
```

### 4.2 사용

```python
from diffusers import (
    DiffusionPipeline,
    LCMScheduler,
    AutoPipelineForText2Image
)

def lcm_generation():
    """LCM 빠른 생성"""

    # LCM-LoRA 사용 (기존 모델에 적용)
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    # LCM-LoRA 로드
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

    # LCM 스케줄러
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # 빠른 생성 (4 스텝!)
    image = pipe(
        prompt="A beautiful sunset over mountains",
        num_inference_steps=4,  # 매우 적은 스텝
        guidance_scale=1.5,     # LCM은 낮은 guidance 권장
    ).images[0]

    return image


def lcm_real_time():
    """실시간 이미지 생성 데모"""
    import time

    pipe = DiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    prompts = [
        "A red apple",
        "A blue car",
        "A green forest",
        "A yellow sun"
    ]

    for prompt in prompts:
        start = time.time()
        image = pipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=1.0,
            height=512,
            width=512
        ).images[0]
        elapsed = time.time() - start

        print(f"'{prompt}': {elapsed:.2f}s")


def turbo_generation():
    """SDXL-Turbo: 1-4 스텝 생성"""

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    # 단 1 스텝!
    image = pipe(
        prompt="A cinematic shot of a cat wearing a hat",
        num_inference_steps=1,
        guidance_scale=0.0,  # Turbo는 guidance 불필요
    ).images[0]

    return image
```

---

## 5. 고급 기법

### 5.1 Inpainting & Outpainting

```python
from diffusers import StableDiffusionInpaintPipeline

def inpainting_example():
    """영역 수정 (Inpainting)"""

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")

    # 원본 이미지와 마스크
    image = Image.open("original.jpg")
    mask = Image.open("mask.png")  # 흰색 = 수정할 영역

    result = pipe(
        prompt="A cat sitting on the couch",
        image=image,
        mask_image=mask,
        num_inference_steps=30,
    ).images[0]

    return result


def outpainting_example():
    """이미지 확장 (Outpainting)"""

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")

    # 원본 이미지를 캔버스에 배치
    original = Image.open("original.jpg")
    canvas_size = (1024, 1024)
    canvas = Image.new("RGB", canvas_size, (128, 128, 128))

    # 중앙에 배치
    offset = ((canvas_size[0] - original.width) // 2,
              (canvas_size[1] - original.height) // 2)
    canvas.paste(original, offset)

    # 마스크: 원본 영역 외 흰색
    mask = Image.new("L", canvas_size, 255)
    mask.paste(0, offset, (offset[0] + original.width, offset[1] + original.height))

    # 확장
    result = pipe(
        prompt="A beautiful landscape extending the scene",
        image=canvas,
        mask_image=mask,
    ).images[0]

    return result
```

### 5.2 Image-to-Image Translation

```python
from diffusers import StableDiffusionImg2ImgPipeline

def style_transfer():
    """스타일 변환"""

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    # 입력 이미지
    init_image = Image.open("photo.jpg").resize((512, 512))

    # 스타일 변환
    result = pipe(
        prompt="oil painting, impressionist style, vibrant colors",
        image=init_image,
        strength=0.75,  # 0~1, 높을수록 큰 변화
        num_inference_steps=30,
    ).images[0]

    return result
```

### 5.3 텍스트 임베딩 조작

```python
def prompt_weighting():
    """프롬프트 가중치 조절"""
    from compel import Compel

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    # 가중치 문법
    prompts = [
        "a (beautiful)++ sunset",           # ++ = 1.21배
        "a (beautiful)+++ sunset",          # +++ = 1.33배
        "a (ugly)-- sunset",                # -- = 0.83배
        "a (red:1.5) and (blue:0.5) sunset" # 명시적 가중치
    ]

    for prompt in prompts:
        conditioning = compel.build_conditioning_tensor(prompt)

        image = pipe(
            prompt_embeds=conditioning,
            num_inference_steps=30,
        ).images[0]


def prompt_blending():
    """프롬프트 블렌딩"""
    from compel import Compel

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    # 두 프롬프트 블렌딩
    prompt1 = "a photo of a cat"
    prompt2 = "a photo of a dog"

    cond1 = compel.build_conditioning_tensor(prompt1)
    cond2 = compel.build_conditioning_tensor(prompt2)

    # 50:50 블렌딩
    blended = (cond1 + cond2) / 2

    image = pipe(
        prompt_embeds=blended,
        num_inference_steps=30,
    ).images[0]

    return image
```

---

## 6. 최적화 기법

### 6.1 메모리 최적화

```python
def optimize_memory():
    """메모리 최적화 기법"""

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    )

    # 1. CPU Offload
    pipe.enable_model_cpu_offload()

    # 2. Sequential CPU Offload (더 느리지만 메모리 절약)
    # pipe.enable_sequential_cpu_offload()

    # 3. VAE Slicing (큰 이미지용)
    pipe.enable_vae_slicing()

    # 4. VAE Tiling (매우 큰 이미지용)
    pipe.enable_vae_tiling()

    # 5. Attention Slicing
    pipe.enable_attention_slicing(slice_size="auto")

    # 6. xFormers
    pipe.enable_xformers_memory_efficient_attention()

    return pipe


def batch_generation():
    """배치 생성 최적화"""

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    prompts = [
        "A red apple",
        "A blue car",
        "A green tree",
        "A yellow sun",
    ]

    # 배치 생성 (더 효율적)
    images = pipe(
        prompt=prompts,
        num_inference_steps=30,
    ).images

    return images
```

---

## 참고 자료

### 논문
- Podell et al. (2023). "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis"
- Zhang et al. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models" (ControlNet)
- Ye et al. (2023). "IP-Adapter: Text Compatible Image Prompt Adapter"
- Luo et al. (2023). "Latent Consistency Models"

### 모델
- [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [ControlNet](https://huggingface.co/lllyasviel/ControlNet)
- [IP-Adapter](https://huggingface.co/h94/IP-Adapter)
- [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdxl)

### 관련 레슨
- [../Deep_Learning/17_Diffusion_Models.md](../Deep_Learning/17_Diffusion_Models.md)

---

## 연습 문제

### 연습 문제 1: SDXL 이중 인코더 분석
SDXL은 두 개의 텍스트 인코더를 사용합니다: CLIP ViT-L/14와 OpenCLIP ViT-bigG. 이 임베딩들은 UNet에 전달되기 전에 연결(concatenate)됩니다. 이중 인코더 설계의 목적을 설명하고 각 인코더가 기여할 가능성이 높은 것을 설명하세요. 또한 SDXL의 마이크로 컨디셔닝(micro-conditioning)(original_size와 target_size 파라미터 전달)이 생성 품질을 향상시키는 이유를 설명하세요.

<details>
<summary>정답 보기</summary>

**이중 인코더의 목적**: 두 인코더는 서로 다른 아키텍처와 훈련 데이터를 가지며, 이는 상호 보완적인 강점으로 이어집니다. 임베딩을 연결하면 어느 하나의 인코더만 사용할 때보다 더 풍부하고 상세한 텍스트 표현을 제공하여, UNet이 복잡하고 미묘한 프롬프트를 더 잘 따를 수 있습니다.

**개별 기여**:
- **CLIP ViT-L/14**: 4억 개의 이미지-텍스트 쌍(OpenAI WIT 데이터셋)으로 훈련. 시각-의미론적 정렬 및 텍스트 개념과 시각적 특징 연결에 뛰어납니다. 일반적인 의미론적 이해에 강합니다("a cat", "sunset", "impressionist style").
- **OpenCLIP ViT-bigG**: LAION-5B(50억 쌍)로 훈련된 더 큰 모델(18억 파라미터 vs 3억 700만). 세밀한 디테일, 복잡한 구성 설명, 희귀 개념에서 더 뛰어납니다. 상세한 지시를 위한 더 많은 용량 제공.

함께 사용하면 일반적인 의미론적 정렬(CLIP)과 세밀한 디테일 포착(OpenCLIP bigG)을 모두 제공합니다.

**마이크로 컨디셔닝의 근거**: SD 1.5 훈련 중 이미지는 512×512로 크기 조정 및 중앙 크롭되었습니다. 이로 인해 모델이 결함을 학습했습니다: 잘린 객체, 비정상적인 구성. 모델이 이러한 왜곡을 자연스러운 이미지 분포의 일부로 "학습"한 것입니다.

SDXL의 마이크로 컨디셔닝은 원본 이미지 크기와 크롭 좌표를 추가 컨디셔닝 신호로 전달합니다. 이를 통해 모델이:
1. 고품질 원본(전체 이미지, 크롭 없음)을 훈련 시간의 왜곡과 구별할 수 있습니다.
2. 추론 시 `original_size=target_size` 설정은 "고품질, 크롭되지 않은 이미지처럼 생성하라"는 신호를 보내어 일관되게 더 나은 구성과 더 적은 결함을 산출합니다.

</details>

### 연습 문제 2: ControlNet 제로 컨볼루션(Zero Convolution)
ControlNet은 "제로 컨볼루션(zero convolution)" 레이어 — 가중치가 0으로 초기화된 컨볼루션 — 를 사용하여 제어 신호를 동결된 UNet에 주입합니다. 제로 초기화가 ControlNet의 훈련 안정성에 왜 중요한지 설명하고, 제어 인코더 가중치가 무작위로 초기화된다면 어떤 일이 발생할지 설명하세요.

```python
# ControlNet 주입 메커니즘 (단순화)
class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        # 중요: 가중치와 편향을 0으로 초기화
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

# 훈련 시작 시:
# zero_conv(any_input) = 0
# UNet output = original_output + 0 = original_output

# 이것이 왜 중요한가?
```

<details>
<summary>정답 보기</summary>

**제로 초기화가 중요한 이유**:

ControlNet 훈련 시작 시, 제로 컨볼루션은 제어 신호가 정확히 0임을 보장합니다: `zero_conv(condition_features) = 0`. 이는 다음을 의미합니다:
- **UNet 출력 = 원래 동결된 UNet 출력 + 0 = 원래 출력**
- 사전 훈련된 UNet은 ControlNet 추가 이전과 동일하게 동작합니다.
- 모델은 완벽한 기준선(사전 훈련된 확산 모델)에서 시작하며, 무작위로 손상된 기준선이 아닙니다.

훈련이 진행되면서 제로 컨볼루션 가중치는 적절한 제어 신호를 주입하도록 점진적으로 학습하여 0에서 의미 있는 값으로 성장합니다.

**무작위 초기화 시 발생하는 일**:
- 제어 인코더가 UNet의 모든 레이어에 무작위 노이즈를 주입합니다.
- UNet은 모든 어텐션(attention) 레이어에서 손상된 중간 활성화를 받습니다.
- 좋은 이미지를 생성하는 사전 훈련된 가중치가 이 무작위 주입에 압도되어 초기 불안정성이 심각해집니다.
- 최적화 경관이 매우 혼란스러워집니다 — 손상된 UNet 출력의 그래디언트(gradient)가 올바른 제어 동작 학습에 의미가 없습니다.
- 훈련이 발산하거나 초기 손상을 극복하기 위해 매우 느린 워밍업 스케줄이 필요할 수 있습니다.

제로 초기화는 파인 튜닝(fine-tuning)을 모듈식으로 만드는 우아한 해법입니다: 원래 모델이 완벽한 시작점으로 보존되고, 새로운 제어 경로가 점진적으로 추가됩니다.

</details>

### 연습 문제 3: LCM vs. 표준 확산 속도/품질 트레이드오프
LCM은 표준 확산 모델의 20-50 스텝 대신 2-4 스텝으로 고품질 이미지를 생성합니다. 다음 비교를 분석하세요:

```python
# 표준 DDIM 샘플링 (50 스텝)
standard_image = pipe(
    prompt="A photorealistic portrait",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

# LCM 샘플링 (4 스텝)
lcm_image = pipe(
    prompt="A photorealistic portrait",
    num_inference_steps=4,
    guidance_scale=1.5  # 참고: 훨씬 낮은 가이던스 스케일
).images[0]
```

다음 질문에 답하세요:
- A) LCM이 표준 확산보다 훨씬 낮은 guidance_scale을 사용하는 이유는 무엇인가요?
- B) 두 출력 간에 일반적으로 어떤 품질 차이를 관찰할 수 있나요?
- C) 프로덕션 시스템에서 LCM과 표준 확산을 각각 언제 사용해야 하나요?

<details>
<summary>정답 보기</summary>

**A) LCM에서 낮은 guidance_scale을 사용하는 이유**:
분류기 없는 가이던스(Classifier-Free Guidance, CFG)는 무조건부 노이즈에 비해 텍스트 조건부 신호를 증폭합니다. 높은 가이던스 스케일(7.5)에서 모델은 프롬프트를 강하게 따르지만 결함과 비현실적인 디테일도 증폭합니다. 표준 확산에서 50 스텝을 거치면 이 결함들이 많은 노이즈 제거 스텝에 걸쳐 부드러워집니다.

LCM은 어떤 노이즈 레벨에서도 매우 적은 스텝으로 깨끗한 이미지에 직접 매핑하는 일관성 목표(consistency objective)를 사용합니다. 이는 각 스텝이 잠재 공간(latent space)에서 훨씬 큰 "점프"를 한다는 것을 의미합니다. 높은 guidance_scale에서 이 큰 점프는 과수정이 되어 — 모델이 포화되고 결함이 많은 출력을 향해 지나치게 이동합니다. 낮은 guidance_scale(1.0-2.0)은 점프를 제어 가능하게 유지하며 4 스텝에서 더 자연스러운 결과를 생성합니다.

**B) 일반적인 품질 차이**:
- **표준 확산 (50 스텝)**: 텍스처(머리카락, 직물, 미세한 패턴)에서 더 많은 디테일, 더 나은 색상 그래디언트, 더 사실적인 피부 톤, 덜한 과채도.
- **LCM (4 스텝)**: 약간 더 부드러운 디테일, 하이라이트에서 약간의 과채도 가능성, 때로는 미세한 텍스처 디테일이 적음. 그러나 많은 피사체에서 차이는 놀라울 정도로 작습니다 — LCM은 매우 좋습니다.
- 추상적/예술적 프롬프트의 경우 LCM 품질은 표준 확산과 거의 동일합니다. 세밀한 디테일이 있는 매우 사실적인 인물 사진의 경우 표준 확산이 눈에 띄게 더 낫습니다.

**C) 프로덕션 사용 사례 선택**:
- **LCM 사용 시**: 실시간 상호작용이 필요한 경우(사용자 대면 애플리케이션의 1초 미만 생성), 제한된 컴퓨팅을 가진 모바일/엣지 배포, 많은 프롬프트를 빠르게 A/B 테스트, 썸네일 또는 미리보기 생성.
- **표준 확산 사용 시**: 최대 품질이 우선순위인 경우(제품 사진, 전문 예술 출력), 미세한 텍스처 디테일이 중요한 경우, 출력을 대형으로 표시하거나 인쇄할 경우, 지연 시간이 중요하지 않은 오프라인 배치 처리.

실용적인 전략: 반복 작업과 미리보기에 LCM을 사용하고, 최종 고품질 렌더링에 표준 확산으로 전환합니다.

</details>

### 연습 문제 4: 다중 기법 파이프라인 설계
ControlNet + IP-Adapter + LCM-LoRA를 함께 사용하여 다음 태스크를 해결하는 완전한 이미지 생성 파이프라인을 설계하세요:

**태스크**: 사람의 참조 사진과 장면의 연필 스케치가 주어졌을 때, 스케치가 묘사하는 장면에 그 사람이 배치된 이미지를 일관된 예술 스타일로 생성합니다.

각 컴포넌트의 역할을 설명하고 의사 코드(pseudo-code)로 파이프라인을 스케치하세요.

<details>
<summary>정답 보기</summary>

**각 컴포넌트의 역할**:
- **ControlNet (Scribble/Canny)**: 연필 스케치를 공간적 제어 신호로 사용하여 스케치에서 장면 구성(객체 배치, 일반적인 레이아웃)을 보존합니다.
- **IP-Adapter (Face 변형)**: 참조 사진을 사용하여 생성된 이미지에서 사람의 정체성(얼굴 특징, 전반적인 외모)을 보존합니다.
- **LCM-LoRA**: 생성 스텝을 30에서 4로 줄여 프롬프트나 스케치를 정제할 때 실시간 반복을 가능하게 합니다.

**파이프라인 의사 코드**:

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
import torch

# 1. 컴포넌트 로드
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# 얼굴 정체성 보존을 위한 IP-Adapter 로드
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                     weight_name="ip-adapter-full-face_sd15.bin")
pipe.set_ip_adapter_scale(0.7)  # 강한 정체성 보존

# 빠른 생성을 위한 LCM-LoRA 로드
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# 2. 입력 준비
reference_photo = Image.open("person_reference.jpg")     # 정체성 소스
pencil_sketch = Image.open("scene_sketch.jpg")           # 공간적 레이아웃 소스

# 3. 생성
result = pipe(
    prompt="A professional portrait of a person in [scene description], artistic style",
    negative_prompt="blurry, distorted face, multiple people",
    image=pencil_sketch,               # ControlNet: 장면 레이아웃
    ip_adapter_image=reference_photo,  # IP-Adapter: 사람 정체성
    num_inference_steps=4,             # LCM: 빠른 생성
    guidance_scale=1.5,               # LCM에 적합한 스케일
    controlnet_conditioning_scale=0.8  # 장면 vs. 정체성 균형
).images[0]
```

**주요 설계 결정사항**:
- `ip_adapter_scale=0.7`: 강한 정체성(얼굴) 보존 — 사람의 얼굴을 유지할 만큼 높지만, 스타일 변형을 허용할 만큼 낮음.
- `controlnet_conditioning_scale=0.8`: 강한 장면 구조 제어, 1.0보다 약간 낮아 스케치 레이아웃의 자연스러운 적응 허용.
- `guidance_scale=1.5`: LCM 호환 낮은 가이던스.
- 텍스트 프롬프트는 여전히 전체적인 스타일을 안내하고 스케치나 정체성이 제한하지 않는 디테일을 채웁니다.

</details>
