# 15. Image Generation 심화

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
