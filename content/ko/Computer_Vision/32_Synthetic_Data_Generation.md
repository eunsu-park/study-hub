[이전: Optical Flow](./31_Optical_Flow.md)

---

# 32. 합성 데이터 생성

## 학습 목표

이 수업을 완료하면 다음을 수행할 수 있습니다:

1. 컴퓨터 비전 학습 파이프라인에서 합성 데이터의 역할 설명
2. 강건한 모델 학습을 위한 도메인 랜덤화 구현
3. 탐지 및 세그멘테이션을 위한 절차적 데이터 생성 파이프라인 구축
4. 제어된 데이터 증강을 위해 확산 모델 사용
5. 합성 데이터와 실제 데이터 간의 도메인 갭 분석

---

## 목차

1. [왜 합성 데이터인가?](#1-왜-합성-데이터인가)
2. [도메인 랜덤화](#2-도메인-랜덤화)
3. [절차적 데이터 생성](#3-절차적-데이터-생성)
4. [3D 렌더링 파이프라인](#4-3d-렌더링-파이프라인)
5. [데이터 증강을 위한 확산 모델](#5-데이터-증강을-위한-확산-모델)
6. [도메인 적응](#6-도메인-적응)
7. [실용적 합성 데이터 파이프라인](#7-실용적-합성-데이터-파이프라인)
8. [연습문제](#8-연습문제)

---

## 1. 왜 합성 데이터인가?

### 이론: 합성 데이터가 도움이 되는 이유

**주석 비용**: 단일 인간 주석된 COCO 인스턴스 세그멘테이션 마스크는 30+분 걸림; 합성 마스크는 무료이고 픽셀 완벽. 주석이 병목인 작업(세그멘테이션, 깊이, optical flow ground truth)에는 합성 데이터가 가능하게 함.

**드문 사건 커버리지**: "어린이가 길로 뛰어드는" 또는 "초기 단계 종양"의 예제 수집은 윤리적, 실용적으로 불가능. 합성 생성이 이를 요청 시 만들 수 있음.

**픽셀 완벽 레이블**: 인간 주석가는 잡음 도입; 렌더링은 정확히 올바른 ground truth 생성.

**프라이버시**: 얼굴, 의료 기록, 다른 사적 콘텐츠를 합성 등가물로 대체 가능.

비용: 합성 데이터로만 훈련하고 실제 데이터에 배포하면 도메인 갭 때문에 보통 실패.

### 이론: 도메인 갭

합성 이미지 A로 훈련된 모델이 실제 이미지 B에서 평가하면 기대보다 나쁨. 이유:

- **텍스처 사실성**: 렌더링된 표면이 "너무 깨끗"해 보임 — 긁힘, 먼지, 풍화 없음.
- **조명과 그림자**: 시뮬레이션된 조명이 종종 단순화됨(전역 조명 없음, caustic 없음).
- **센서 효과**: 실제 이미지는 센서 잡음, 모션 블러, JPEG 아티팩트 가짐; 합성은 안 가짐.
- **분포 변화**: 합성 장면의 객체 배치, 포즈, 공동 발생이 실제 장면 통계와 일치하지 않음.
- **배경 변동성**: 실제 배경은 어수선하고 다양; 합성 배경은 종종 구성되고 제한됨.

이들 각각이 모델이 학습할 수 있는 가짜 단서 생성("텍스처가 이렇게 깨끗하면 합성 의자"). 실제 이미지의 테스트 시점에 모델은 이 의지를 잃음.

### 1.1 데이터 문제

```
실제 데이터의 과제:
  - 수집 및 어노테이션 비용이 높음
  - 개인정보 보호 우려 (얼굴, 의료 데이터)
  - 희귀 이벤트가 과소 대표됨
  - 어노테이션이 불일치하거나 잘못될 수 있음
  - 다양성 제한 (날씨, 조명, 시점)

합성 데이터의 장점:
  + 무료, 무제한, 완벽한 어노테이션
  + 조건의 완전한 제어 (조명, 날씨, 자세)
  + 희귀/위험한 시나리오를 안전하게 생성 가능
  + 개인정보 문제 없음
  + 재현 가능

합성 데이터의 과제:
  - 도메인 갭: 합성 ≠ 실제
  - 렌더링 품질이 전이에 영향
  - 충분한 다양성을 포함해야 함
  - 합성 아티팩트에 대한 "지름길 학습" 위험
```

### 1.2 성공 사례

```
합성 데이터가 효과적인 곳:

자율 주행:
  - NVIDIA DRIVE Sim, CARLA 시뮬레이터
  - 합성 데이터 사전 학습 + 실제 데이터 파인튜닝
  - 희귀 시나리오 (사고)에 특히 유용

로봇공학:
  - Sim-to-real 전이 (RL의 23강)
  - 도메인 랜덤화로 정책을 강건하게 만듦

의료 영상:
  - 희귀 병변 생성
  - 소규모 데이터셋 증강
  - 개인정보 보호를 위한 학습

산업 검사:
  - 프로그래밍 방식으로 결함 이미지 생성
  - 희귀 결함 유형이 잘 대표됨

얼굴 인식:
  - 학습을 위한 합성 얼굴 (개인정보 보호)
  - 인구통계 및 자세 제어
```

---

## 2. 도메인 랜덤화

### 이론: 도메인 랜덤화

도메인 랜덤화(Tobin 등, 2017)는 **반직관적 해결**: 합성 데이터를 더 사실적으로 만들려는 대신, 의도적으로 **여러 면에서 덜 사실적으로** 만들기. 변화시킴:

- **조명**: 광원의 무작위 방향, 강도, 색상.
- **텍스처**: 그럴듯한 텍스처를 무작위 패턴, 색상, 비현실적 재료로 대체.
- **객체 배치**: 위치, 방향, 스케일 무작위화.
- **카메라 포즈**: 합리적 범위 내에서 시점 무작위화.
- **방해 요소**: 장면에 무작위 추가 객체.
- **배경**: 무작위 이미지, 무작위 잡음, 무작위 어수선함.

통찰: 모델이 훈련 중 충분한 변화를 보면, **변화하는 단서**(텍스처, 조명, 배경)를 **무시**하고 **불변 단서**(객체 모양, 기하)에 집중하도록 학습. 실제 데이터에 배포될 때 — 단지 이 다양한 분포의 한 특정 점 — 모델이 실세계 외관을 자신의 훈련 경험 내 또 다른 "도메인"으로 취급.

이것이 도메인 랜덤화 합성 데이터가 종종 신중하게 photorealistic한 합성 데이터보다 실제로 더 잘 전이되는 이유: photorealism은 모델이 과적합하는 단일 훈련 분포 생성; 랜덤화는 모델이 작업 필수 특징을 학습하도록 강제.

### 2.1 랜덤화 전략

```python
import cv2
import numpy as np
from PIL import Image


class DomainRandomizer:
    """합성 이미지에 도메인 랜덤화 적용."""

    def __init__(self):
        self.randomizers = [
            self.random_lighting,
            self.random_blur,
            self.random_noise,
            self.random_color_jitter,
            self.random_texture_overlay,
        ]

    def apply(self, image, n_random=3):
        """증강의 랜덤 서브셋 적용."""
        selected = np.random.choice(
            self.randomizers, min(n_random, len(self.randomizers)), replace=False
        )
        for fn in selected:
            image = fn(image)
        return image

    def random_lighting(self, image):
        """다양한 조명 조건 시뮬레이션."""
        brightness = np.random.uniform(0.5, 1.5)
        contrast = np.random.uniform(0.7, 1.3)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        mean = image.mean()
        image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
        return image

    def random_blur(self, image):
        """랜덤 가우시안 블러."""
        ksize = np.random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def random_noise(self, image):
        """랜덤 가우시안 노이즈 추가."""
        sigma = np.random.uniform(5, 25)
        noise = np.random.normal(0, sigma, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)

    def random_color_jitter(self, image):
        """랜덤 색상, 채도, 명도 변화."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] += np.random.uniform(-10, 10)
        hsv[:, :, 1] *= np.random.uniform(0.7, 1.3)
        hsv[:, :, 2] *= np.random.uniform(0.7, 1.3)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def random_texture_overlay(self, image):
        """배경 변화를 위한 랜덤 텍스처 오버레이."""
        H, W = image.shape[:2]
        # 랜덤 패턴 생성
        pattern = np.random.randint(0, 50, (H, W, 3), dtype=np.uint8)
        alpha = np.random.uniform(0.0, 0.15)
        return cv2.addWeighted(image, 1 - alpha, pattern, alpha, 0)
```

---

## 3. 절차적 데이터 생성

### 이론: 절차적 생성 파이프라인

합성 데이터셋 구축은 일반적으로 다음 포함:

1. **3D 자산 라이브러리**: 검출/세그멘트하려는 객체의 모델(의자, 자동차, 사람, ...). 출처: ShapeNet, Objaverse, Sketchfab, 커스텀 CAD.
2. **장면 컴포저**: 3D 환경에 객체의 프로그래밍 방식 배치. 도구: Blender Python API, Unity, Unreal, NVIDIA Isaac Sim.
3. **도메인 랜덤화 컨트롤러**: 조명, 재료, 카메라, 방해 요소의 이미지별 랜덤화.
4. **렌더러**: Path-traced 또는 rasterized 렌더링. Photorealism은 path tracing(Cycles, Mitsuba); 속도는 rasterization.
5. **주석 추출기**: 렌더링이 자동으로 ground truth(깊이, 마스크, 법선, optical flow) 생성 — 인간 주석 불필요.

전형적 파이프라인이 GPU 렌더링 몇 시간에 시나리오당 10K-1M 주석된 이미지 생성.

### 3.1 객체 배치

```python
class SyntheticSceneGenerator:
    """어노테이션이 포함된 합성 장면 생성."""

    def __init__(self, background_dir, object_dir, image_size=(640, 480)):
        self.backgrounds = self._load_images(background_dir)
        self.objects = self._load_images_with_masks(object_dir)
        self.image_size = image_size

    def _load_images(self, directory):
        import glob
        paths = glob.glob(f"{directory}/*.jpg") + glob.glob(f"{directory}/*.png")
        return [cv2.imread(p) for p in paths]

    def _load_images_with_masks(self, directory):
        # 알파 채널이 있는 객체 이미지 로드
        import glob
        paths = glob.glob(f"{directory}/*.png")
        objects = []
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img.shape[2] == 4:  # 알파 있음
                objects.append(img)
        return objects

    def generate_scene(self, n_objects=5):
        """어노테이션이 포함된 하나의 합성 장면 생성."""
        W, H = self.image_size

        # 랜덤 배경
        bg_idx = np.random.randint(len(self.backgrounds))
        scene = cv2.resize(self.backgrounds[bg_idx], (W, H))

        annotations = []
        instance_mask = np.zeros((H, W), dtype=np.int32)

        for i in range(n_objects):
            # 랜덤 객체
            obj_idx = np.random.randint(len(self.objects))
            obj = self.objects[obj_idx].copy()

            # 랜덤 스케일
            scale = np.random.uniform(0.3, 1.5)
            obj_h, obj_w = int(obj.shape[0] * scale), int(obj.shape[1] * scale)
            obj = cv2.resize(obj, (obj_w, obj_h))

            # 랜덤 위치
            x = np.random.randint(0, max(1, W - obj_w))
            y = np.random.randint(0, max(1, H - obj_h))

            # 장면에 객체 붙여넣기
            if obj.shape[2] == 4:
                alpha = obj[:, :, 3:] / 255.0
                rgb = obj[:, :, :3]

                y_end = min(y + obj_h, H)
                x_end = min(x + obj_w, W)
                obj_h_clip = y_end - y
                obj_w_clip = x_end - x

                scene[y:y_end, x:x_end] = (
                    scene[y:y_end, x:x_end] * (1 - alpha[:obj_h_clip, :obj_w_clip]) +
                    rgb[:obj_h_clip, :obj_w_clip] * alpha[:obj_h_clip, :obj_w_clip]
                ).astype(np.uint8)

                # 인스턴스 마스크
                mask_region = (alpha[:obj_h_clip, :obj_w_clip, 0] > 0.5)
                instance_mask[y:y_end, x:x_end][mask_region] = i + 1

            # 바운딩 박스 어노테이션
            annotations.append({
                'class_id': obj_idx,
                'bbox': [x, y, x_end, y_end],
                'instance_id': i + 1,
            })

        return scene, annotations, instance_mask

    def generate_dataset(self, n_images=1000, output_dir='synthetic_data'):
        """전체 합성 데이터셋 생성."""
        import os
        import json

        os.makedirs(f"{output_dir}/images", exist_ok=True)
        all_annotations = []

        for img_id in range(n_images):
            n_objects = np.random.randint(1, 8)
            scene, annotations, mask = self.generate_scene(n_objects)

            # 도메인 랜덤화 적용
            randomizer = DomainRandomizer()
            scene = randomizer.apply(scene)

            # 저장
            img_path = f"{output_dir}/images/{img_id:06d}.jpg"
            cv2.imwrite(img_path, scene)

            mask_path = f"{output_dir}/images/{img_id:06d}_mask.png"
            cv2.imwrite(mask_path, mask.astype(np.uint16))

            for ann in annotations:
                ann['image_id'] = img_id
            all_annotations.extend(annotations)

            if (img_id + 1) % 100 == 0:
                print(f"{img_id + 1}/{n_images}개 이미지 생성 완료")

        # 어노테이션 저장
        with open(f"{output_dir}/annotations.json", 'w') as f:
            json.dump(all_annotations, f)

        print(f"데이터셋이 {output_dir}에 저장됨: {n_images}개 이미지")
```

---

## 4. 3D 렌더링 파이프라인

### 4.1 합성 데이터를 위한 Blender

```python
# 합성 데이터 생성을 위한 Blender Python API
# Blender 내에서 실행: blender --background --python generate.py

"""
import bpy
import numpy as np
import os

def setup_scene():
    # 장면 초기화
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 카메라 추가
    bpy.ops.object.camera_add(location=(0, -5, 3))
    cam = bpy.context.object
    cam.rotation_euler = (1.1, 0, 0)
    bpy.context.scene.camera = cam

    # 조명 추가
    bpy.ops.object.light_add(type='SUN', location=(3, -3, 5))
    light = bpy.context.object
    light.data.energy = 3

    # 도메인 랜덤화를 위한 랜덤 조명 위치
    light.location.x = np.random.uniform(-5, 5)
    light.location.y = np.random.uniform(-5, 5)

def add_random_objects(n_objects=5):
    objects = []
    for i in range(n_objects):
        # 랜덤 프리미티브
        shape = np.random.choice(['cube', 'sphere', 'cylinder'])
        if shape == 'cube':
            bpy.ops.mesh.primitive_cube_add()
        elif shape == 'sphere':
            bpy.ops.mesh.primitive_uv_sphere_add()
        else:
            bpy.ops.mesh.primitive_cylinder_add()

        obj = bpy.context.object

        # 랜덤 위치
        obj.location = (
            np.random.uniform(-2, 2),
            np.random.uniform(-2, 2),
            np.random.uniform(0, 2),
        )

        # 랜덤 스케일
        s = np.random.uniform(0.2, 0.8)
        obj.scale = (s, s, s)

        # 랜덤 색상
        mat = bpy.data.materials.new(name=f"mat_{i}")
        mat.diffuse_color = (*np.random.uniform(0, 1, 3), 1)
        obj.data.materials.append(mat)

        objects.append(obj)

    return objects

def render_and_save(output_path, resolution=(640, 480)):
    scene = bpy.context.scene
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

# 사용법:
# setup_scene()
# objects = add_random_objects(5)
# render_and_save('/tmp/synthetic_001.png')
"""
```

---

## 5. 데이터 증강을 위한 확산 모델

### 이론: 데이터 증강을 위한 생성 모델

최근 변화: 3D 장면을 만들고 렌더링하는 대신, **생성 모델**을 사용해 합성 이미지를 직접 생성:

#### E.1 Diffusion 기반 증강

Stable Diffusion, ControlNet, 유사 모델이 가능:

- 구조를 보존하면서 기존 이미지의 변형 생성("다른 조명의 같은 자동차").
- Prompt된 콘텐츠로 특정 영역 inpaint("여기 보행자 추가").
- Layout/skeleton/depth 조건으로 생성, 주어진 ground truth와 정렬된 새 이미지 생성.

장점: 3D 모델이나 렌더링 파이프라인 없이 photorealistic 합성. 단점: 덜 정확한 ground truth(생성된 마스크가 약간 틀릴 수 있음)과 3D 기반 랜덤화보다 덜 다양한 변화.

#### E.2 GAN 기반 증강

이전 시대: GAN(StyleGAN, BigGAN)이 잡음에서 사실적 이미지 생성. 얼굴/장면 증강에 사용. 일반 사용에는 대부분 diffusion 모델로 대체됐지만, 특정 도메인(GAN이 잘 튜닝된 의료 영상)에는 여전히 사용.

### 5.1 제어된 생성

```
데이터 증강을 위한 확산 모델 (Stable Diffusion) 사용:

접근법:
1. Text-to-Image 생성:
   "밤에 비 오는 도로 위의 빨간 자동차"
   → 다양한 학습 이미지 생성

2. ControlNet을 사용한 Image-to-Image:
   입력: 세그멘테이션 맵 + 텍스트 프롬프트
   출력: 레이아웃에 맞는 사실적 이미지

3. 증강을 위한 인페인팅:
   실제 이미지의 일부를 마스킹
   다른 객체/텍스처로 채우기

4. 스타일 전이:
   합성 렌더를 사실적 스타일로 변환
   도메인 갭을 크게 줄임
```

### 5.2 합성 데이터를 위한 ControlNet

```python
# 데이터 증강을 위한 확산 모델 사용의 개념적 예제
# 필요: pip install diffusers transformers

def generate_with_controlnet(segmentation_map, prompt, n_images=5):
    """
    세그멘테이션 맵에서 사실적 이미지 생성.
    시맨틱 세그멘테이션에 조건화된 ControlNet 사용.
    """
    # from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    # controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg")
    # pipe = StableDiffusionControlNetPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5", controlnet=controlnet
    # )

    images = []
    for i in range(n_images):
        # 동일 레이아웃, 다른 외형
        # image = pipe(
        #     prompt=prompt,
        #     image=segmentation_map,
        #     num_inference_steps=20,
        # ).images[0]
        # images.append(image)
        pass

    return images

# 예시:
# seg_map = load_segmentation("city_layout.png")
# images = generate_with_controlnet(
#     seg_map,
#     "urban street scene, sunny day, high resolution photo",
#     n_images=10
# )
# 각 이미지는 동일한 레이아웃이지만 다른 시각적 외형
```

---

## 6. 도메인 적응

### 이론: 도메인 적응: 갭 좁히기

합성만으로 충분하지 않을 때, **도메인 적응** 기법이 합성과 실제 분포를 명시적으로 정렬 시도:

- **Domain-adversarial 훈련**: 작업용과 입력이 합성인지 실제인지 예측용 두 헤드를 가진 네트워크 훈련. 특징이 구별 불가능해지도록 적대적으로 훈련. 작업 헤드가 그 다음 합성에서 실제로 전이.
- **CycleGAN 스타일 변환**: 합성 → 사실적 스타일 변환을 위한 GAN 훈련, 그 다음 변환된 이미지로 후단 모델 훈련.
- **Few-shot 파인튜닝**: 합성 데이터로 사전 훈련, 그 다음 작은 양의 실제 데이터로 파인튜닝. 종종 가장 효과적인 실용 접근.
- **특징 공간 정렬**: 합성과 실제 도메인 간 특징 통계(BatchNorm running average, 평균과 분산) 매치.

오늘날 표준 레시피: 도메인 랜덤화 합성 데이터 + few-shot 실제 파인튜닝 + 광범위한 증강. 각 구성 요소가 갭의 다른 측면 다룸.

### 6.1 도메인 갭 해소

```
합성에서 실제로의 도메인 갭을 줄이는 기법:

1. Feature 수준 적응:
   합성과 실제 특징을 구분할 수 없도록 판별기 학습
   (적대적 도메인 적응)

2. Image 수준 적응:
   합성 이미지를 더 사실적으로 변환
   (CycleGAN, 스타일 전이)

3. 자기 학습:
   합성으로 학습 → 실제 데이터에 의사 레이블 → 결합하여 재학습

4. 파인튜닝:
   합성으로 사전 학습, 소규모 실제 데이터셋으로 파인튜닝
   가장 실용적인 접근법

5. 도메인 랜덤화:
   합성 데이터를 매우 다양하게 만들어 실제 데이터가 분포 내에 들어오도록
```

---

## 7. 실용적 합성 데이터 파이프라인

### 7.1 엔드-투-엔드 파이프라인

```python
def synthetic_data_pipeline(real_data_path, output_dir, n_synthetic=10000):
    """
    완전한 합성 데이터 파이프라인:
    1. 실제 데이터 분포 분석
    2. 통계에 맞는 합성 데이터 생성
    3. 도메인 랜덤화 적용
    4. 실제 데이터와 혼합
    5. 학습 및 평가
    """
    # 1단계: 실제 데이터 분석
    real_stats = analyze_dataset(real_data_path)
    print(f"실제 데이터: {real_stats['n_images']}개 이미지, "
          f"{real_stats['n_classes']}개 클래스")

    # 2단계: 합성 데이터 생성
    generator = SyntheticSceneGenerator(
        background_dir=f"{real_data_path}/backgrounds",
        object_dir=f"{real_data_path}/objects",
    )
    generator.generate_dataset(n_images=n_synthetic, output_dir=output_dir)

    # 3단계: 도메인 랜덤화 (생성기에 이미 적용됨)

    # 4단계: 데이터셋 혼합
    mixed_dataset = create_mixed_dataset(
        real_path=real_data_path,
        synthetic_path=output_dir,
        synthetic_ratio=0.5,  # 50% 합성
    )

    # 5단계: 학습
    model = train_model(mixed_dataset)

    # 6단계: 실제 데이터만으로 평가
    results = evaluate_model(model, f"{real_data_path}/test")
    print(f"실제 테스트 세트에서 mAP: {results['mAP']:.4f}")

    return model, results
```

---

## 8. 연습문제

### 연습문제 1: 합성 객체 탐지 데이터셋

객체 탐지를 위한 합성 데이터셋 생성:
1. 투명도가 있는 10개의 객체 이미지 수집 (알파 채널 PNG)
2. 20개의 배경 이미지 수집
3. 랜덤 배치, 스케일링, 회전 구현
4. COCO 형식 어노테이션으로 5000개 학습 이미지 생성
5. YOLOv8 학습 및 실제 이미지에서 평가

### 연습문제 2: 도메인 랜덤화 연구

도메인 랜덤화의 효과 연구:
1. 랜덤화 없이 합성 데이터 생성
2. 증가하는 랜덤화 수준으로 생성 (1-5개 증강)
3. 각 변형에서 동일 모델 학습
4. 모든 모델을 동일한 실제 테스트 세트에서 테스트
5. 플롯: 실제 데이터 성능 vs 랜덤화 수준

### 연습문제 3: 확산 기반 증강

데이터 증강을 위한 확산 모델 사용:
1. 소규모 실제 데이터셋 (100개 이미지)으로 시작
2. img2img 확산을 사용하여 변형 생성
3. 원본 + 생성된 이미지를 결합하여 학습
4. 비교: 원본만 vs 증강된 학습
5. 측정: 증강량 증가에 따른 수확 체감

### 연습문제 4: 도메인 갭 분석

도메인 갭 정량화 및 감소:
1. 합성 데이터만으로 모델 학습, 실제로 테스트
2. 합성 vs 실제의 특징 분포 시각화 (t-SNE)
3. CycleGAN을 적용하여 합성 → 사실적 스타일 변환
4. 스타일 변환된 합성 데이터로 재학습
5. 각 단계에서 도메인 갭 감소 측정

### 연습문제 5: 완전한 합성 파이프라인

프로덕션급 합성 데이터 파이프라인 구축:
1. 대상 응용 정의 (예: 소매점 제품 탐지)
2. 대상 객체의 3D 모델 생성
3. Blender/Unity로 사실적 렌더 생성
4. 포괄적 도메인 랜덤화 적용
5. 학습, 평가, 파이프라인 반복적 개선

---

*32강 끝 - Computer Vision 과정을 완료하셨습니다!*
