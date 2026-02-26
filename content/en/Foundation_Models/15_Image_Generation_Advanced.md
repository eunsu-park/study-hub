# 15. Advanced Image Generation

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the architectural improvements in SDXL over Stable Diffusion 1.5 and describe how dual text encoders enhance image generation quality
2. Implement ControlNet to achieve spatial control over image generation using depth maps, edge maps, and pose skeletons
3. Apply IP-Adapter to transfer image style and identity while maintaining prompt-guided content generation
4. Compare Latent Consistency Models (LCM) with standard diffusion sampling and explain the speed-quality trade-offs
5. Design end-to-end image generation pipelines that combine multiple conditioning techniques for fine-grained control

---

## Overview

This lesson covers the latest image generation techniques after Stable Diffusion. We explore practical techniques including SDXL, ControlNet, IP-Adapter, and Latent Consistency Models.

---

## 1. SDXL (Stable Diffusion XL)

### 1.1 Architecture Improvements

```
┌──────────────────────────────────────────────────────────────────┐
│                    SDXL vs SD 1.5 Comparison                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SD 1.5:                                                         │
│  - UNet: 860M params                                            │
│  - Text Encoder: CLIP ViT-L/14 (77 tokens)                      │
│  - Resolution: 512×512                                          │
│  - VAE: 4× downscale                                            │
│                                                                  │
│  SDXL:                                                           │
│  - UNet: 2.6B params (3x increase)                              │
│  - Text Encoder: CLIP ViT-L + OpenCLIP ViT-bigG (dual)          │
│  - Resolution: 1024×1024                                        │
│  - VAE: Improved VAE-FT                                         │
│  - Refiner model (optional)                                      │
│                                                                  │
│  Key Improvements:                                               │
│  - Richer text understanding (dual encoder)                     │
│  - High resolution generation (4x pixels)                       │
│  - Micro-conditioning (size, aspect ratio)                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Using SDXL

```python
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

def sdxl_generation():
    """SDXL image generation"""

    # Load base model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")

    # Memory optimization
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    # Generation
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
    """SDXL Base + Refiner pipeline"""

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
    """Using SDXL Micro-Conditioning"""

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    ).to("cuda")

    prompt = "A portrait of a woman"

    # Generate with various aspect ratios
    aspect_ratios = [
        (1024, 1024),  # 1:1
        (1152, 896),   # 4:3
        (896, 1152),   # 3:4
        (1216, 832),   # ~3:2
        (832, 1216),   # ~2:3
    ]

    images = []
    for width, height in aspect_ratios:
        # Micro-conditioning: original resolution hint
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            original_size=(height, width),  # Original size during training
            target_size=(height, width),    # Target size
            crops_coords_top_left=(0, 0),   # Crop coordinates
        ).images[0]
        images.append(image)

    return images
```

---

## 2. ControlNet

### 2.1 Concept

```
ControlNet: Adding Conditional Control

Inject additional control signals without modifying the original Diffusion model

Supported conditions:
- Canny Edge (edges)
- Depth Map (depth)
- Pose (pose)
- Segmentation (segmentation)
- Normal Map (normals)
- Scribble (scribble)
- Line Art

How it works:
1. Condition image → Condition encoder
2. Encoded condition → Inject into UNet (zero convolution)
3. Freeze original model weights, train only ControlNet
```

### 2.2 Implementation and Usage

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
    """ControlNet-based image generation"""

    def __init__(self, base_model: str = "runwayml/stable-diffusion-v1-5"):
        self.base_model = base_model
        self.controlnets = {}
        self.detectors = {
            'canny': CannyDetector(),
            'openpose': OpenposeDetector(),
        }

    def load_controlnet(self, control_type: str):
        """Load ControlNet"""
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
        """Canny Edge control"""

        # Extract Canny edges
        canny_image = cv2.Canny(image, low_threshold, high_threshold)
        canny_image = np.stack([canny_image] * 3, axis=-1)

        # Load ControlNet
        controlnet = self.load_controlnet('canny')

        # Pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model,
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # Generate
        output = pipe(
            prompt=prompt,
            image=canny_image,
            num_inference_steps=20,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,  # Control strength
        ).images[0]

        return output, canny_image

    def generate_with_pose(self, image: np.ndarray, prompt: str):
        """Pose control"""

        # Extract OpenPose
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
        """Multiple ControlNets"""

        # Load multiple ControlNets
        controlnets = [self.load_controlnet(ct) for ct in control_types]

        # Extract condition images
        control_images = []
        for ct in control_types:
            if ct == 'canny':
                canny = cv2.Canny(image, 100, 200)
                control_images.append(np.stack([canny]*3, axis=-1))
            elif ct == 'depth':
                # Depth extraction (e.g., MiDaS)
                depth = self.extract_depth(image)
                control_images.append(depth)

        # Multi ControlNet pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model,
            controlnet=controlnets,
            torch_dtype=torch.float16
        ).to("cuda")

        output = pipe(
            prompt=prompt,
            image=control_images,
            controlnet_conditioning_scale=[1.0, 0.5],  # Strength for each
        ).images[0]

        return output


# Usage example
generator = ControlNetGenerator()

# Keep composition from reference image while changing style
reference_image = cv2.imread("reference.jpg")
result, canny = generator.generate_with_canny(
    reference_image,
    "A beautiful anime girl, studio ghibli style"
)
```

---

## 3. IP-Adapter (Image Prompt Adapter)

### 3.1 Concept

```
IP-Adapter: Using Images as Prompts

Direct style/content with images instead of/alongside text

┌────────────────────────────────────────────────────────────┐
│                    IP-Adapter Structure                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Reference Image → CLIP Image Encoder → Image Features     │
│                         ↓                                  │
│                  Projection Layer (trainable)              │
│                         ↓                                  │
│              Inject into Cross-Attention                   │
│                         ↓                                  │
│  Text Prompt + Image Features → UNet → Generated Image    │
│                                                            │
│  Use cases:                                                │
│  - Style transfer (style reference)                        │
│  - Face similarity preservation (face reference)           │
│  - Composition/color reference (composition)               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 3.2 Usage

```python
from diffusers import StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection
import torch

def use_ip_adapter():
    """Using IP-Adapter"""

    # Base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    # Load IP-Adapter
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter_sd15.bin"
    )

    # Set scale (0~1, higher = more reference image influence)
    pipe.set_ip_adapter_scale(0.6)

    # Reference image
    from PIL import Image
    style_image = Image.open("style_reference.jpg")

    # Generate
    output = pipe(
        prompt="A portrait of a woman",
        ip_adapter_image=style_image,
        num_inference_steps=30,
    ).images[0]

    return output


def ip_adapter_face():
    """IP-Adapter Face: Maintaining face similarity"""

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    # Face-specific IP-Adapter
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter-full-face_sd15.bin"
    )

    pipe.set_ip_adapter_scale(0.7)

    # Reference face
    face_image = Image.open("face_reference.jpg")

    # Generate in various styles
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
    """IP-Adapter Plus: Stronger image conditioning"""

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    # Plus version (finer control)
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter-plus_sd15.bin"
    )

    # Multiple image references
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

### 4.1 Concept

```
LCM: Ultra-fast Image Generation

Traditional Diffusion: Requires 20-50 steps
LCM: High-quality generation in 2-4 steps

How it works:
1. Distill original Diffusion model with consistency objective
2. Map any noise level directly to clean image
3. Generate with single or few steps

Advantages:
- Real-time generation possible (< 1 second)
- Interactive applications
- Low-power devices possible
```

### 4.2 Usage

```python
from diffusers import (
    DiffusionPipeline,
    LCMScheduler,
    AutoPipelineForText2Image
)

def lcm_generation():
    """LCM fast generation"""

    # Use LCM-LoRA (applies to existing models)
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    # Load LCM-LoRA
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

    # LCM scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Fast generation (4 steps!)
    image = pipe(
        prompt="A beautiful sunset over mountains",
        num_inference_steps=4,  # Very few steps
        guidance_scale=1.5,     # LCM recommends low guidance
    ).images[0]

    return image


def lcm_real_time():
    """Real-time image generation demo"""
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
    """SDXL-Turbo: 1-4 step generation"""

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    # Just 1 step!
    image = pipe(
        prompt="A cinematic shot of a cat wearing a hat",
        num_inference_steps=1,
        guidance_scale=0.0,  # Turbo doesn't need guidance
    ).images[0]

    return image
```

---

## 5. Advanced Techniques

### 5.1 Inpainting & Outpainting

```python
from diffusers import StableDiffusionInpaintPipeline

def inpainting_example():
    """Region editing (Inpainting)"""

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")

    # Original image and mask
    image = Image.open("original.jpg")
    mask = Image.open("mask.png")  # White = region to edit

    result = pipe(
        prompt="A cat sitting on the couch",
        image=image,
        mask_image=mask,
        num_inference_steps=30,
    ).images[0]

    return result


def outpainting_example():
    """Image extension (Outpainting)"""

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")

    # Place original image on canvas
    original = Image.open("original.jpg")
    canvas_size = (1024, 1024)
    canvas = Image.new("RGB", canvas_size, (128, 128, 128))

    # Center placement
    offset = ((canvas_size[0] - original.width) // 2,
              (canvas_size[1] - original.height) // 2)
    canvas.paste(original, offset)

    # Mask: white outside original region
    mask = Image.new("L", canvas_size, 255)
    mask.paste(0, offset, (offset[0] + original.width, offset[1] + original.height))

    # Extend
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
    """Style transformation"""

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    # Input image
    init_image = Image.open("photo.jpg").resize((512, 512))

    # Style transformation
    result = pipe(
        prompt="oil painting, impressionist style, vibrant colors",
        image=init_image,
        strength=0.75,  # 0~1, higher = more change
        num_inference_steps=30,
    ).images[0]

    return result
```

### 5.3 Text Embedding Manipulation

```python
def prompt_weighting():
    """Prompt weight adjustment"""
    from compel import Compel

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    # Weight syntax
    prompts = [
        "a (beautiful)++ sunset",           # ++ = 1.21x
        "a (beautiful)+++ sunset",          # +++ = 1.33x
        "a (ugly)-- sunset",                # -- = 0.83x
        "a (red:1.5) and (blue:0.5) sunset" # Explicit weights
    ]

    for prompt in prompts:
        conditioning = compel.build_conditioning_tensor(prompt)

        image = pipe(
            prompt_embeds=conditioning,
            num_inference_steps=30,
        ).images[0]


def prompt_blending():
    """Prompt blending"""
    from compel import Compel

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    # Blend two prompts
    prompt1 = "a photo of a cat"
    prompt2 = "a photo of a dog"

    cond1 = compel.build_conditioning_tensor(prompt1)
    cond2 = compel.build_conditioning_tensor(prompt2)

    # 50:50 blending
    blended = (cond1 + cond2) / 2

    image = pipe(
        prompt_embeds=blended,
        num_inference_steps=30,
    ).images[0]

    return image
```

---

## 6. Optimization Techniques

### 6.1 Memory Optimization

```python
def optimize_memory():
    """Memory optimization techniques"""

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    )

    # 1. CPU Offload
    pipe.enable_model_cpu_offload()

    # 2. Sequential CPU Offload (slower but saves more memory)
    # pipe.enable_sequential_cpu_offload()

    # 3. VAE Slicing (for large images)
    pipe.enable_vae_slicing()

    # 4. VAE Tiling (for very large images)
    pipe.enable_vae_tiling()

    # 5. Attention Slicing
    pipe.enable_attention_slicing(slice_size="auto")

    # 6. xFormers
    pipe.enable_xformers_memory_efficient_attention()

    return pipe


def batch_generation():
    """Batch generation optimization"""

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

    # Batch generation (more efficient)
    images = pipe(
        prompt=prompts,
        num_inference_steps=30,
    ).images

    return images
```

---

## References

### Papers
- Podell et al. (2023). "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis"
- Zhang et al. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models" (ControlNet)
- Ye et al. (2023). "IP-Adapter: Text Compatible Image Prompt Adapter"
- Luo et al. (2023). "Latent Consistency Models"

### Models
- [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [ControlNet](https://huggingface.co/lllyasviel/ControlNet)
- [IP-Adapter](https://huggingface.co/h94/IP-Adapter)
- [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdxl)

### Related Lessons
- [../Deep_Learning/17_Diffusion_Models.md](../Deep_Learning/17_Diffusion_Models.md)

---

## Exercises

### Exercise 1: SDXL Dual Encoder Analysis
SDXL uses two text encoders: CLIP ViT-L/14 and OpenCLIP ViT-bigG. Their embeddings are concatenated before being passed to the UNet. Explain the purpose of this dual-encoder design and describe what each encoder likely contributes. Also explain why SDXL's micro-conditioning (passing `original_size` and `target_size` parameters) improves generation quality.

<details>
<summary>Show Answer</summary>

**Dual encoder purpose**: The two encoders have different architectures and training data, leading to complementary strengths. Concatenating their embeddings provides a richer, more detailed text representation than either encoder alone, which allows the UNet to better follow complex, nuanced prompts.

**Individual contributions**:
- **CLIP ViT-L/14**: Trained on 400M image-text pairs (OpenAI WIT dataset). Good at visual-semantic alignment and associating text concepts with visual features. Strong at general semantic understanding ("a cat," "sunset," "impressionist style").
- **OpenCLIP ViT-bigG**: Larger model (1.8B params vs 307M) trained on LAION-5B (5B pairs). Better at fine-grained detail, complex compositional descriptions, and rare concepts. Provides more capacity for detailed instructions.

Together, they provide both general semantic alignment (CLIP) and fine-grained detail capture (OpenCLIP bigG).

**Micro-conditioning rationale**: During SD 1.5 training, images were resized and center-cropped to 512×512. This caused the model to learn artifacts: objects getting cropped, unusual compositions. The model "learned" these distortions as part of natural image distributions.

SDXL's micro-conditioning passes the original image size and crop coordinates as additional conditioning signals. This allows the model to:
1. Distinguish high-quality original (full image, no crop) from training-time distortions.
2. At inference, setting `original_size=target_size` signals "generate as if this is a high-quality, uncropped image," consistently yielding better compositions and fewer artifacts.

</details>

### Exercise 2: ControlNet Zero Convolution
ControlNet injects control signals into the frozen UNet using "zero convolution" layers — convolutions initialized with zero weights. Explain why zero initialization is critical for ControlNet's training stability, and describe what would happen if the control encoder's weights were initialized randomly instead.

```python
# ControlNet injection mechanism (simplified)
class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        # CRITICAL: Initialize weights and bias to zero
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

# At the start of training:
# zero_conv(any_input) = 0
# UNet output = original_output + 0 = original_output

# Why is this important?
```

<details>
<summary>Show Answer</summary>

**Why zero initialization is critical**:

At the start of ControlNet training, the zero convolutions ensure the control signal is exactly zero: `zero_conv(condition_features) = 0`. This means:
- **UNet output = original frozen UNet output + 0 = original output**
- The pre-trained UNet behaves identically to before ControlNet was added.
- The model starts from a perfect baseline (the pre-trained diffusion model), not a randomly corrupted one.

As training proceeds, the zero convolution weights gradually learn to inject the appropriate control signal, growing from zero toward meaningful values.

**If initialized randomly instead**:
- The control encoder would inject random noise into every layer of the UNet.
- The UNet would receive corrupted intermediate activations at every attention layer.
- The pre-trained weights (which produce good images) would be overwhelmed by this random injection, causing severe initial instability.
- The optimization landscape would be highly chaotic — gradients from the corrupted UNet output would be meaningless for learning the correct control behavior.
- Training would likely diverge or require an extremely slow warm-up schedule to overcome the initial corruption.

Zero initialization is an elegant solution that makes fine-tuning modular: the original model is preserved as a perfect starting point, and the new control pathway is added incrementally.

</details>

### Exercise 3: LCM vs. Standard Diffusion Speed/Quality Trade-off
LCM generates high-quality images in 2-4 steps vs. 20-50 steps for standard diffusion models. Analyze the following comparison:

```python
# Standard DDIM sampling (50 steps)
standard_image = pipe(
    prompt="A photorealistic portrait",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

# LCM sampling (4 steps)
lcm_image = pipe(
    prompt="A photorealistic portrait",
    num_inference_steps=4,
    guidance_scale=1.5  # Note: MUCH lower guidance scale
).images[0]
```

Answer the following:
- A) Why does LCM use a much lower guidance_scale than standard diffusion?
- B) What quality differences would you typically observe between the outputs?
- C) When should you use LCM vs. standard diffusion in a production system?

<details>
<summary>Show Answer</summary>

**A) Why lower guidance_scale for LCM**:
Classifier-Free Guidance (CFG) amplifies the text-conditioned signal relative to unconditional noise. At high guidance scales (7.5), the model strongly follows the prompt but also amplifies artifacts and unrealistic details. With standard diffusion at 50 steps, these artifacts are smoothed out over many denoising steps.

LCM uses a consistency objective that maps any noise level directly to the clean image in very few steps. This means each step makes much larger "jumps" in latent space. At high guidance_scale, these large jumps become overcorrections — the model overshoots toward saturated, artifact-heavy outputs. Lower guidance_scale (1.0-2.0) keeps the jumps controlled and produces more natural results in 4 steps.

**B) Typical quality differences**:
- **Standard diffusion (50 steps)**: More detail in textures (hair, fabric, fine patterns), better color gradients, more photorealistic skin tones, less over-saturation.
- **LCM (4 steps)**: Slightly softer details, occasional slight over-saturation in highlights, sometimes less fine texture detail. However, for many subjects the difference is surprisingly small — LCM is remarkably good.
- For abstract/artistic prompts, LCM quality is nearly identical to standard diffusion. For highly photorealistic portraits with fine detail, standard diffusion is noticeably better.

**C) Production use case selection**:
- **Use LCM when**: Real-time interactivity is required (< 1 second generation for user-facing applications), mobile/edge deployment with limited compute, A/B testing many prompts quickly, generating thumbnails or previews.
- **Use standard diffusion when**: Maximum quality is the priority (product photography, professional artistic output), fine textural details are important, the output will be displayed at large size or printed, the generation is offline batch processing where latency doesn't matter.

A practical strategy: use LCM for iteration and preview, switch to standard diffusion for final high-quality renders.

</details>

### Exercise 4: Multi-technique Pipeline Design
Design a complete image generation pipeline using ControlNet + IP-Adapter + LCM-LoRA together to solve the following task:

**Task**: Given a reference photo of a person and a rough pencil sketch of a scene, generate an image of that person placed in the scene depicted by the sketch, in a consistent artistic style.

Describe the role of each component and sketch the pipeline with pseudo-code.

<details>
<summary>Show Answer</summary>

**Role of each component**:
- **ControlNet (Scribble/Canny)**: Uses the pencil sketch as a spatial control signal to preserve the scene composition (object placements, general layout) from the sketch.
- **IP-Adapter (Face variant)**: Uses the reference photo to preserve the person's identity (face features, general appearance) in the generated image.
- **LCM-LoRA**: Reduces generation steps from 30 to 4, enabling real-time iteration when refining prompt or sketch.

**Pipeline pseudo-code**:

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
import torch

# 1. Load components
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Load IP-Adapter for face identity preservation
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                     weight_name="ip-adapter-full-face_sd15.bin")
pipe.set_ip_adapter_scale(0.7)  # Strong identity preservation

# Load LCM-LoRA for fast generation
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# 2. Prepare inputs
reference_photo = Image.open("person_reference.jpg")     # Identity source
pencil_sketch = Image.open("scene_sketch.jpg")           # Spatial layout source

# 3. Generate
result = pipe(
    prompt="A professional portrait of a person in [scene description], artistic style",
    negative_prompt="blurry, distorted face, multiple people",
    image=pencil_sketch,               # ControlNet: scene layout
    ip_adapter_image=reference_photo,  # IP-Adapter: person identity
    num_inference_steps=4,             # LCM: fast generation
    guidance_scale=1.5,               # LCM-appropriate scale
    controlnet_conditioning_scale=0.8  # Balance scene vs. identity
).images[0]
```

**Key design decisions**:
- `ip_adapter_scale=0.7`: Strong identity (face) preservation — high enough to keep the person's face, low enough to allow style variation.
- `controlnet_conditioning_scale=0.8`: Strong scene structure control, slightly below 1.0 to allow natural adaptation of the sketch layout.
- `guidance_scale=1.5`: LCM-compatible low guidance.
- The text prompt still guides overall style and fills details not constrained by sketch or identity.

</details>
