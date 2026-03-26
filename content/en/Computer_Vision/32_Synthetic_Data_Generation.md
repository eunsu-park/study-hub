[Previous: Optical Flow](./31_Optical_Flow.md) | [Back to Overview](./00_Overview.md)

---

# 32. Synthetic Data Generation

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the role of synthetic data in computer vision training pipelines
2. Implement domain randomization for robust model training
3. Build procedural data generation pipelines for detection and segmentation
4. Use diffusion models for controlled data augmentation
5. Analyze the domain gap between synthetic and real data

---

## Table of Contents

1. [Why Synthetic Data?](#1-why-synthetic-data)
2. [Domain Randomization](#2-domain-randomization)
3. [Procedural Data Generation](#3-procedural-data-generation)
4. [3D Rendering Pipelines](#4-3d-rendering-pipelines)
5. [Diffusion Models for Data Augmentation](#5-diffusion-models-for-data-augmentation)
6. [Domain Adaptation](#6-domain-adaptation)
7. [Practical Synthetic Data Pipeline](#7-practical-synthetic-data-pipeline)
8. [Exercises](#8-exercises)

---

## 1. Why Synthetic Data?

### 1.1 The Data Problem

```
Real data challenges:
  - Expensive to collect and annotate
  - Privacy concerns (faces, medical data)
  - Rare events are underrepresented
  - Annotations may be inconsistent or wrong
  - Limited diversity (weather, lighting, viewpoints)

Synthetic data advantages:
  + Free, unlimited, perfect annotations
  + Full control over conditions (lighting, weather, pose)
  + Can generate rare/dangerous scenarios safely
  + No privacy issues
  + Reproducible

Synthetic data challenges:
  - Domain gap: synthetic ≠ real
  - Rendering quality affects transfer
  - Must cover sufficient variation
  - Risk of "shortcut learning" on synthetic artifacts
```

### 1.2 Success Stories

```
Where synthetic data works:

Autonomous driving:
  - NVIDIA DRIVE Sim, CARLA simulator
  - Synthetic data pre-training + real data fine-tuning
  - Especially useful for rare scenarios (accidents)

Robotics:
  - Sim-to-real transfer (Lesson 23 in RL)
  - Domain randomization makes policies robust

Medical imaging:
  - Generate rare pathologies
  - Augment small datasets
  - Privacy-preserving training

Industrial inspection:
  - Generate defect images programmatically
  - Rare defect types well-represented

Face recognition:
  - Synthetic faces for training (privacy-friendly)
  - Control demographics and poses
```

---

## 2. Domain Randomization

### 2.1 Randomization Strategies

```python
import cv2
import numpy as np
from PIL import Image


class DomainRandomizer:
    """Apply domain randomization to synthetic images."""

    def __init__(self):
        self.randomizers = [
            self.random_lighting,
            self.random_blur,
            self.random_noise,
            self.random_color_jitter,
            self.random_texture_overlay,
        ]

    def apply(self, image, n_random=3):
        """Apply random subset of augmentations."""
        selected = np.random.choice(
            self.randomizers, min(n_random, len(self.randomizers)), replace=False
        )
        for fn in selected:
            image = fn(image)
        return image

    def random_lighting(self, image):
        """Simulate different lighting conditions."""
        brightness = np.random.uniform(0.5, 1.5)
        contrast = np.random.uniform(0.7, 1.3)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        mean = image.mean()
        image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
        return image

    def random_blur(self, image):
        """Random Gaussian blur."""
        ksize = np.random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def random_noise(self, image):
        """Add random Gaussian noise."""
        sigma = np.random.uniform(5, 25)
        noise = np.random.normal(0, sigma, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)

    def random_color_jitter(self, image):
        """Random hue, saturation, value shifts."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] += np.random.uniform(-10, 10)
        hsv[:, :, 1] *= np.random.uniform(0.7, 1.3)
        hsv[:, :, 2] *= np.random.uniform(0.7, 1.3)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def random_texture_overlay(self, image):
        """Overlay random texture for background variation."""
        H, W = image.shape[:2]
        # Generate random pattern
        pattern = np.random.randint(0, 50, (H, W, 3), dtype=np.uint8)
        alpha = np.random.uniform(0.0, 0.15)
        return cv2.addWeighted(image, 1 - alpha, pattern, alpha, 0)
```

---

## 3. Procedural Data Generation

### 3.1 Object Placement

```python
class SyntheticSceneGenerator:
    """Generate synthetic scenes with annotations."""

    def __init__(self, background_dir, object_dir, image_size=(640, 480)):
        self.backgrounds = self._load_images(background_dir)
        self.objects = self._load_images_with_masks(object_dir)
        self.image_size = image_size

    def _load_images(self, directory):
        import glob
        paths = glob.glob(f"{directory}/*.jpg") + glob.glob(f"{directory}/*.png")
        return [cv2.imread(p) for p in paths]

    def _load_images_with_masks(self, directory):
        # Load object images with alpha channel
        import glob
        paths = glob.glob(f"{directory}/*.png")
        objects = []
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img.shape[2] == 4:  # Has alpha
                objects.append(img)
        return objects

    def generate_scene(self, n_objects=5):
        """Generate one synthetic scene with annotations."""
        W, H = self.image_size

        # Random background
        bg_idx = np.random.randint(len(self.backgrounds))
        scene = cv2.resize(self.backgrounds[bg_idx], (W, H))

        annotations = []
        instance_mask = np.zeros((H, W), dtype=np.int32)

        for i in range(n_objects):
            # Random object
            obj_idx = np.random.randint(len(self.objects))
            obj = self.objects[obj_idx].copy()

            # Random scale
            scale = np.random.uniform(0.3, 1.5)
            obj_h, obj_w = int(obj.shape[0] * scale), int(obj.shape[1] * scale)
            obj = cv2.resize(obj, (obj_w, obj_h))

            # Random position
            x = np.random.randint(0, max(1, W - obj_w))
            y = np.random.randint(0, max(1, H - obj_h))

            # Paste object onto scene
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

                # Instance mask
                mask_region = (alpha[:obj_h_clip, :obj_w_clip, 0] > 0.5)
                instance_mask[y:y_end, x:x_end][mask_region] = i + 1

            # Bounding box annotation
            annotations.append({
                'class_id': obj_idx,
                'bbox': [x, y, x_end, y_end],
                'instance_id': i + 1,
            })

        return scene, annotations, instance_mask

    def generate_dataset(self, n_images=1000, output_dir='synthetic_data'):
        """Generate full synthetic dataset."""
        import os
        import json

        os.makedirs(f"{output_dir}/images", exist_ok=True)
        all_annotations = []

        for img_id in range(n_images):
            n_objects = np.random.randint(1, 8)
            scene, annotations, mask = self.generate_scene(n_objects)

            # Apply domain randomization
            randomizer = DomainRandomizer()
            scene = randomizer.apply(scene)

            # Save
            img_path = f"{output_dir}/images/{img_id:06d}.jpg"
            cv2.imwrite(img_path, scene)

            mask_path = f"{output_dir}/images/{img_id:06d}_mask.png"
            cv2.imwrite(mask_path, mask.astype(np.uint16))

            for ann in annotations:
                ann['image_id'] = img_id
            all_annotations.extend(annotations)

            if (img_id + 1) % 100 == 0:
                print(f"Generated {img_id + 1}/{n_images} images")

        # Save annotations
        with open(f"{output_dir}/annotations.json", 'w') as f:
            json.dump(all_annotations, f)

        print(f"Dataset saved to {output_dir}: {n_images} images")
```

---

## 4. 3D Rendering Pipelines

### 4.1 Blender for Synthetic Data

```python
# Blender Python API for synthetic data generation
# Run inside Blender: blender --background --python generate.py

"""
import bpy
import numpy as np
import os

def setup_scene():
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Add camera
    bpy.ops.object.camera_add(location=(0, -5, 3))
    cam = bpy.context.object
    cam.rotation_euler = (1.1, 0, 0)
    bpy.context.scene.camera = cam

    # Add light
    bpy.ops.object.light_add(type='SUN', location=(3, -3, 5))
    light = bpy.context.object
    light.data.energy = 3

    # Random light position for domain randomization
    light.location.x = np.random.uniform(-5, 5)
    light.location.y = np.random.uniform(-5, 5)

def add_random_objects(n_objects=5):
    objects = []
    for i in range(n_objects):
        # Random primitive
        shape = np.random.choice(['cube', 'sphere', 'cylinder'])
        if shape == 'cube':
            bpy.ops.mesh.primitive_cube_add()
        elif shape == 'sphere':
            bpy.ops.mesh.primitive_uv_sphere_add()
        else:
            bpy.ops.mesh.primitive_cylinder_add()

        obj = bpy.context.object

        # Random position
        obj.location = (
            np.random.uniform(-2, 2),
            np.random.uniform(-2, 2),
            np.random.uniform(0, 2),
        )

        # Random scale
        s = np.random.uniform(0.2, 0.8)
        obj.scale = (s, s, s)

        # Random color
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

# Usage:
# setup_scene()
# objects = add_random_objects(5)
# render_and_save('/tmp/synthetic_001.png')
"""
```

---

## 5. Diffusion Models for Data Augmentation

### 5.1 Controlled Generation

```
Using diffusion models (Stable Diffusion) for data augmentation:

Approaches:
1. Text-to-Image Generation:
   "A red car on a rainy road at night"
   → Generate diverse training images

2. Image-to-Image with ControlNet:
   Input: segmentation map + text prompt
   Output: Photorealistic image matching the layout

3. Inpainting for Augmentation:
   Mask out part of real image
   Fill with different object/texture

4. Style Transfer:
   Convert synthetic renders to photorealistic style
   Reduces domain gap significantly
```

### 5.2 ControlNet for Synthetic Data

```python
# Conceptual example using diffusion for data augmentation
# Requires: pip install diffusers transformers

def generate_with_controlnet(segmentation_map, prompt, n_images=5):
    """
    Generate photorealistic images from segmentation maps.
    Uses ControlNet conditioned on semantic segmentation.
    """
    # from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    # controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg")
    # pipe = StableDiffusionControlNetPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5", controlnet=controlnet
    # )

    images = []
    for i in range(n_images):
        # Same layout, different appearance
        # image = pipe(
        #     prompt=prompt,
        #     image=segmentation_map,
        #     num_inference_steps=20,
        # ).images[0]
        # images.append(image)
        pass

    return images

# Example:
# seg_map = load_segmentation("city_layout.png")
# images = generate_with_controlnet(
#     seg_map,
#     "urban street scene, sunny day, high resolution photo",
#     n_images=10
# )
# Each image has the SAME layout but different visual appearance
```

---

## 6. Domain Adaptation

### 6.1 Bridging the Domain Gap

```
Techniques to reduce synthetic-to-real domain gap:

1. Feature-level adaptation:
   Train discriminator to make synthetic and real features indistinguishable
   (Adversarial domain adaptation)

2. Image-level adaptation:
   Transform synthetic images to look more realistic
   (CycleGAN, style transfer)

3. Self-training:
   Train on synthetic → pseudo-label real data → retrain on combined

4. Fine-tuning:
   Pre-train on synthetic, fine-tune on small real dataset
   Most practical approach

5. Domain randomization:
   Make synthetic data SO varied that real data falls within the distribution
```

---

## 7. Practical Synthetic Data Pipeline

### 7.1 End-to-End Pipeline

```python
def synthetic_data_pipeline(real_data_path, output_dir, n_synthetic=10000):
    """
    Complete synthetic data pipeline:
    1. Analyze real data distribution
    2. Generate synthetic data matching statistics
    3. Apply domain randomization
    4. Mix with real data
    5. Train and evaluate
    """
    # Step 1: Analyze real data
    real_stats = analyze_dataset(real_data_path)
    print(f"Real data: {real_stats['n_images']} images, "
          f"{real_stats['n_classes']} classes")

    # Step 2: Generate synthetic data
    generator = SyntheticSceneGenerator(
        background_dir=f"{real_data_path}/backgrounds",
        object_dir=f"{real_data_path}/objects",
    )
    generator.generate_dataset(n_images=n_synthetic, output_dir=output_dir)

    # Step 3: Domain randomization (already applied in generator)

    # Step 4: Mix datasets
    mixed_dataset = create_mixed_dataset(
        real_path=real_data_path,
        synthetic_path=output_dir,
        synthetic_ratio=0.5,  # 50% synthetic
    )

    # Step 5: Train
    model = train_model(mixed_dataset)

    # Step 6: Evaluate on real-only test set
    results = evaluate_model(model, f"{real_data_path}/test")
    print(f"mAP on real test set: {results['mAP']:.4f}")

    return model, results
```

---

## 8. Exercises

### Exercise 1: Synthetic Object Detection Dataset

Create a synthetic dataset for object detection:
1. Collect 10 object images with transparency (PNG with alpha)
2. Collect 20 background images
3. Implement random placement, scaling, rotation
4. Generate 5000 training images with COCO-format annotations
5. Train YOLOv8 and evaluate on real images

### Exercise 2: Domain Randomization Study

Study the effect of domain randomization:
1. Generate synthetic data WITHOUT randomization
2. Generate with increasing randomization levels (1-5 augmentations)
3. Train the same model on each variant
4. Test all models on the same real test set
5. Plot: real-data performance vs randomization level

### Exercise 3: Diffusion-Based Augmentation

Use diffusion models for data augmentation:
1. Start with a small real dataset (100 images)
2. Generate variations using img2img diffusion
3. Combine original + generated for training
4. Compare: original-only vs augmented training
5. Measure: diminishing returns as augmentation amount increases

### Exercise 4: Domain Gap Analysis

Quantify and reduce the domain gap:
1. Train model on synthetic data only, test on real
2. Visualize feature distributions (t-SNE) for synthetic vs real
3. Apply CycleGAN to transform synthetic → realistic style
4. Re-train on styled synthetic data
5. Measure domain gap reduction at each step

### Exercise 5: Complete Synthetic Pipeline

Build a production-grade synthetic data pipeline:
1. Define target application (e.g., product detection in retail)
2. Create 3D models of target objects
3. Generate photorealistic renders with Blender/Unity
4. Apply comprehensive domain randomization
5. Train, evaluate, and iteratively improve the pipeline

---

*End of Lesson 32 - Congratulations on completing the Computer Vision course!*
