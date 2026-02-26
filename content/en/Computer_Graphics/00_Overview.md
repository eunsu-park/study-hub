# Computer Graphics

## Overview

Computer graphics is the science and art of generating images using computers — from 2D shapes on a screen to photorealistic 3D renderings and real-time interactive worlds. This topic covers the mathematical foundations (transformations, projections, shading models), the graphics pipeline (rasterization, texturing, lighting), modern GPU programming (shaders, compute), and advanced rendering techniques (ray tracing, global illumination). Whether you're building games, scientific visualizations, or film VFX, understanding computer graphics gives you deep control over how digital worlds are created and displayed.

## Prerequisites

- **Mathematical Methods**: Linear algebra, matrix operations (Mathematical_Methods L03-L05)
- **Programming**: Object-oriented programming, basic data structures (Programming L05-L06)
- **Python**: NumPy, Matplotlib (Python L01-L08)
- **Web Development** (optional): HTML/CSS/JS for WebGL lessons (Web_Development L01-L08)

## Learning Path

```
Foundations (L01-L04)
├── L01: Graphics Pipeline Overview
├── L02: 2D Transformations
├── L03: 3D Transformations and Projections
└── L04: Rasterization

Shading & Textures (L05-L08)
├── L05: Shading Models
├── L06: Texture Mapping
├── L07: WebGL Fundamentals
└── L08: Shader Programming (GLSL)

Advanced Rendering (L09-L12)
├── L09: Scene Graphs and Spatial Data Structures
├── L10: Ray Tracing Basics
├── L11: Path Tracing and Global Illumination
└── L12: Animation and Skeletal Systems

Modern Techniques (L13-L16)
├── L13: Particle Systems and Effects
├── L14: GPU Computing
├── L15: Real-Time Rendering Techniques
└── L16: Modern Graphics APIs Overview
```

## Lessons

| # | Lesson | Description |
|---|--------|-------------|
| 01 | [Graphics Pipeline Overview](01_Graphics_Pipeline_Overview.md) | CPU vs GPU stages, vertex/fragment processing, double buffering |
| 02 | [2D Transformations](02_2D_Transformations.md) | Translation, rotation, scaling, homogeneous coordinates, composition |
| 03 | [3D Transformations and Projections](03_3D_Transformations_and_Projections.md) | Model/view/projection matrices, perspective vs orthographic, camera models |
| 04 | [Rasterization](04_Rasterization.md) | Line drawing, triangle rasterization, z-buffer, anti-aliasing |
| 05 | [Shading Models](05_Shading_Models.md) | Phong, Blinn-Phong, PBR (Cook-Torrance), BRDF, Fresnel |
| 06 | [Texture Mapping](06_Texture_Mapping.md) | UV coordinates, mipmaps, filtering, normal/bump mapping, PBR textures |
| 07 | [WebGL Fundamentals](07_WebGL_Fundamentals.md) | WebGL context, buffers, shaders, drawing, interaction |
| 08 | [Shader Programming (GLSL)](08_Shader_Programming_GLSL.md) | Vertex/fragment shaders, uniforms, varyings, built-in functions |
| 09 | [Scene Graphs and Spatial Data Structures](09_Scene_Graphs_and_Spatial_Data_Structures.md) | Scene hierarchy, BVH, octree, BSP tree, frustum culling |
| 10 | [Ray Tracing Basics](10_Ray_Tracing_Basics.md) | Ray generation, intersection tests, recursive ray tracing, shadows, reflections |
| 11 | [Path Tracing and Global Illumination](11_Path_Tracing_and_Global_Illumination.md) | Monte Carlo integration, rendering equation, importance sampling, denoising |
| 12 | [Animation and Skeletal Systems](12_Animation_and_Skeletal_Systems.md) | Keyframes, interpolation, forward/inverse kinematics, skinning |
| 13 | [Particle Systems and Effects](13_Particle_Systems_and_Effects.md) | Emitters, forces, billboards, GPU particles, volumetric effects |
| 14 | [GPU Computing](14_GPU_Computing.md) | Compute shaders, GPGPU, parallel patterns, image processing |
| 15 | [Real-Time Rendering Techniques](15_Real_Time_Rendering_Techniques.md) | Deferred shading, shadow maps, SSAO, bloom, HDR, LOD |
| 16 | [Modern Graphics APIs Overview](16_Modern_Graphics_APIs_Overview.md) | Vulkan, Metal, DirectX 12, explicit resource management, render graphs |

## Relationship to Other Topics

| Topic | Connection |
|-------|-----------|
| Mathematical_Methods | Linear algebra, matrix transformations, coordinate systems |
| Optics | Physical basis of light transport, reflection models, lenses |
| Computer_Vision | Image formation, camera models, 3D reconstruction |
| Deep_Learning | Neural rendering (NeRF), differentiable rendering |
| Web_Development | WebGL, Three.js, canvas-based graphics |
| Signal_Processing | Anti-aliasing, texture filtering, image processing |

## Example Files

Located in `examples/Computer_Graphics/`:

| File | Description |
|------|-------------|
| `01_transformations_2d.py` | 2D transformation matrices, composition, animation |
| `02_transformations_3d.py` | 3D model/view/projection, camera orbit |
| `03_rasterizer.py` | Software rasterizer: line drawing, triangle fill, z-buffer |
| `04_shading.py` | Phong, Blinn-Phong, and PBR shading comparison |
| `05_texture_mapping.py` | UV mapping, bilinear filtering, mipmaps |
| `06_webgl_triangle.html` | Minimal WebGL: colored triangle with vertex/fragment shaders |
| `07_webgl_cube.html` | Interactive 3D cube with lighting and textures |
| `08_ray_tracer.py` | Simple recursive ray tracer: spheres, planes, reflections |
| `09_path_tracer.py` | Monte Carlo path tracer with importance sampling |
| `10_particle_system.py` | GPU-style particle system with forces and billboards |
| `11_animation.py` | Keyframe interpolation, skeletal animation demo |
| `12_scene_graph.py` | Hierarchical scene graph with BVH traversal |
