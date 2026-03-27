"""
Exercises for Lesson 15: Advanced Image Generation
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""


# === Exercise 1: SDXL Dual Encoder Analysis ===
# Problem: Explain dual-encoder design and micro-conditioning.

def exercise_1():
    """Solution: SDXL dual encoder analysis"""
    print("  Dual encoder purpose:")
    print("    Two encoders with different architectures/training data provide")
    print("    complementary strengths. Concatenated embeddings give richer")
    print("    text representation for the UNet.")
    print()
    print("  Individual contributions:")
    print("    CLIP ViT-L/14 (307M params, 400M pairs):")
    print("      Good at visual-semantic alignment and general concepts")
    print("      ('a cat', 'sunset', 'impressionist style')")
    print()
    print("    OpenCLIP ViT-bigG (1.8B params, LAION-5B 5B pairs):")
    print("      Better at fine-grained detail, complex compositions,")
    print("      rare concepts. More capacity for detailed instructions.")
    print()
    print("  Micro-conditioning rationale:")
    print("    SD 1.5 training resized/cropped to 512x512, learning distortions.")
    print("    SDXL passes original_size and crop_coords as conditioning.")
    print("    At inference, setting original_size=target_size signals")
    print("    'generate as if high-quality, uncropped image' -> better compositions.")


# === Exercise 2: ControlNet Zero Convolution ===
# Problem: Explain why zero initialization is critical.

def exercise_2():
    """Solution: ControlNet zero convolution"""
    print("  Zero convolution: Conv2d initialized with zero weights and bias.")
    print()
    print("  Why zero initialization is critical:")
    print("    At start: zero_conv(any_input) = 0")
    print("    UNet output = original_output + 0 = original_output")
    print("    Pre-trained UNet behaves identically to before ControlNet was added.")
    print("    Model starts from a PERFECT baseline.")
    print()
    print("    As training proceeds, zero conv weights gradually learn")
    print("    to inject appropriate control signal, growing from zero.")
    print()
    print("  If initialized randomly instead:")
    print("    Control encoder injects random noise into every UNet layer.")
    print("    Pre-trained weights overwhelmed by random injection.")
    print("    Optimization landscape is chaotic -- gradients meaningless.")
    print("    Training likely diverges or needs extremely slow warm-up.")
    print()
    print("  Zero initialization = elegant modular fine-tuning:")
    print("    Original model preserved as starting point,")
    print("    new control pathway added incrementally.")


# === Exercise 3: LCM vs Standard Diffusion ===
# Problem: Analyze speed/quality trade-offs.

def exercise_3():
    """Solution: LCM vs standard diffusion analysis"""
    print("  A) Why LCM uses lower guidance_scale:")
    print("    Standard diffusion at 50 steps: high CFG (7.5) artifacts")
    print("    smoothed over many denoising steps.")
    print("    LCM at 4 steps: each step makes much larger 'jumps'.")
    print("    At high guidance_scale, jumps become overcorrections ->")
    print("    saturated, artifact-heavy outputs. Lower guidance (1.0-2.0)")
    print("    keeps jumps controlled.")
    print()

    print("  B) Typical quality differences:")
    print("    Standard (50 steps): More texture detail, better gradients,")
    print("      more photorealistic skin tones, less over-saturation.")
    print("    LCM (4 steps): Slightly softer details, occasional")
    print("      over-saturation, less fine texture. But surprisingly good")
    print("      for many subjects.")
    print()

    print("  C) Production use case selection:")
    print("    Use LCM when: Real-time interactivity (<1s), edge deployment,")
    print("      A/B testing many prompts, thumbnails/previews.")
    print("    Use standard when: Maximum quality priority, fine textures,")
    print("      large display/print, offline batch processing.")
    print("    Practical strategy: LCM for iteration, standard for final renders.")


# === Exercise 4: Multi-technique Pipeline Design ===
# Problem: Design pipeline using ControlNet + IP-Adapter + LCM-LoRA.

def exercise_4():
    """Solution: Multi-technique pipeline design"""
    print("  Task: Reference photo of person + pencil sketch of scene ->")
    print("  generate person placed in sketched scene")
    print()

    print("  Component roles:")
    print("    ControlNet (Scribble/Canny): Uses pencil sketch as spatial control")
    print("      signal to preserve scene composition from sketch.")
    print("    IP-Adapter (Face variant): Uses reference photo to preserve")
    print("      person's identity (face features, appearance).")
    print("    LCM-LoRA: Reduces generation steps from 30 to 4 for")
    print("      real-time iteration when refining prompt/sketch.")
    print()

    print("  Pipeline pseudo-code:")
    print("    # 1. Load components")
    print("    controlnet = ControlNetModel('sd-controlnet-scribble')")
    print("    pipe = SDControlNetPipeline('sd-v1-5', controlnet=controlnet)")
    print("    pipe.load_ip_adapter('ip-adapter-full-face_sd15.bin')")
    print("    pipe.set_ip_adapter_scale(0.7)  # Strong identity")
    print("    pipe.load_lora_weights('lcm-lora-sdv1-5')")
    print("    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)")
    print()
    print("    # 2. Generate")
    print("    result = pipe(")
    print("      prompt='A professional portrait...',")
    print("      image=pencil_sketch,                # ControlNet: scene layout")
    print("      ip_adapter_image=reference_photo,   # IP-Adapter: identity")
    print("      num_inference_steps=4,              # LCM: fast generation")
    print("      guidance_scale=1.5,                 # LCM-appropriate")
    print("      controlnet_conditioning_scale=0.8   # Balance scene vs identity")
    print("    )")
    print()

    print("  Key design decisions:")
    print("    ip_adapter_scale=0.7: Strong face preservation, allows style variation")
    print("    controlnet_scale=0.8: Strong scene structure, slightly below 1.0")
    print("    guidance_scale=1.5: LCM-compatible low guidance")
    print("    Text prompt guides overall style and fills unconstrained details")


if __name__ == "__main__":
    print("=== Exercise 1: SDXL Dual Encoder Analysis ===")
    exercise_1()
    print("\n=== Exercise 2: ControlNet Zero Convolution ===")
    exercise_2()
    print("\n=== Exercise 3: LCM vs Standard Diffusion ===")
    exercise_3()
    print("\n=== Exercise 4: Multi-technique Pipeline Design ===")
    exercise_4()
    print("\nAll exercises completed!")
