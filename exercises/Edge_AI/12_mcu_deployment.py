"""
Exercises for Lesson 12: MCU Deployment
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""

import math
import numpy as np


# === Exercise 1: MCU Memory Constraints ===
# Problem: Given MCU specifications, calculate whether a model fits
# in Flash and RAM, considering code, data, and activation storage.

def exercise_1():
    """Calculate model fit for MCU memory constraints."""
    mcus = [
        {
            "name": "ARM Cortex-M4 (STM32F4)",
            "flash_kb": 512,
            "sram_kb": 128,
            "clock_mhz": 168,
            "mac_per_cycle": 1,
        },
        {
            "name": "ARM Cortex-M7 (STM32H7)",
            "flash_kb": 2048,
            "sram_kb": 512,
            "clock_mhz": 480,
            "mac_per_cycle": 1,
        },
        {
            "name": "ARM Cortex-M55 (with Helium MVE)",
            "flash_kb": 2048,
            "sram_kb": 512,
            "clock_mhz": 400,
            "mac_per_cycle": 8,  # Vector extension
        },
        {
            "name": "ESP32-S3",
            "flash_kb": 8192,  # External flash
            "sram_kb": 512,
            "clock_mhz": 240,
            "mac_per_cycle": 1,
        },
    ]

    models = [
        {
            "name": "Keyword Spotting (DS-CNN)",
            "weights_kb": 25,
            "peak_activation_kb": 15,
            "runtime_kb": 30,  # TFLite Micro runtime
            "ops_per_inference": 6_000_000,
        },
        {
            "name": "Person Detection (MobileNet 0.25x)",
            "weights_kb": 250,
            "peak_activation_kb": 80,
            "runtime_kb": 50,
            "ops_per_inference": 60_000_000,
        },
        {
            "name": "Anomaly Detection (AutoEncoder)",
            "weights_kb": 10,
            "peak_activation_kb": 5,
            "runtime_kb": 20,
            "ops_per_inference": 500_000,
        },
        {
            "name": "Image Classification (MCUNet)",
            "weights_kb": 500,
            "peak_activation_kb": 200,
            "runtime_kb": 50,
            "ops_per_inference": 80_000_000,
        },
    ]

    print("  MCU Model Fit Analysis:\n")

    for mcu in mcus:
        print(f"  [{mcu['name']}]")
        print(f"    Flash: {mcu['flash_kb']} KB, "
              f"SRAM: {mcu['sram_kb']} KB, "
              f"Clock: {mcu['clock_mhz']} MHz")

        # Firmware overhead
        firmware_flash_kb = 100  # OS + drivers
        firmware_sram_kb = 20    # Stack + heap

        avail_flash = mcu['flash_kb'] - firmware_flash_kb
        avail_sram = mcu['sram_kb'] - firmware_sram_kb

        print(f"    Available: Flash={avail_flash} KB, SRAM={avail_sram} KB\n")
        print(f"    {'Model':<30} {'Flash':>8} {'SRAM':>8} {'Fits?':>6} {'Latency':>10}")
        print("    " + "-" * 66)

        for model in models:
            flash_needed = model['weights_kb'] + model['runtime_kb']
            sram_needed = model['peak_activation_kb'] + firmware_sram_kb
            fits_flash = flash_needed <= avail_flash
            fits_sram = sram_needed <= avail_sram
            fits = "YES" if (fits_flash and fits_sram) else "NO"

            # Latency estimate
            mac_per_sec = mcu['clock_mhz'] * 1e6 * mcu['mac_per_cycle']
            latency_ms = (model['ops_per_inference'] / mac_per_sec) * 1000

            print(f"    {model['name']:<30} {flash_needed:>6}KB {sram_needed:>6}KB "
                  f"{fits:>6} {latency_ms:>8.1f}ms")

        print()


# === Exercise 2: TFLite Micro Memory Layout ===
# Problem: Explain how TFLite Micro manages memory on MCUs without
# dynamic allocation, and calculate the tensor arena size.

def exercise_2():
    """Explain TFLite Micro memory layout and arena sizing."""
    print("  TFLite Micro Memory Layout:\n")
    print("  +--------------------------------------------------+")
    print("  |                   FLASH (ROM)                     |")
    print("  |  +-----------+  +-----------+  +---------------+ |")
    print("  |  | Firmware  |  | TFLite    |  | Model Weights | |")
    print("  |  | + Drivers |  | Micro     |  | (Flatbuffer)  | |")
    print("  |  | ~100 KB   |  | Runtime   |  | 10-500 KB     | |")
    print("  |  |           |  | 20-50 KB  |  |               | |")
    print("  |  +-----------+  +-----------+  +---------------+ |")
    print("  +--------------------------------------------------+")
    print()
    print("  +--------------------------------------------------+")
    print("  |                    SRAM (RAM)                     |")
    print("  |  +--------+  +--------------+  +---------------+ |")
    print("  |  | Stack  |  | Tensor Arena |  | Interpreter   | |")
    print("  |  | + Heap |  | (activations |  | State         | |")
    print("  |  | ~20 KB |  |  + scratch)  |  | ~5 KB         | |")
    print("  |  |        |  | 10-200 KB    |  |               | |")
    print("  |  +--------+  +--------------+  +---------------+ |")
    print("  +--------------------------------------------------+")

    print("\n  Key constraint: NO dynamic memory allocation (no malloc/free)")
    print("  Everything must be statically allocated or use the tensor arena.\n")

    # Calculate tensor arena size for a simple model
    print("  Tensor Arena Sizing Example (Keyword Spotting Model):\n")

    layers = [
        {"name": "Input (MFCC)", "output_shape": (1, 49, 10, 1), "dtype": "int8"},
        {"name": "Conv2D_1", "output_shape": (1, 25, 5, 64), "dtype": "int8"},
        {"name": "Conv2D_2", "output_shape": (1, 13, 3, 64), "dtype": "int8"},
        {"name": "Conv2D_3", "output_shape": (1, 7, 2, 64), "dtype": "int8"},
        {"name": "Flatten", "output_shape": (1, 896), "dtype": "int8"},
        {"name": "Dense", "output_shape": (1, 12), "dtype": "int8"},
    ]

    print(f"  {'Layer':<20} {'Output Shape':<20} {'Size (bytes)':>12}")
    print("  " + "-" * 55)

    max_activation = 0
    for layer in layers:
        size = 1
        for dim in layer['output_shape']:
            size *= dim
        bytes_per_elem = 1  # INT8
        total_bytes = size * bytes_per_elem
        max_activation = max(max_activation, total_bytes)
        print(f"  {layer['name']:<20} {str(layer['output_shape']):<20} "
              f"{total_bytes:>12,}")

    # TFLite Micro uses double-buffering (current + next layer)
    # Plus scratch space for im2col and other temporaries
    arena_estimate = max_activation * 3  # Conservative estimate

    print(f"\n  Peak single tensor:       {max_activation:,} bytes")
    print(f"  Estimated tensor arena:   {arena_estimate:,} bytes ({arena_estimate/1024:.1f} KB)")
    print(f"  (Includes double-buffering and scratch space)")
    print(f"\n  In code:")
    print(f"    constexpr int kTensorArenaSize = {arena_estimate};")
    print(f"    uint8_t tensor_arena[kTensorArenaSize];")


# === Exercise 3: CMSIS-NN Operator Mapping ===
# Problem: Map common neural network operations to their CMSIS-NN
# optimized implementations for ARM Cortex-M processors.

def exercise_3():
    """Map neural network ops to CMSIS-NN kernels."""
    op_mapping = [
        {
            "nn_op": "Conv2D (INT8)",
            "cmsis_nn": "arm_convolve_s8()",
            "optimization": "SIMD (2 MACs/cycle on M4), im2col for data layout",
            "speedup": "4-8x vs naive C",
        },
        {
            "nn_op": "Depthwise Conv2D (INT8)",
            "cmsis_nn": "arm_depthwise_conv_s8()",
            "optimization": "Per-channel quantization, optimized for groups=C_in",
            "speedup": "3-5x",
        },
        {
            "nn_op": "Fully Connected (INT8)",
            "cmsis_nn": "arm_fully_connected_s8()",
            "optimization": "Matrix-vector multiply with SIMD, output shift/saturate",
            "speedup": "3-6x",
        },
        {
            "nn_op": "Average Pooling",
            "cmsis_nn": "arm_avgpool_s8()",
            "optimization": "Accumulate in int32, divide at end",
            "speedup": "2-3x",
        },
        {
            "nn_op": "Max Pooling",
            "cmsis_nn": "arm_max_pool_s8()",
            "optimization": "SIMD comparison, window sliding",
            "speedup": "2-3x",
        },
        {
            "nn_op": "ReLU",
            "cmsis_nn": "arm_relu_q7() / arm_nn_activation_s8()",
            "optimization": "SIMD max(0, x), operates on 4 values at once",
            "speedup": "4x",
        },
        {
            "nn_op": "Softmax",
            "cmsis_nn": "arm_softmax_s8()",
            "optimization": "Fixed-point exp() approximation, avoids FPU",
            "speedup": "5-10x",
        },
        {
            "nn_op": "Add (element-wise)",
            "cmsis_nn": "arm_elementwise_add_s8()",
            "optimization": "Requantize both inputs to common scale, SIMD add",
            "speedup": "2-4x",
        },
    ]

    print("  CMSIS-NN Operator Mapping (ARM Cortex-M):\n")
    print(f"  {'NN Operation':<28} {'CMSIS-NN Function':<32} {'Speedup':<10}")
    print("  " + "-" * 72)

    for op in op_mapping:
        print(f"  {op['nn_op']:<28} {op['cmsis_nn']:<32} {op['speedup']}")

    print("\n  Detailed optimizations:")
    for op in op_mapping[:4]:
        print(f"\n  {op['nn_op']}:")
        print(f"    Function:     {op['cmsis_nn']}")
        print(f"    Optimization: {op['optimization']}")

    print("\n  Integration with TFLite Micro:")
    print("    TFLite Micro automatically uses CMSIS-NN kernels when:")
    print("    1. Target is ARM Cortex-M (M4, M7, M33, M55)")
    print("    2. Model uses INT8 quantization")
    print("    3. CMSIS-NN is linked during build")
    print("    4. Operator shapes match CMSIS-NN requirements")


# === Exercise 4: TinyML Model Optimization ===
# Problem: Given a model that doesn't fit on a target MCU,
# apply optimization techniques to make it fit.

def exercise_4():
    """Optimize a model to fit on a constrained MCU."""
    target = {
        "name": "ARM Cortex-M4 (STM32F446)",
        "flash_kb": 512,
        "sram_kb": 128,
        "available_flash_kb": 400,  # After firmware
        "available_sram_kb": 100,   # After stack/heap
    }

    original = {
        "name": "Original MobileNet 0.5x (INT8)",
        "weights_kb": 700,
        "peak_activation_kb": 150,
        "accuracy": 0.85,
        "ops": 50_000_000,
    }

    print(f"  Target: {target['name']}")
    print(f"    Available Flash: {target['available_flash_kb']} KB")
    print(f"    Available SRAM:  {target['available_sram_kb']} KB\n")

    print(f"  Original model: {original['name']}")
    print(f"    Weights:     {original['weights_kb']} KB (needs {target['available_flash_kb']} KB)")
    print(f"    Activations: {original['peak_activation_kb']} KB (needs {target['available_sram_kb']} KB)")
    print(f"    Accuracy:    {original['accuracy']:.0%}")
    fits_flash = original['weights_kb'] <= target['available_flash_kb']
    fits_sram = original['peak_activation_kb'] <= target['available_sram_kb']
    print(f"    Fits Flash:  {'YES' if fits_flash else 'NO - need to reduce by ' + str(original['weights_kb'] - target['available_flash_kb']) + ' KB'}")
    print(f"    Fits SRAM:   {'YES' if fits_sram else 'NO - need to reduce by ' + str(original['peak_activation_kb'] - target['available_sram_kb']) + ' KB'}")

    # Optimization steps
    optimizations = [
        {
            "step": "1. Width multiplier 0.5x -> 0.25x",
            "effect": "Weights ~4x smaller, activations ~2x smaller",
            "weights_kb": 180,
            "activation_kb": 80,
            "accuracy": 0.78,
        },
        {
            "step": "2. Reduce input resolution 96x96 -> 64x64",
            "effect": "Activations reduced by (64/96)^2 = 0.44x",
            "weights_kb": 180,  # Weights unchanged
            "activation_kb": 35,
            "accuracy": 0.75,
        },
        {
            "step": "3. Knowledge distillation from 0.5x teacher",
            "effect": "Recover accuracy without changing size",
            "weights_kb": 180,
            "activation_kb": 35,
            "accuracy": 0.80,
        },
        {
            "step": "4. Structured pruning (30% channels)",
            "effect": "Further reduce weights and activations",
            "weights_kb": 126,
            "activation_kb": 25,
            "accuracy": 0.78,
        },
        {
            "step": "5. INT4 quantization for last 3 conv layers",
            "effect": "Mixed INT8/INT4 reduces remaining weight size",
            "weights_kb": 95,
            "activation_kb": 25,
            "accuracy": 0.77,
        },
    ]

    print(f"\n  Optimization pipeline:")
    print(f"  {'Step':<48} {'Flash':>7} {'SRAM':>7} {'Acc':>6} {'Fits?':>6}")
    print("  " + "-" * 78)

    for opt in optimizations:
        fits_f = opt['weights_kb'] <= target['available_flash_kb']
        fits_s = opt['activation_kb'] <= target['available_sram_kb']
        fits = "YES" if (fits_f and fits_s) else "NO"
        print(f"  {opt['step']:<48} {opt['weights_kb']:>5}KB "
              f"{opt['activation_kb']:>5}KB {opt['accuracy']:>5.0%} {fits:>6}")

    print(f"\n  Final model fits in target MCU:")
    final = optimizations[-1]
    print(f"    Flash: {final['weights_kb']} / {target['available_flash_kb']} KB "
          f"({final['weights_kb']/target['available_flash_kb']:.0%} used)")
    print(f"    SRAM:  {final['activation_kb']} / {target['available_sram_kb']} KB "
          f"({final['activation_kb']/target['available_sram_kb']:.0%} used)")
    print(f"    Accuracy: {original['accuracy']:.0%} -> {final['accuracy']:.0%} "
          f"({(original['accuracy'] - final['accuracy'])*100:.0f}% drop)")


if __name__ == "__main__":
    print("=== Exercise 1: MCU Memory Constraints ===")
    exercise_1()
    print("\n=== Exercise 2: TFLite Micro Memory Layout ===")
    exercise_2()
    print("\n=== Exercise 3: CMSIS-NN Operator Mapping ===")
    exercise_3()
    print("\n=== Exercise 4: TinyML Model Optimization ===")
    exercise_4()
    print("\nAll exercises completed!")
