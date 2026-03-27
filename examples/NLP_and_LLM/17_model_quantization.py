"""
13. Model Quantization Example

INT8/INT4 quantization, bitsandbytes, GPTQ, AWQ practice
"""

import numpy as np

print("=" * 60)
print("Model Quantization")
print("=" * 60)


# ============================================
# 1. Basic Quantization Concepts
# ============================================
print("\n[1] Basic Quantization Concepts")
print("-" * 40)

def quantize_symmetric(tensor, bits=8):
    """Symmetric Quantization"""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    # Compute scale
    abs_max = np.abs(tensor).max()
    scale = abs_max / qmax if abs_max != 0 else 1.0

    # Quantize
    quantized = np.round(tensor / scale).astype(np.int8)
    quantized = np.clip(quantized, qmin, qmax)

    return quantized, scale

def dequantize(quantized, scale):
    """Dequantize"""
    return quantized.astype(np.float32) * scale


# Test
original = np.array([0.5, -1.2, 0.3, 2.1, -0.8, 0.0], dtype=np.float32)
print(f"Original tensor: {original}")

quantized, scale = quantize_symmetric(original, bits=8)
print(f"Quantized (INT8): {quantized}")
print(f"Scale: {scale:.6f}")

recovered = dequantize(quantized, scale)
print(f"Recovered: {recovered}")

error = np.abs(original - recovered).mean()
print(f"Mean quantization error: {error:.6f}")


# ============================================
# 2. Asymmetric Quantization
# ============================================
print("\n[2] Asymmetric Quantization")
print("-" * 40)

def quantize_asymmetric(tensor, bits=8):
    """Asymmetric Quantization"""
    qmin = 0
    qmax = 2 ** bits - 1

    min_val = tensor.min()
    max_val = tensor.max()

    scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0
    zero_point = round(-min_val / scale) if scale != 0 else 0

    quantized = np.round(tensor / scale + zero_point).astype(np.uint8)
    quantized = np.clip(quantized, qmin, qmax)

    return quantized, scale, zero_point

def dequantize_asymmetric(quantized, scale, zero_point):
    """Asymmetric dequantization"""
    return (quantized.astype(np.float32) - zero_point) * scale


# Test
asym_quantized, asym_scale, zero_point = quantize_asymmetric(original, bits=8)
print(f"Asymmetric quantized (UINT8): {asym_quantized}")
print(f"Scale: {asym_scale:.6f}, Zero Point: {zero_point}")

asym_recovered = dequantize_asymmetric(asym_quantized, asym_scale, zero_point)
print(f"Recovered: {asym_recovered}")


# ============================================
# 3. Group Quantization
# ============================================
print("\n[3] Group Quantization")
print("-" * 40)

def group_quantize(tensor, group_size=4, bits=4):
    """Group quantization - improved accuracy"""
    flat = tensor.flatten()
    pad_size = (group_size - len(flat) % group_size) % group_size
    if pad_size > 0:
        flat = np.pad(flat, (0, pad_size))

    groups = flat.reshape(-1, group_size)
    quantized_groups = []
    scales = []

    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))

    for group in groups:
        abs_max = np.abs(group).max()
        scale = abs_max / qmax if abs_max != 0 else 1.0
        q = np.round(group / scale).astype(np.int8)
        q = np.clip(q, qmin, qmax)
        quantized_groups.append(q)
        scales.append(scale)

    return np.array(quantized_groups), np.array(scales)

def group_dequantize(quantized_groups, scales):
    """Group dequantization"""
    recovered = []
    for q, s in zip(quantized_groups, scales):
        recovered.append(q.astype(np.float32) * s)
    return np.concatenate(recovered)


# Test
larger_tensor = np.random.randn(16).astype(np.float32)
print(f"Original (16 values): {larger_tensor[:8]}...")

g_quantized, g_scales = group_quantize(larger_tensor, group_size=4, bits=4)
print(f"Number of groups: {len(g_scales)}, group size: 4")
print(f"Scales: {g_scales}")

g_recovered = group_dequantize(g_quantized, g_scales)
g_error = np.abs(larger_tensor - g_recovered).mean()
print(f"Group quantization mean error: {g_error:.6f}")


# ============================================
# 4. Bit Precision Comparison
# ============================================
print("\n[4] Bit Precision Comparison")
print("-" * 40)

def compare_bit_precision(tensor):
    """Compare various bit precisions"""
    results = {}

    for bits in [8, 4, 2]:
        q, s = quantize_symmetric(tensor, bits=bits)
        r = dequantize(q, s)
        error = np.abs(tensor - r).mean()
        results[f"INT{bits}"] = {
            "error": error,
            "range": (-(2**(bits-1)), 2**(bits-1)-1)
        }

    return results

comparison = compare_bit_precision(original)
print("Quantization comparison by bit width:")
for name, result in comparison.items():
    print(f"  {name}: error={result['error']:.6f}, range={result['range']}")


# ============================================
# 5. bitsandbytes Example (code only)
# ============================================
print("\n[5] bitsandbytes Usage (code example)")
print("-" * 40)

bnb_code = '''
# bitsandbytes 8-bit quantization
from transformers import AutoModelForCausalLM, AutoTokenizer

model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)

# bitsandbytes 4-bit quantization (NF4)
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Normal Float 4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True       # Double quantization
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"4bit model memory: {model_4bit.get_memory_footprint() / 1e9:.2f} GB")
'''
print(bnb_code)


# ============================================
# 6. GPTQ Example (code only)
# ============================================
print("\n[6] GPTQ Quantization (code example)")
print("-" * 40)

gptq_code = '''
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# GPTQ configuration
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    dataset=calibration_data,
    tokenizer=tokenizer
)

# Quantize
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=gptq_config,
    device_map="auto"
)

model.save_pretrained("./llama-2-7b-gptq-4bit")

# Load pre-quantized model
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto"
)
'''
print(gptq_code)


# ============================================
# 7. AWQ Example (code only)
# ============================================
print("\n[7] AWQ Quantization (code example)")
print("-" * 40)

awq_code = '''
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load model
model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# AWQ quantization configuration
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Quantize
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("./llama-2-7b-awq")

# AWQ model inference
model = AutoAWQForCausalLM.from_quantized(
    "./llama-2-7b-awq",
    fuse_layers=True  # Layer fusion for speed improvement
)
'''
print(awq_code)


# ============================================
# 8. QLoRA Example (code only)
# ============================================
print("\n[8] QLoRA Fine-tuning (code example)")
print("-" * 40)

qlora_code = '''
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: ~0.1%
'''
print(qlora_code)


# ============================================
# 9. Quantization Memory Savings Simulation
# ============================================
print("\n[9] Quantization Memory Savings Simulation")
print("-" * 40)

def estimate_model_size(params_billions, bits):
    """Estimate model size (GB)"""
    bytes_per_param = bits / 8
    size_gb = params_billions * 1e9 * bytes_per_param / (1024**3)
    return size_gb

model_sizes = {
    "7B": 7,
    "13B": 13,
    "70B": 70,
}

precisions = {
    "FP32": 32,
    "FP16": 16,
    "INT8": 8,
    "INT4": 4,
}

print("Estimated model size (GB):")
print("-" * 60)
header = "Model\t" + "\t".join(precisions.keys())
print(header)
print("-" * 60)

for model_name, params in model_sizes.items():
    sizes = [f"{estimate_model_size(params, bits):.1f}" for bits in precisions.values()]
    print(f"{model_name}\t" + "\t".join(sizes))


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Quantization Summary")
print("=" * 60)

summary = """
Quantization Key Concepts:

1. Symmetric Quantization:
   - scale = max(|x|) / (2^(bits-1) - 1)
   - x_q = round(x / scale)
   - x' = x_q * scale

2. Asymmetric Quantization:
   - scale = (max - min) / (2^bits - 1)
   - zero_point = round(-min / scale)
   - x_q = round(x / scale + zero_point)

3. Quantization Method Comparison:
   - bitsandbytes: Fast application, dynamic quantization
   - GPTQ: High quality, calibration required
   - AWQ: Fast quantization, activation-based
   - QLoRA: Quantization + LoRA fine-tuning

4. Selection Guide:
   - Prototyping: bitsandbytes (load_in_8bit)
   - Memory-constrained: bitsandbytes (load_in_4bit)
   - Production: GPTQ or AWQ
   - Fine-tuning: QLoRA
"""
print(summary)
