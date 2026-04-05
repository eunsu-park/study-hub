# 15. NLP를 위한 Edge AI

**이전**: [컴퓨터 비전을 위한 Edge AI](./14_Edge_AI_for_Computer_Vision.md) | **다음**: [배포와 모니터링](./16_Deployment_and_Monitoring.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 양자화를 활용하여 경량 LLM(TinyLlama, Phi, Gemma)을 엣지 디바이스에 배포할 수 있다
2. 모델 증류와 양자화를 통해 온디바이스 음성 인식을 위한 Whisper를 최적화할 수 있다
3. 효율적인 트랜스포머 대안을 사용하여 엣지 디바이스에서 텍스트 분류를 실행할 수 있다
4. 최소 전력으로 항시 작동하는 음성 활성화를 위한 keyword spotting을 구현할 수 있다
5. 오프라인 언어 번역을 위한 온디바이스 번역 모델을 배포할 수 있다
6. 제한된 하드웨어에서 NLP 작업에 적합한 모델 크기와 양자화를 선택할 수 있다

---

엣지 디바이스에서의 자연어 처리는 극적으로 발전했습니다. 과거에는 클라우드 API가 필요했던 작업들(음성 인식, 텍스트 분류, 대화형 AI까지)이 이제는 스마트폰과 임베디드 디바이스에서 로컬로 실행될 수 있습니다. 핵심 동력은 더 작은 모델 아키텍처(증류된 트랜스포머, 효율적인 어텐션 메커니즘), 공격적인 양자화(4비트 및 8비트), 전문화된 추론 런타임(llama.cpp, MLC-LLM)입니다. 이 레슨에서는 초소형 keyword spotter부터 수십억 파라미터 언어 모델까지, NLP 워크로드를 엣지에 배포하는 실용적인 기법들을 다룹니다.

---

## 1. 엣지에서의 경량 LLM

### 1.1 엣지 LLM 현황

```
+-----------------------------------------------------------------+
|           Lightweight LLMs for Edge Deployment                   |
+-----------------------------------------------------------------+
|                                                                   |
|   Model           Params    MMLU    RAM (Q4)   Tokens/s*         |
|   +----------------------------------------------------------+  |
|   | TinyLlama-1.1B   1.1B   25.4    ~1.0 GB    25-40 t/s    |  |
|   | Phi-2             2.7B   56.3    ~2.0 GB    15-25 t/s    |  |
|   | Phi-3-mini        3.8B   68.8    ~2.5 GB    10-20 t/s    |  |
|   | Gemma-2B          2.5B   51.4    ~1.8 GB    18-28 t/s    |  |
|   | Gemma-2-2B        2.6B   56.1    ~1.9 GB    16-26 t/s    |  |
|   | Llama-3.2-1B      1.2B   32.2    ~1.1 GB    22-35 t/s    |  |
|   | Llama-3.2-3B      3.2B   58.0    ~2.2 GB    12-20 t/s    |  |
|   | Qwen2.5-0.5B      0.5B   19.8    ~0.5 GB    40-60 t/s    |  |
|   | Qwen2.5-1.5B      1.5B   42.5    ~1.2 GB    20-32 t/s    |  |
|   | SmolLM2-1.7B      1.7B   35.8    ~1.3 GB    20-30 t/s    |  |
|   +----------------------------------------------------------+  |
|                                                                   |
|   * Tokens/sec on Apple M2 or Snapdragon 8 Gen 3 (Q4_K_M)      |
|   * RAM estimate for 4-bit quantized model + KV cache            |
|                                                                   |
+-----------------------------------------------------------------+
```

### 1.2 llama.cpp로 LLM 실행

```bash
# llama.cpp: C/C++ inference engine for LLMs on edge
# Supports: ARM NEON, AVX2, Metal, CUDA, Vulkan

# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j

# Download a GGUF model (pre-quantized)
# Example: TinyLlama-1.1B in Q4_K_M format (~700 MB)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Run inference
./llama-cli \
    -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -p "Explain edge AI in one paragraph:" \
    -n 128 \
    --threads 4 \
    --ctx-size 2048

# Benchmark
./llama-bench \
    -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -t 4 \
    -ngl 0  # 0 = CPU only, >0 = GPU layers
```

### 1.3 llama-cpp-python을 통한 Python 통합

```python
#!/usr/bin/env python3
"""Run edge LLMs using llama-cpp-python."""

from llama_cpp import Llama
import time


class EdgeLLM:
    """Lightweight LLM wrapper for edge inference.

    Uses GGUF format models with quantization for minimal memory.
    Q4_K_M is the recommended quantization level -- it provides
    a good balance between quality and size (~4.5 bits/weight).
    """

    def __init__(self, model_path: str,
                 n_ctx: int = 2048,
                 n_threads: int = 4,
                 n_gpu_layers: int = 0):
        """
        Args:
            model_path: Path to .gguf model file
            n_ctx: Context window size (affects RAM usage)
            n_threads: CPU threads for inference
            n_gpu_layers: Layers to offload to GPU (0=CPU only)
        """
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.model_path = model_path

    def generate(self, prompt: str, max_tokens: int = 128,
                 temperature: float = 0.7) -> dict:
        """Generate text with performance metrics."""
        start = time.perf_counter()

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "\n\n"],
        )

        elapsed = time.perf_counter() - start
        text = output["choices"][0]["text"]
        tokens_generated = output["usage"]["completion_tokens"]
        tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

        return {
            "text": text,
            "tokens": tokens_generated,
            "time_s": round(elapsed, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "prompt_tokens": output["usage"]["prompt_tokens"],
        }

    def benchmark(self, prompt: str = "The meaning of life is",
                  max_tokens: int = 100,
                  num_runs: int = 5) -> dict:
        """Benchmark generation speed."""
        speeds = []
        for _ in range(num_runs):
            result = self.generate(prompt, max_tokens=max_tokens,
                                   temperature=0.0)
            speeds.append(result["tokens_per_sec"])

        import numpy as np
        return {
            "mean_tokens_per_sec": round(float(np.mean(speeds)), 1),
            "std_tokens_per_sec": round(float(np.std(speeds)), 1),
            "time_to_first_token_ms": "N/A",  # Requires streaming
        }


if __name__ == "__main__":
    llm = EdgeLLM(
        "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=4
    )

    result = llm.generate(
        "<|system|>\nYou are a helpful assistant.</s>\n"
        "<|user|>\nWhat is edge AI?</s>\n"
        "<|assistant|>\n",
        max_tokens=100
    )

    print(f"Response: {result['text']}")
    print(f"Speed: {result['tokens_per_sec']} tokens/sec")
    print(f"Time: {result['time_s']}s")
```

### 1.4 양자화 형식 비교

```python
#!/usr/bin/env python3
"""Compare GGUF quantization levels for edge deployment."""

quantization_formats = {
    "Q2_K": {
        "bits_per_weight": 2.5,
        "quality_loss": "High (noticeable degradation)",
        "size_factor": 0.31,
        "use_case": "Extreme memory constraints (<512 MB RAM)",
    },
    "Q3_K_M": {
        "bits_per_weight": 3.4,
        "quality_loss": "Moderate",
        "size_factor": 0.42,
        "use_case": "Tight memory (512 MB - 1 GB)",
    },
    "Q4_K_M": {
        "bits_per_weight": 4.5,
        "quality_loss": "Small (recommended default)",
        "size_factor": 0.56,
        "use_case": "General edge deployment (1-4 GB RAM)",
    },
    "Q5_K_M": {
        "bits_per_weight": 5.5,
        "quality_loss": "Very small",
        "size_factor": 0.69,
        "use_case": "Quality-sensitive applications",
    },
    "Q6_K": {
        "bits_per_weight": 6.5,
        "quality_loss": "Negligible",
        "size_factor": 0.81,
        "use_case": "Maximum quality with some compression",
    },
    "Q8_0": {
        "bits_per_weight": 8.0,
        "quality_loss": "None (practically lossless)",
        "size_factor": 1.0,
        "use_case": "Baseline quantization",
    },
}

# Calculate sizes for a 3B parameter model
base_size_gb = 3.0 * 2  # FP16 baseline = 6 GB

print(f"{'Format':<10} {'Bits':>6} {'Size (3B)':>12} {'Quality Loss':<35}")
print("-" * 70)
for fmt, info in quantization_formats.items():
    size_gb = base_size_gb * info["size_factor"]
    print(f"{fmt:<10} {info['bits_per_weight']:>6.1f} "
          f"{size_gb:>10.2f} GB  {info['quality_loss']:<35}")
```

---

## 2. 온디바이스 음성 인식

### 2.1 엣지에서의 Whisper

```
+-----------------------------------------------------------------+
|           Whisper Model Sizes for Edge                            |
+-----------------------------------------------------------------+
|                                                                   |
|   Model         Params    Size(FP16)  WER(en)  Latency*          |
|   +----------------------------------------------------------+  |
|   | whisper-tiny    39M      78 MB     7.6%     ~2x realtime |  |
|   | whisper-base    74M     148 MB     5.0%     ~1.5x RT     |  |
|   | whisper-small  244M     488 MB     3.4%     ~0.5x RT     |  |
|   | whisper-medium 769M    1.53 GB     2.9%     ~0.2x RT     |  |
|   +----------------------------------------------------------+  |
|                                                                   |
|   * Latency relative to audio duration on Apple M2 (FP16)       |
|   * WER = Word Error Rate on LibriSpeech test-clean              |
|                                                                   |
|   For edge: whisper-tiny or whisper-base with INT8 quantization  |
|   Distilled variants (distil-whisper) are 2x faster              |
|                                                                   |
+-----------------------------------------------------------------+
```

### 2.2 엣지를 위한 Whisper 최적화

```python
#!/usr/bin/env python3
"""Optimized Whisper inference on edge devices."""

import numpy as np
import time


class EdgeWhisper:
    """Optimized Whisper for on-device speech recognition.

    Optimization strategies:
    1. Use whisper-tiny or distil-whisper (smaller model)
    2. INT8 quantization via CTranslate2 or ONNX
    3. Chunked processing for streaming audio
    4. VAD (Voice Activity Detection) to skip silence
    """

    def __init__(self, model_size: str = "tiny",
                 compute_type: str = "int8"):
        """
        Args:
            model_size: "tiny", "base", "small"
            compute_type: "int8", "float16", "float32"
        """
        import faster_whisper

        self.model = faster_whisper.WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=4,
        )
        self.sample_rate = 16000

    def transcribe(self, audio_path: str,
                   language: str = "en") -> dict:
        """Transcribe audio file."""
        start = time.perf_counter()

        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=1,         # Greedy decoding (fastest)
            best_of=1,
            vad_filter=True,     # Skip silence segments
            vad_parameters={
                "min_silence_duration_ms": 500,
            },
        )

        # Collect segments
        text_segments = []
        for segment in segments:
            text_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            })

        elapsed = time.perf_counter() - start
        full_text = " ".join(s["text"] for s in text_segments)

        return {
            "text": full_text,
            "segments": text_segments,
            "language": info.language,
            "duration_s": info.duration,
            "processing_time_s": round(elapsed, 2),
            "realtime_factor": round(elapsed / info.duration, 2)
                               if info.duration > 0 else 0,
        }

    def transcribe_streaming(self, audio_chunks: list,
                             chunk_duration_s: float = 5.0) -> list:
        """Process audio in chunks for near-realtime transcription.

        For streaming applications, process overlapping chunks and
        stitch transcriptions together. Use a 0.5s overlap to
        avoid cutting words at chunk boundaries.
        """
        results = []
        for i, chunk in enumerate(audio_chunks):
            # Save chunk to temp file (faster-whisper requires file/array)
            import tempfile, soundfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                soundfile.write(f.name, chunk, self.sample_rate)
                result = self.transcribe(f.name)
                results.append(result)
                print(f"Chunk {i}: [{result['processing_time_s']}s] "
                      f"{result['text'][:50]}...")

        return results


if __name__ == "__main__":
    whisper = EdgeWhisper(model_size="tiny", compute_type="int8")

    # Benchmark
    result = whisper.transcribe("test_audio.wav")
    print(f"Text: {result['text']}")
    print(f"Duration: {result['duration_s']}s")
    print(f"Processing: {result['processing_time_s']}s")
    print(f"Realtime factor: {result['realtime_factor']}x")
```

---

## 3. 엣지에서의 텍스트 분류

### 3.1 효율적인 텍스트 분류 모델

```
+-----------------------------------------------------------------+
|        Text Classification Models for Edge                       |
+-----------------------------------------------------------------+
|                                                                   |
|   Approach              Size      Accuracy    Latency            |
|   +----------------------------------------------------------+  |
|   | TF-IDF + Linear     <1 MB     85-90%      <1ms           |  |
|   | FastText             5-50 MB   88-92%      <1ms           |  |
|   | DistilBERT (INT8)    66 MB     91-95%      15-30ms        |  |
|   | MobileBERT (INT8)    25 MB     90-93%      10-20ms        |  |
|   | TinyBERT-4L          14 MB     89-92%      8-15ms         |  |
|   | ALBERT-tiny          5 MB      85-88%      5-10ms         |  |
|   +----------------------------------------------------------+  |
|                                                                   |
|   Rule of thumb:                                                 |
|   - <5ms budget: TF-IDF or FastText                             |
|   - 5-20ms budget: MobileBERT or TinyBERT                       |
|   - 20-50ms budget: DistilBERT                                  |
|                                                                   |
+-----------------------------------------------------------------+
```

### 3.2 TFLite 텍스트 분류기

```python
#!/usr/bin/env python3
"""Deploy a text classifier on edge using TFLite."""

import numpy as np
import time
from typing import List

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


class EdgeTextClassifier:
    """Text classifier using quantized DistilBERT/MobileBERT on TFLite."""

    def __init__(self, model_path: str, vocab_path: str,
                 labels: List[str], max_length: int = 128):
        self.max_length = max_length
        self.labels = labels

        # Load tokenizer vocabulary
        self.vocab = self._load_vocab(vocab_path)

        # Load TFLite model
        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=4
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _load_vocab(self, vocab_path: str) -> dict:
        """Load WordPiece vocabulary."""
        vocab = {}
        with open(vocab_path, "r") as f:
            for idx, line in enumerate(f):
                token = line.strip()
                vocab[token] = idx
        return vocab

    def _tokenize(self, text: str) -> dict:
        """Simple WordPiece tokenizer (production: use HF tokenizers).

        For edge deployment, consider pre-compiled tokenizers like
        SentencePiece or HuggingFace Tokenizers (Rust-based) which
        are significantly faster than Python implementations.
        """
        tokens = ["[CLS]"]
        for word in text.lower().split():
            if word in self.vocab:
                tokens.append(word)
            else:
                # Simple subword fallback
                tokens.append("[UNK]")
        tokens.append("[SEP]")

        # Convert to IDs
        input_ids = [self.vocab.get(t, self.vocab["[UNK]"]) for t in tokens]

        # Pad/truncate to max_length
        input_ids = input_ids[:self.max_length]
        attention_mask = [1] * len(input_ids)

        # Pad
        pad_len = self.max_length - len(input_ids)
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

        return {
            "input_ids": np.array([input_ids], dtype=np.int32),
            "attention_mask": np.array([attention_mask], dtype=np.int32),
        }

    def classify(self, text: str) -> dict:
        """Classify a text string."""
        tokens = self._tokenize(text)

        start = time.perf_counter()

        # Set inputs
        for detail in self.input_details:
            name = detail["name"]
            if "input_ids" in name:
                self.interpreter.set_tensor(
                    detail["index"], tokens["input_ids"]
                )
            elif "attention_mask" in name:
                self.interpreter.set_tensor(
                    detail["index"], tokens["attention_mask"]
                )

        self.interpreter.invoke()

        logits = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )[0]

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        predicted_idx = probs.argmax()

        return {
            "label": self.labels[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "all_scores": {
                self.labels[i]: float(probs[i])
                for i in range(len(self.labels))
            },
            "latency_ms": round(elapsed_ms, 2),
        }
```

### 3.3 초고속 분류를 위한 FastText

```python
#!/usr/bin/env python3
"""FastText: sub-millisecond text classification on any edge device."""

import numpy as np
import time
from collections import defaultdict


class LightweightClassifier:
    """Bag-of-ngrams classifier (FastText-style) for edge.

    FastText uses character and word n-grams hashed into a fixed-size
    embedding table. This makes it:
    - Extremely fast (<1ms on any CPU)
    - Small (compressible to <5 MB)
    - Handles out-of-vocabulary words via character n-grams
    """

    def __init__(self, num_classes: int, embedding_dim: int = 64,
                 hash_buckets: int = 100000):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hash_buckets = hash_buckets

        # Embedding table (shared for all n-grams)
        self.embeddings = np.random.randn(
            hash_buckets, embedding_dim
        ).astype(np.float32) * 0.01

        # Classification weight matrix
        self.weights = np.random.randn(
            embedding_dim, num_classes
        ).astype(np.float32) * 0.01

    def _extract_ngrams(self, text: str, n_range: tuple = (2, 4)) -> list:
        """Extract character n-grams and word unigrams."""
        ngrams = []

        # Word unigrams
        words = text.lower().split()
        ngrams.extend(words)

        # Character n-grams
        padded = f"<{text.lower()}>"
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(padded) - n + 1):
                ngrams.append(padded[i:i+n])

        return ngrams

    def _hash_ngrams(self, ngrams: list) -> np.ndarray:
        """Hash n-grams to embedding indices."""
        indices = []
        for ng in ngrams:
            h = hash(ng) % self.hash_buckets
            indices.append(h)
        return np.array(indices)

    def predict(self, text: str) -> dict:
        """Classify text in <1ms."""
        start = time.perf_counter()

        ngrams = self._extract_ngrams(text)
        indices = self._hash_ngrams(ngrams)

        # Average n-gram embeddings
        embedding = self.embeddings[indices].mean(axis=0)

        # Linear classifier
        logits = embedding @ self.weights

        # Softmax
        exp_l = np.exp(logits - logits.max())
        probs = exp_l / exp_l.sum()

        elapsed_us = (time.perf_counter() - start) * 1e6

        return {
            "class_id": int(probs.argmax()),
            "confidence": float(probs.max()),
            "latency_us": round(elapsed_us, 1),
        }


if __name__ == "__main__":
    classifier = LightweightClassifier(num_classes=4)

    texts = [
        "The stock market rallied today on strong earnings",
        "Scientists discover new species in deep ocean",
        "Local team wins championship in overtime thriller",
        "New smartphone features improved camera technology",
    ]

    for text in texts:
        result = classifier.predict(text)
        print(f"[{result['latency_us']:.0f} us] class={result['class_id']}, "
              f"conf={result['confidence']:.3f}: {text[:50]}...")
```

---

## 4. Keyword Spotting

### 4.1 Keyword Spotting 아키텍처

```
+-----------------------------------------------------------------+
|           Keyword Spotting Pipeline                               |
+-----------------------------------------------------------------+
|                                                                   |
|   Audio Input (microphone)                                       |
|       |                                                          |
|       v                                                          |
|   +------------------+                                           |
|   | Audio Buffer     |  Ring buffer: 1-2 seconds                 |
|   | (16 kHz, 16-bit) |                                           |
|   +--------+---------+                                           |
|            |                                                     |
|            v                                                     |
|   +------------------+                                           |
|   | Feature Extract  |  MFCC or Mel spectrogram                  |
|   | (40 Mel bins)    |  ~1ms on CPU                              |
|   +--------+---------+                                           |
|            |                                                     |
|            v                                                     |
|   +------------------+                                           |
|   | KWS Model        |  Small CNN or DS-CNN                      |
|   | (20-250 KB)      |  ~1-5ms on CPU                            |
|   +--------+---------+                                           |
|            |                                                     |
|            v                                                     |
|   Keyword detected? ----Yes----> Wake up main system             |
|                      \                                           |
|                       No----> Continue listening (low power)     |
|                                                                   |
|   Power budget: <1 mW (always-on listening)                      |
|   Models: DS-CNN, SVDF, TC-ResNet                                |
|                                                                   |
+-----------------------------------------------------------------+
```

### 4.2 Keyword Spotter 구현

```python
#!/usr/bin/env python3
"""Keyword spotting for edge devices (always-on voice activation)."""

import numpy as np
import time

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


class KeywordSpotter:
    """Lightweight keyword spotter using Mel spectrogram + small CNN.

    Designed for always-on operation:
    - Model size: 20-250 KB
    - Inference: 1-5 ms
    - Runs on MCU (Cortex-M4+) or as background task on Linux
    """

    def __init__(self, model_path: str, keywords: list,
                 sample_rate: int = 16000,
                 window_size_ms: int = 30,
                 window_stride_ms: int = 10,
                 num_mel_bins: int = 40,
                 detection_threshold: float = 0.8):
        self.keywords = keywords
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * window_size_ms / 1000)
        self.window_stride = int(sample_rate * window_stride_ms / 1000)
        self.num_mel_bins = num_mel_bins
        self.threshold = detection_threshold

        # Load model
        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=1  # Single thread for low power
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Pre-compute Mel filterbank
        self.mel_filterbank = self._create_mel_filterbank()

        # Sliding window buffer
        self.audio_buffer = np.zeros(sample_rate, dtype=np.float32)

    def _create_mel_filterbank(self) -> np.ndarray:
        """Create Mel-scale triangular filterbank."""
        num_fft = self.window_size
        low_freq = 20
        high_freq = self.sample_rate // 2

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_low = hz_to_mel(low_freq)
        mel_high = hz_to_mel(high_freq)
        mel_points = np.linspace(mel_low, mel_high, self.num_mel_bins + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor(
            (num_fft + 1) * hz_points / self.sample_rate
        ).astype(int)

        filterbank = np.zeros((self.num_mel_bins, num_fft // 2 + 1))
        for i in range(self.num_mel_bins):
            for j in range(bin_points[i], bin_points[i + 1]):
                filterbank[i, j] = (
                    (j - bin_points[i]) /
                    (bin_points[i + 1] - bin_points[i] + 1e-8)
                )
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                filterbank[i, j] = (
                    (bin_points[i + 2] - j) /
                    (bin_points[i + 2] - bin_points[i + 1] + 1e-8)
                )

        return filterbank

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract Mel spectrogram features from audio."""
        num_frames = 1 + (len(audio) - self.window_size) // self.window_stride
        features = np.zeros((num_frames, self.num_mel_bins))

        window = np.hanning(self.window_size)

        for i in range(num_frames):
            start = i * self.window_stride
            frame = audio[start:start + self.window_size] * window

            # FFT
            spectrum = np.abs(np.fft.rfft(frame)) ** 2

            # Apply Mel filterbank
            mel_spectrum = self.mel_filterbank @ spectrum

            # Log scale
            features[i] = np.log(mel_spectrum + 1e-8)

        return features

    def detect(self, audio_chunk: np.ndarray) -> dict:
        """Check for keyword in an audio chunk."""
        features = self.extract_features(audio_chunk)

        # Reshape for model input
        input_shape = self.input_details[0]["shape"]
        features_resized = np.zeros(input_shape[1:], dtype=np.float32)
        h = min(features.shape[0], features_resized.shape[0])
        w = min(features.shape[1], features_resized.shape[1])
        features_resized[:h, :w] = features[:h, :w]

        input_data = np.expand_dims(features_resized, 0).astype(
            self.input_details[0]["dtype"]
        )

        # Inference
        start = time.perf_counter()
        self.interpreter.set_tensor(
            self.input_details[0]["index"], input_data
        )
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )[0]
        latency_ms = (time.perf_counter() - start) * 1000

        # Check detection
        max_idx = output.argmax()
        max_score = float(output[max_idx])

        detected = max_score > self.threshold and max_idx < len(self.keywords)

        return {
            "detected": detected,
            "keyword": self.keywords[max_idx] if detected else None,
            "score": max_score,
            "latency_ms": round(latency_ms, 2),
            "all_scores": {
                self.keywords[i]: float(output[i])
                for i in range(min(len(self.keywords), len(output)))
            },
        }
```

---

## 5. 엣지 번역

### 5.1 온디바이스 번역 모델

```
+-----------------------------------------------------------------+
|           On-Device Translation Options                          |
+-----------------------------------------------------------------+
|                                                                   |
|   Model                  Params   Size(Q8)   Quality    Speed    |
|   +----------------------------------------------------------+  |
|   | NLLB-200 (600M)       600M    ~600 MB    High       Slow   |  |
|   | NLLB-200-distilled    600M    ~300 MB    Good      Medium  |  |
|   | Opus-MT (per pair)   ~40M     ~80 MB     Good       Fast   |  |
|   | MarianMT (per pair)  ~40M     ~80 MB     Good       Fast   |  |
|   | T5-small (fine-tuned) 60M     ~120 MB    Moderate   Fast   |  |
|   | CTranslate2 (any)    varies   ~50% size  Same      2-4x   |  |
|   +----------------------------------------------------------+  |
|                                                                   |
|   For edge: Opus-MT or MarianMT per language pair with           |
|   CTranslate2 quantization gives the best size/speed/quality.   |
|                                                                   |
+-----------------------------------------------------------------+
```

### 5.2 CTranslate2를 활용한 엣지 번역

```python
#!/usr/bin/env python3
"""On-device translation using CTranslate2."""

import time


class EdgeTranslator:
    """Offline translation using quantized seq2seq models.

    CTranslate2 provides:
    - INT8 quantization (2-4x faster, 50% smaller)
    - Efficient beam search implementation
    - CPU optimized (AVX, NEON)
    - No Python dependencies at runtime (C++ core)
    """

    def __init__(self, model_dir: str, device: str = "cpu",
                 compute_type: str = "int8"):
        """
        Args:
            model_dir: CTranslate2 model directory
            device: "cpu" or "cuda"
            compute_type: "int8", "int8_float16", "float16", "float32"
        """
        import ctranslate2
        import sentencepiece

        self.translator = ctranslate2.Translator(
            model_dir,
            device=device,
            compute_type=compute_type,
            inter_threads=1,
            intra_threads=4,
        )

        # Load SentencePiece tokenizer
        sp_model = f"{model_dir}/source.spm"
        self.tokenizer = sentencepiece.SentencePieceProcessor()
        self.tokenizer.Load(sp_model)

        # Target tokenizer (may differ for some models)
        tgt_sp = f"{model_dir}/target.spm"
        self.tgt_tokenizer = sentencepiece.SentencePieceProcessor()
        self.tgt_tokenizer.Load(tgt_sp)

    def translate(self, text: str, beam_size: int = 2,
                  max_length: int = 200) -> dict:
        """Translate a single sentence."""
        start = time.perf_counter()

        # Tokenize
        tokens = self.tokenizer.Encode(text, out_type=str)

        # Translate
        results = self.translator.translate_batch(
            [tokens],
            beam_size=beam_size,
            max_decoding_length=max_length,
        )

        # Detokenize
        output_tokens = results[0].hypotheses[0]
        translated = self.tgt_tokenizer.Decode(output_tokens)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "source": text,
            "translation": translated,
            "latency_ms": round(elapsed_ms, 2),
            "source_tokens": len(tokens),
            "target_tokens": len(output_tokens),
        }

    def translate_batch(self, texts: list, beam_size: int = 2) -> list:
        """Batch translation for higher throughput."""
        start = time.perf_counter()

        all_tokens = [self.tokenizer.Encode(t, out_type=str) for t in texts]

        results = self.translator.translate_batch(
            all_tokens, beam_size=beam_size
        )

        translations = []
        for r in results:
            output_tokens = r.hypotheses[0]
            translations.append(self.tgt_tokenizer.Decode(output_tokens))

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "translations": translations,
            "total_latency_ms": round(elapsed_ms, 2),
            "avg_latency_ms": round(elapsed_ms / len(texts), 2),
        }


if __name__ == "__main__":
    translator = EdgeTranslator(
        "opus-mt-en-de-ct2",  # English -> German
        compute_type="int8"
    )

    result = translator.translate("Edge AI enables intelligent devices everywhere.")
    print(f"EN: {result['source']}")
    print(f"DE: {result['translation']}")
    print(f"Latency: {result['latency_ms']} ms")
```

---

## 6. 모델 선택 가이드

### 6.1 NLP 작업 결정 매트릭스

```python
#!/usr/bin/env python3
"""Decision guide for choosing edge NLP models."""

nlp_task_guide = {
    "keyword_spotting": {
        "budget_ms": "<5",
        "model_size": "<250 KB",
        "recommended": ["DS-CNN", "SVDF", "TC-ResNet"],
        "framework": "TFLite Micro / TFLite",
        "hardware": "MCU (Cortex-M4+) or any",
        "notes": "Always-on, must be extremely lightweight",
    },
    "text_classification": {
        "budget_ms": "<10",
        "model_size": "<50 MB",
        "recommended": ["FastText", "TinyBERT", "MobileBERT"],
        "framework": "TFLite / ONNX Runtime",
        "hardware": "Any (even RPi Zero)",
        "notes": "FastText for <1ms, BERT variants for higher accuracy",
    },
    "speech_recognition": {
        "budget_ms": "Realtime (1x audio duration)",
        "model_size": "80-500 MB",
        "recommended": ["whisper-tiny (INT8)", "distil-whisper"],
        "framework": "faster-whisper / whisper.cpp",
        "hardware": "RPi 4+, smartphone, Jetson",
        "notes": "VAD preprocessing critical for efficiency",
    },
    "translation": {
        "budget_ms": "<500 per sentence",
        "model_size": "80-300 MB",
        "recommended": ["Opus-MT + CTranslate2", "MarianMT"],
        "framework": "CTranslate2",
        "hardware": "RPi 4+, smartphone",
        "notes": "Per-language-pair models are smaller and faster",
    },
    "conversational_ai": {
        "budget_ms": "<100 per token",
        "model_size": "500 MB - 4 GB",
        "recommended": ["Phi-3-mini (Q4)", "Llama-3.2-1B", "Qwen2.5-1.5B"],
        "framework": "llama.cpp / MLC-LLM",
        "hardware": "4+ GB RAM, smartphone/laptop",
        "notes": "Q4_K_M quantization recommended for quality/size balance",
    },
}

for task, info in nlp_task_guide.items():
    print(f"\n{'='*50}")
    print(f"Task: {task}")
    print(f"{'='*50}")
    print(f"  Latency budget: {info['budget_ms']}")
    print(f"  Model size:     {info['model_size']}")
    print(f"  Recommended:    {', '.join(info['recommended'])}")
    print(f"  Framework:      {info['framework']}")
    print(f"  Hardware:       {info['hardware']}")
    print(f"  Notes:          {info['notes']}")
```

---

## 연습 문제

### 연습 1: 엣지 LLM
1. TinyLlama-1.1B를 Q4_K_M 및 Q8_0 GGUF 형식으로 다운로드하십시오
2. llama.cpp로 사용 중인 머신에서 tokens/sec를 벤치마킹하십시오
3. 5개의 프롬프트에서 양자화 수준 간 출력 품질을 비교하십시오

### 연습 2: Keyword Spotting
1. Google Speech Commands 데이터셋(12개 키워드)에서 간단한 keyword spotter를 훈련하십시오
2. INT8로 양자화하고 TFLite로 변환하십시오
3. 정확도, 모델 크기, 추론 지연 시간을 측정하십시오

### 연습 3: 온디바이스 음성 인식
1. tiny 모델(INT8)로 faster-whisper를 설정하십시오
2. 5개의 오디오 샘플을 녹음하고 WER(Word Error Rate)을 측정하십시오
3. tiny 모델과 base 모델의 처리 속도를 비교하십시오

---

**이전**: [컴퓨터 비전을 위한 Edge AI](./14_Edge_AI_for_Computer_Vision.md) | **다음**: [배포와 모니터링](./16_Deployment_and_Monitoring.md)
