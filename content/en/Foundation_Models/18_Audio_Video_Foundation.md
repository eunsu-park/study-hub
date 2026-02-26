# 18. Audio/Video Foundation Models

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Whisper architecture and describe how log-Mel spectrograms are processed by encoder-decoder Transformers for speech recognition and translation
2. Implement speech transcription and speaker diarization pipelines using Whisper and related audio foundation models
3. Describe the architecture of music generation models such as MusicGen and AudioCraft, including how they handle multi-stream audio tokens
4. Compare video understanding approaches (VideoMAE, Video-LLaMA) and explain how temporal information is encoded across frames
5. Design multimodal pipelines that integrate audio, video, and text foundation models for tasks such as automatic video captioning

---

## Overview

Foundation Models for the Audio and Video domains comprehensively handle various multimedia tasks including speech recognition, music generation, and video understanding.

---

## 1. Speech Foundation Models

### 1.1 Whisper

OpenAI's general-purpose speech recognition model:

```
Whisper Architecture:
┌─────────────────────────────────────────────┐
│  Audio Input (30-second segments)           │
│       ↓                                      │
│  Log-Mel Spectrogram (80 bins)              │
│       ↓                                      │
│  ┌──────────────────────┐                   │
│  │   Audio Encoder      │                   │
│  │   (Transformer)      │                   │
│  │   - Conv1d stem      │                   │
│  │   - Sinusoidal pos   │                   │
│  │   - N layers         │                   │
│  └──────────────────────┘                   │
│       ↓                                      │
│  Audio Features                              │
│       ↓                                      │
│  ┌──────────────────────┐                   │
│  │   Text Decoder       │                   │
│  │   (Transformer)      │                   │
│  │   - Cross-attention  │                   │
│  │   - Causal masking   │                   │
│  └──────────────────────┘                   │
│       ↓                                      │
│  Text Output (Transcription/Translation)    │
└─────────────────────────────────────────────┘

Model sizes:
- tiny:   39M params,  ~32x realtime
- base:   74M params,  ~16x realtime
- small:  244M params, ~6x realtime
- medium: 769M params, ~2x realtime
- large:  1.55B params, ~1x realtime
```

```python
import torch
import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Using OpenAI whisper
def transcribe_with_whisper():
    """Speech recognition with OpenAI Whisper"""
    model = whisper.load_model("base")

    # Basic transcription
    result = model.transcribe("audio.mp3")
    print(result["text"])

    # Language detection and translation
    result = model.transcribe(
        "audio.mp3",
        task="translate",  # Translate to English
        language=None,     # Auto-detect
        fp16=torch.cuda.is_available()
    )

    # With timestamps
    result = model.transcribe(
        "audio.mp3",
        word_timestamps=True
    )

    for segment in result["segments"]:
        print(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")

    return result


# Using HuggingFace Transformers
def transcribe_with_hf_whisper():
    """Using HuggingFace Whisper"""
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

    # Load audio (16kHz)
    import librosa
    audio, sr = librosa.load("audio.mp3", sr=16000)

    # Process input
    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    # Generate
    predicted_ids = model.generate(
        input_features,
        language="korean",
        task="transcribe"
    )

    # Decode
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return transcription


# Whisper Fine-tuning
class WhisperFineTuner:
    """Domain-specific Whisper Fine-tuning"""

    def __init__(self, model_name: str = "openai/whisper-small"):
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
            Seq2SeqTrainingArguments,
            Seq2SeqTrainer
        )

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

        # Freeze encoder (optional)
        for param in self.model.model.encoder.parameters():
            param.requires_grad = False

    def prepare_dataset(self, dataset):
        """Dataset preprocessing"""
        def prepare_example(example):
            audio = example["audio"]["array"]

            # Extract input features
            input_features = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features[0]

            # Tokenize labels
            labels = self.processor.tokenizer(
                example["transcription"]
            ).input_ids

            return {
                "input_features": input_features,
                "labels": labels
            }

        return dataset.map(prepare_example)

    def train(self, train_dataset, eval_dataset):
        """Run fine-tuning"""
        from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

        training_args = Seq2SeqTrainingArguments(
            output_dir="./whisper-finetuned",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=1e-5,
            warmup_steps=500,
            num_train_epochs=3,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            fp16=True,
            predict_with_generate=True,
            generation_max_length=225
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor.feature_extractor,
        )

        trainer.train()
```

### 1.2 Speech Synthesis (TTS)

```python
# VITS/Coqui TTS
def text_to_speech_coqui():
    """Speech synthesis with Coqui TTS"""
    from TTS.api import TTS

    # Multilingual TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    # Speech synthesis
    tts.tts_to_file(
        text="Hello, this is Foundation Model learning material.",
        file_path="output.wav",
        speaker_wav="reference_voice.wav",  # Voice cloning
        language="en"
    )


# Bark (Suno AI)
def text_to_speech_bark():
    """Speech synthesis with Bark (including non-verbal expressions)"""
    from transformers import AutoProcessor, BarkModel
    import scipy.io.wavfile as wavfile

    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")

    # Text (can include non-verbal expressions)
    text = "[laughs] Hello! This is amazing. [sighs]"

    inputs = processor(text, return_tensors="pt")
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()

    # Save
    wavfile.write("bark_output.wav", 24000, audio_array)
```

---

## 2. Audio Generation Models

### 2.1 AudioLM

Google's Audio Language Model:

```
AudioLM Structure:
┌────────────────────────────────────────────────────┐
│                   Audio Input                       │
│                       ↓                             │
│  ┌──────────────────────────────────────────────┐  │
│  │           Semantic Tokens (w2v-BERT)          │  │
│  │           - High-level content               │  │
│  │           - ~25 tokens/second                │  │
│  └──────────────────────────────────────────────┘  │
│                       ↓                             │
│  ┌──────────────────────────────────────────────┐  │
│  │         Coarse Acoustic Tokens (SoundStream) │  │
│  │           - Medium-level details             │  │
│  │           - ~50 tokens/second                │  │
│  └──────────────────────────────────────────────┘  │
│                       ↓                             │
│  ┌──────────────────────────────────────────────┐  │
│  │          Fine Acoustic Tokens (SoundStream)  │  │
│  │           - Fine-grained details             │  │
│  │           - ~100 tokens/second               │  │
│  └──────────────────────────────────────────────┘  │
│                       ↓                             │
│                 SoundStream Decoder                 │
│                       ↓                             │
│                   Audio Output                      │
└────────────────────────────────────────────────────┘

3-stage generation:
1. Semantic → Semantic (continuation)
2. Semantic → Coarse Acoustic
3. Coarse → Fine Acoustic
```

### 2.2 MusicGen

Meta's music generation model:

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile

class MusicGenerator:
    """Music generation using MusicGen"""

    def __init__(self, model_size: str = "small"):
        """
        model_size: "small" (300M), "medium" (1.5B), "large" (3.3B)
        """
        model_name = f"facebook/musicgen-{model_size}"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def generate_from_text(
        self,
        prompt: str,
        duration: float = 10.0,
        temperature: float = 1.0,
        guidance_scale: float = 3.0
    ):
        """Generate music from text prompt"""
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Calculate token count (32kHz, 50 tokens/second)
        max_new_tokens = int(duration * 50)

        audio_values = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            guidance_scale=guidance_scale
        )

        return audio_values[0, 0].cpu().numpy()

    def generate_with_melody(
        self,
        prompt: str,
        melody_audio: torch.Tensor,
        duration: float = 10.0
    ):
        """Melody-conditioned generation (melody model only)"""
        inputs = self.processor(
            text=[prompt],
            audio=melody_audio,
            sampling_rate=32000,
            padding=True,
            return_tensors="pt"
        )

        max_new_tokens = int(duration * 50)

        audio_values = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

        return audio_values[0, 0].cpu().numpy()

    def save_audio(self, audio: np.ndarray, path: str):
        """Save audio"""
        wavfile.write(path, 32000, audio)


# Usage examples
def music_generation_examples():
    """Various music generation examples"""
    generator = MusicGenerator("small")

    # Text-based generation
    prompts = [
        "A calm piano melody with soft strings in the background",
        "Upbeat electronic dance music with heavy bass drops",
        "Traditional Korean music with gayageum and janggu",
        "Jazz trio improvisation with drums, bass, and piano"
    ]

    for i, prompt in enumerate(prompts):
        audio = generator.generate_from_text(
            prompt,
            duration=15.0,
            temperature=0.9,
            guidance_scale=3.5
        )
        generator.save_audio(audio, f"music_{i}.wav")
        print(f"Generated: {prompt[:50]}...")
```

### 2.3 AudioCraft (Audio Diffusion)

```python
# AudioGen (sound effects generation)
def generate_sound_effects():
    """Generate sound effects with AudioGen"""
    from audiocraft.models import AudioGen
    from audiocraft.data.audio import audio_write

    model = AudioGen.get_pretrained("facebook/audiogen-medium")
    model.set_generation_params(duration=5)

    descriptions = [
        "Dog barking in the distance",
        "Thunder and heavy rain",
        "Car engine starting and driving away"
    ]

    wav = model.generate(descriptions)

    for i, one_wav in enumerate(wav):
        audio_write(f"sound_{i}", one_wav.cpu(), model.sample_rate)
```

---

## 3. Video Understanding Models

### 3.1 Video-LLaMA / VideoLLaMA 2

```
VideoLLaMA Architecture:
┌─────────────────────────────────────────────────────┐
│  Video Input                                         │
│  [Frame1, Frame2, ..., FrameN]                      │
│          ↓                                           │
│  ┌────────────────────────────────────────────────┐ │
│  │         Visual Encoder (ViT/CLIP)              │ │
│  │         - Frame-level features                 │ │
│  └────────────────────────────────────────────────┘ │
│          ↓                                           │
│  ┌────────────────────────────────────────────────┐ │
│  │       Video Q-Former                           │ │
│  │       - Temporal aggregation                   │ │
│  │       - Cross-attention with queries           │ │
│  └────────────────────────────────────────────────┘ │
│          ↓                                           │
│  Video Embeddings                                    │
│          +                                           │
│  Audio Embeddings (ImageBind)                        │
│          ↓                                           │
│  ┌────────────────────────────────────────────────┐ │
│  │              LLM Backbone                       │ │
│  │           (LLaMA/Vicuna)                        │ │
│  └────────────────────────────────────────────────┘ │
│          ↓                                           │
│  Text Response                                       │
└─────────────────────────────────────────────────────┘
```

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

class VideoUnderstanding:
    """Video understanding model"""

    def __init__(self, model_name: str = "DAMO-NLP-SG/Video-LLaMA-2-7B"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def extract_frames(
        self,
        video_path: str,
        num_frames: int = 8,
        uniform: bool = True
    ):
        """Extract frames from video"""
        import cv2

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if uniform:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            # Random sampling
            indices = sorted(np.random.choice(total_frames, num_frames, replace=False))

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()
        return frames

    def analyze_video(
        self,
        video_path: str,
        question: str,
        num_frames: int = 8
    ):
        """Video analysis and question answering"""
        frames = self.extract_frames(video_path, num_frames)

        # Prepare input
        inputs = self.processor(
            text=question,
            images=frames,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response


# Video Captioning
class VideoCaptioner:
    """Video captioning"""

    def __init__(self):
        from transformers import BlipProcessor, BlipForConditionalGeneration

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip2-opt-2.7b"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16
        )

    def caption_video(
        self,
        video_path: str,
        num_frames: int = 5
    ):
        """Generate video caption"""
        frames = self._extract_frames(video_path, num_frames)

        captions = []
        for frame in frames:
            inputs = self.processor(images=frame, return_tensors="pt")
            output = self.model.generate(**inputs, max_new_tokens=50)
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            captions.append(caption)

        # Integrate captions
        summary = self._summarize_captions(captions)
        return summary

    def _summarize_captions(self, captions: list) -> str:
        """Integrate frame captions into video summary"""
        # Simple integration (using LLM recommended in practice)
        unique_elements = set()
        for caption in captions:
            unique_elements.update(caption.lower().split())

        return " → ".join(captions)
```

### 3.2 Video Generation Concept (Sora)

```
Sora Core Concepts:
┌────────────────────────────────────────────────────────┐
│                     Text Prompt                         │
│  "A cat playing piano in a cozy room with warm light"  │
│                         ↓                               │
│  ┌────────────────────────────────────────────────────┐│
│  │              Text Encoder (T5/CLIP)                ││
│  │              - Semantic understanding              ││
│  └────────────────────────────────────────────────────┘│
│                         ↓                               │
│  ┌────────────────────────────────────────────────────┐│
│  │          Spacetime Latent Patches                  ││
│  │          - Video as 3D patches                     ││
│  │          - Compress H×W×T into latent             ││
│  └────────────────────────────────────────────────────┘│
│                         ↓                               │
│  ┌────────────────────────────────────────────────────┐│
│  │              Diffusion Transformer                 ││
│  │              - DiT backbone                        ││
│  │              - Attention over spacetime            ││
│  │              - Variable resolution/duration        ││
│  └────────────────────────────────────────────────────┘│
│                         ↓                               │
│  ┌────────────────────────────────────────────────────┐│
│  │              VAE Decoder                           ││
│  │              - Latent → Pixel space               ││
│  └────────────────────────────────────────────────────┘│
│                         ↓                               │
│                    Video Output                         │
│              (Variable length, up to 1 min)            │
└────────────────────────────────────────────────────────┘

Key techniques:
1. Spacetime Patches: Divide spacetime into patches
2. DiT (Diffusion Transformer): Transformer-based diffusion
3. Variable Resolution: Support various resolutions/lengths
4. Recaptioning: Retrain with detailed captions
```

```python
# Simple Video Diffusion concept implementation
import torch
import torch.nn as nn
from einops import rearrange

class SpacetimePatchEmbed(nn.Module):
    """Spacetime patch embedding"""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        num_frames: int = 16,
        temporal_patch: int = 2,
        in_channels: int = 4,  # VAE latent
        embed_dim: int = 768
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch = temporal_patch

        # 3D patch embedding
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(temporal_patch, patch_size, patch_size),
            stride=(temporal_patch, patch_size, patch_size)
        )

        # Calculate patch counts
        self.num_spatial_patches = (img_size // patch_size) ** 2
        self.num_temporal_patches = num_frames // temporal_patch
        self.num_patches = self.num_spatial_patches * self.num_temporal_patches

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) - video latent
        Returns:
            patches: (B, N, D) - spacetime patches
        """
        # (B, D, t, h, w)
        x = self.proj(x)
        # (B, D, N) -> (B, N, D)
        x = x.flatten(2).transpose(1, 2)
        return x


class VideoTransformerBlock(nn.Module):
    """Transformer block for video"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        num_spatial_patches: int = 256,
        num_temporal_patches: int = 8
    ):
        super().__init__()
        self.num_spatial = num_spatial_patches
        self.num_temporal = num_temporal_patches

        # Spatial attention
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Temporal attention
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # FFN
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, t_emb=None):
        """
        Args:
            x: (B, T*S, D) - spacetime patches
            t_emb: (B, D) - timestep embedding
        """
        B, N, D = x.shape
        T, S = self.num_temporal, self.num_spatial

        # Spatial attention (within each frame)
        x_spatial = rearrange(x, 'b (t s) d -> (b t) s d', t=T, s=S)
        x_spatial = self.spatial_norm(x_spatial)
        attn_out, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = rearrange(attn_out, '(b t) s d -> b (t s) d', b=B, t=T)
        x = x + x_spatial

        # Temporal attention (across frames at same position)
        x_temporal = rearrange(x, 'b (t s) d -> (b s) t d', t=T, s=S)
        x_temporal = self.temporal_norm(x_temporal)
        attn_out, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        x_temporal = rearrange(attn_out, '(b s) t d -> b (t s) d', b=B, s=S)
        x = x + x_temporal

        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x


class SimpleDiT(nn.Module):
    """Simplified Diffusion Transformer"""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        num_frames: int = 16,
        in_channels: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = SpacetimePatchEmbed(
            img_size, patch_size, num_frames,
            temporal_patch=2, in_channels=in_channels,
            embed_dim=hidden_size
        )

        num_spatial = (img_size // patch_size) ** 2
        num_temporal = num_frames // 2

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_spatial * num_temporal, hidden_size)
        )

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            VideoTransformerBlock(
                hidden_size, num_heads,
                num_spatial_patches=num_spatial,
                num_temporal_patches=num_temporal
            )
            for _ in range(depth)
        ])

        # Output
        self.final_norm = nn.LayerNorm(hidden_size)
        self.final_proj = nn.Linear(
            hidden_size,
            patch_size * patch_size * 2 * in_channels
        )

    def forward(self, x, t, cond=None):
        """
        Args:
            x: (B, C, T, H, W) - noisy video latent
            t: (B,) - diffusion timestep
            cond: (B, L, D) - text conditioning
        """
        # Patch embedding
        x = self.patch_embed(x) + self.pos_embed

        # Timestep embedding (sinusoidal)
        t_emb = self._sinusoidal_embedding(t, x.shape[-1])
        t_emb = self.time_embed(t_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # Output
        x = self.final_norm(x)
        x = self.final_proj(x)

        return x

    def _sinusoidal_embedding(self, t, dim):
        """Sinusoidal timestep embedding"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding
```

---

## 4. Practical Applications

### 4.1 Multimodal Pipeline

```python
class MultimodalPipeline:
    """Integrated audio/video pipeline"""

    def __init__(self):
        # Speech recognition
        self.whisper = whisper.load_model("base")

        # Music generation
        self.music_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.music_model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small"
        )

        # TTS
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    def transcribe_and_translate(
        self,
        audio_path: str,
        target_language: str = "en"
    ):
        """Speech recognition and translation"""
        # Speech recognition
        result = self.whisper.transcribe(audio_path)
        original_text = result["text"]
        source_language = result["language"]

        # Translation (to English)
        if source_language != target_language:
            translation = self.whisper.transcribe(
                audio_path,
                task="translate"
            )["text"]
        else:
            translation = original_text

        return {
            "original": original_text,
            "source_language": source_language,
            "translation": translation
        }

    def generate_soundtrack(
        self,
        video_description: str,
        mood: str,
        duration: float = 30.0
    ):
        """Generate background music based on video description"""
        prompt = f"{mood} music for: {video_description}"

        inputs = self.music_processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        )

        max_new_tokens = int(duration * 50)

        audio_values = self.music_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            guidance_scale=4.0
        )

        return audio_values[0, 0].numpy()

    def create_voiceover(
        self,
        script: str,
        reference_voice: str,
        language: str = "en"
    ):
        """Generate voiceover"""
        output_path = "voiceover.wav"

        self.tts.tts_to_file(
            text=script,
            file_path=output_path,
            speaker_wav=reference_voice,
            language=language
        )

        return output_path


# Usage example
def demo_pipeline():
    """Pipeline demo"""
    pipeline = MultimodalPipeline()

    # 1. Transcribe and translate audio file
    result = pipeline.transcribe_and_translate(
        "korean_speech.mp3",
        target_language="en"
    )
    print(f"Original: {result['original']}")
    print(f"Translation: {result['translation']}")

    # 2. Generate background music for video
    music = pipeline.generate_soundtrack(
        video_description="A documentary about ocean wildlife",
        mood="Calm and majestic",
        duration=60.0
    )

    # 3. Generate narration
    voiceover = pipeline.create_voiceover(
        script="Welcome to our exploration of the deep ocean.",
        reference_voice="narrator_sample.wav",
        language="en"
    )
```

### 4.2 Real-time Processing

```python
import asyncio
from collections import deque

class RealTimeAudioProcessor:
    """Real-time audio processing"""

    def __init__(self, buffer_size: float = 3.0):
        self.buffer_size = buffer_size
        self.sample_rate = 16000
        self.audio_buffer = deque(maxlen=int(buffer_size * self.sample_rate))

        # Whisper model (use small version)
        self.model = whisper.load_model("tiny")

    async def process_stream(self, audio_stream):
        """Process audio stream"""
        while True:
            # Receive audio chunk
            chunk = await audio_stream.receive()
            self.audio_buffer.extend(chunk)

            # Process when buffer is sufficient
            if len(self.audio_buffer) >= self.sample_rate * 2:
                audio_array = np.array(self.audio_buffer)

                # Async transcription
                result = await asyncio.to_thread(
                    self.model.transcribe,
                    audio_array,
                    fp16=False
                )

                yield result["text"]

                # Keep partial buffer (overlap)
                self.audio_buffer = deque(
                    list(self.audio_buffer)[self.sample_rate:],
                    maxlen=int(self.buffer_size * self.sample_rate)
                )


class StreamingVideoAnalyzer:
    """Streaming video analysis"""

    def __init__(self, frame_interval: int = 30):
        self.frame_interval = frame_interval
        self.frame_count = 0

        # CLIP for quick frame analysis
        from transformers import CLIPProcessor, CLIPModel
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def analyze_frame(self, frame, categories: list):
        """Frame classification"""
        inputs = self.processor(
            text=categories,
            images=frame,
            return_tensors="pt",
            padding=True
        )

        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

        return {cat: prob.item() for cat, prob in zip(categories, probs[0])}

    def process_video_stream(self, video_stream, categories: list):
        """Process video stream"""
        import cv2

        while True:
            ret, frame = video_stream.read()
            if not ret:
                break

            self.frame_count += 1

            # Analyze only at intervals
            if self.frame_count % self.frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                analysis = self.analyze_frame(frame_rgb, categories)
                yield self.frame_count, analysis
```

---

## 5. Model Comparison

### 5.1 Speech Models

| Model | Parameters | Features | Use Case |
|-------|-----------|----------|----------|
| Whisper Large | 1.55B | Multilingual, Translation | General ASR |
| Whisper Large-v3 | 1.55B | Improved accuracy | Production |
| wav2vec 2.0 | 300M | Self-supervised | Fine-tuning base |
| HuBERT | 300M-1B | Masked prediction | Speech representation |

### 5.2 Audio Generation

| Model | Size | Features | Output |
|-------|------|----------|--------|
| MusicGen Small | 300M | Fast generation | Music |
| MusicGen Large | 3.3B | High quality | Music |
| AudioGen | 300M-1.5B | Sound effects | Audio |
| Bark | 1B+ | Non-verbal expressions | TTS |

### 5.3 Video Models

| Model | Architecture | Input | Tasks |
|-------|-------------|-------|-------|
| VideoLLaMA | LLaMA + Q-Former | Video + Audio | VQA, Captioning |
| Video-ChatGPT | LLaVA variant | Video | Conversation |
| TimeSformer | Divided attention | Video | Classification |
| ViViT | Factorized | Video | Classification |

---

## Key Summary

### Audio Foundation Models
```
Whisper: General ASR + Translation
├── Encoder-Decoder Transformer
├── 680K hours training data
└── Multilingual (99 languages)

MusicGen: Text→Music
├── Autoregressive Transformer
├── EnCodec tokenization
└── Text/Melody conditioned
```

### Video Foundation Models
```
Video Understanding:
├── Frame sampling → Visual encoder
├── Temporal aggregation (Q-Former/pooling)
└── LLM backbone for reasoning

Video Generation (Sora concept):
├── Spacetime patches (3D tokenization)
├── Diffusion Transformer (DiT)
└── Variable resolution/duration
```

### Practical Points
1. **Whisper**: Can be domain-specialized with fine-tuning
2. **MusicGen**: Control quality/diversity with guidance_scale
3. **Video**: Frame sampling strategy is crucial
4. **Real-time**: Small models + streaming buffer

---

## References

1. Radford et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper)
2. Copet et al. (2023). "Simple and Controllable Music Generation" (MusicGen)
3. Borsos et al. (2023). "AudioLM: a Language Modeling Approach to Audio Generation"
4. Zhang et al. (2023). "Video-LLaMA: An Instruction-tuned Audio-Visual Language Model"
5. OpenAI Sora Technical Report (2024)

---

## Exercises

### Exercise 1: Whisper Architecture and 30-Second Segments
Whisper processes audio in fixed 30-second chunks. Explain the architectural reason for this constraint and describe how the pipeline handles audio files that are much shorter (e.g., 5 seconds) or much longer (e.g., 3 minutes). What problem arises when a word or sentence is split across a 30-second boundary?

<details>
<summary>Show Answer</summary>

**Why 30-second chunks**: Whisper's audio encoder uses a fixed-size log-Mel spectrogram input of 80 frequency bins × 3000 time frames, where each time frame represents 10ms. This gives exactly 30 seconds (3000 × 10ms). The encoder uses absolute sinusoidal positional embeddings — it cannot generalize to variable-length inputs without architectural modifications.

**Handling shorter audio (5 seconds)**:
- The 5-second audio is zero-padded to fill the 30-second buffer.
- The encoder processes the padded spectrogram normally.
- The decoder generates tokens until it produces an `<|endoftext|>` token, naturally stopping at the end of actual speech without generating content for the silent padding.

**Handling longer audio (3 minutes)**:
- The audio is split into overlapping 30-second chunks.
- Each chunk is transcribed independently.
- Transcriptions are stitched together, usually by finding the overlap region using timestamp alignment.
- Whisper's `word_timestamps=True` mode provides timing information to locate exact stitch points.

**Boundary splitting problem (word cut mid-segment)**:
- A word like "extraordinary" might be split: "extraord-" ends chunk 1, "-inary" starts chunk 2.
- The first chunk encoder sees "extraord" as the last word — it may transcribe it as "extraord" (incomplete), be cut off, or hallucinate a different ending.
- Chunk 2 starts with "-inary" in the middle — the decoder lacks the beginning context.
- Result: words near segment boundaries are the most common transcription error sources.
- **Mitigation**: Use overlapping windows (e.g., 25-second chunks with 5-second overlap) and take the higher-confidence transcription for overlapping regions, or use a VAD (Voice Activity Detector) to split only at silences.

</details>

### Exercise 2: AudioLM Hierarchical Tokenization
AudioLM uses a three-level hierarchy: Semantic tokens (w2v-BERT, ~25/sec), Coarse Acoustic tokens (SoundStream, ~50/sec), and Fine Acoustic tokens (SoundStream, ~100/sec). Explain why this hierarchy is necessary — why can't AudioLM just use a single token representation at one resolution?

<details>
<summary>Show Answer</summary>

**Why a single resolution fails**:

**Single high-resolution tokens (fine acoustic only, ~100/sec)**:
- For 10 seconds of audio: 1000 tokens
- A language model must learn to generate coherent speech over 1000 tokens while maintaining semantic consistency (same word, correct grammar, continuous meaning).
- Long-range dependencies are very hard to model at this scale — transformers struggle with O(n²) attention over 1000 tokens to maintain semantic coherence.
- The model would waste capacity learning to copy acoustic details rather than planning semantic content.

**Single low-resolution tokens (semantic only, ~25/sec)**:
- Semantic tokens capture content (what is said) but discard speaker identity, prosody (how it sounds), and fine acoustic details.
- You can't reconstruct high-quality audio from semantic tokens alone — the result sounds robotic or like a vocoder.

**Why the hierarchy solves this**:

1. **Semantic layer** (~25 tokens/sec): Captures "what is said" — word identities, language content. Models here can use long context windows affordably. This is where grammatical coherence and content planning happen.

2. **Coarse Acoustic layer** (~50 tokens/sec): Conditioned on semantic tokens, captures "how it sounds" at a coarse level — speaker identity, general prosody, speaking rate.

3. **Fine Acoustic layer** (~100 tokens/sec): Conditioned on coarse acoustic, adds fine-grained perceptual details — microphone characteristics, fine timing, subtle acoustic textures.

The hierarchy allows each level to focus on its own problem complexity: semantic planning over long context, then progressive acoustic refinement. This divide-and-conquer approach makes each sub-problem tractable.

</details>

### Exercise 3: Video Frame Sampling Strategy
The `VideoUnderstanding` class uses uniform frame sampling (`np.linspace(0, total_frames-1, num_frames)`). Analyze the limitations of this approach and propose two alternative sampling strategies for different video types.

```python
def extract_frames(self, video_path: str, num_frames: int = 8, uniform: bool = True):
    """Extract frames from video"""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if uniform:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    # What other sampling strategies should exist?
```

<details>
<summary>Show Answer</summary>

**Limitations of uniform sampling**:
1. **Redundancy for static videos**: A talking-head interview has 99% identical frames — uniform sampling wastes capacity on near-duplicate frames.
2. **Missing key events**: A sports highlight video may have 3 seconds of action and 27 seconds of replay/commentary — uniform sampling undersamples the action.
3. **Ignores motion**: All frames are treated equally regardless of how much changes between them.
4. **Resolution-length mismatch**: A 2-minute video and a 10-minute video both get 8 frames with uniform sampling, but the 10-minute video has 5x less temporal resolution per second.

**Alternative Strategy 1: Scene-change / keyframe-based sampling**
```python
def sample_keyframes(self, video_path, num_frames=8):
    """Sample frames at scene boundaries"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    scene_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Compute inter-frame difference
            diff = cv2.absdiff(gray, prev_frame).mean()
            scene_scores.append((cap.get(cv2.CAP_PROP_POS_FRAMES), diff))

        prev_frame = gray

    # Select frames at highest scene-change scores
    scene_scores.sort(key=lambda x: -x[1])
    top_indices = sorted([int(s[0]) for s in scene_scores[:num_frames]])
    return top_indices
```
Best for: Action videos, movies, tutorials with distinct steps.

**Alternative Strategy 2: Activity-density-proportional sampling**
```python
def sample_activity_proportional(self, video_path, num_frames=8):
    """Sample more frames from high-motion segments"""
    # 1. Compute optical flow or frame difference for all frames
    # 2. Compute "activity score" per second
    # 3. Allocate frames proportionally to activity score
    # High-activity segment: 3 seconds → 4 frames
    # Low-activity segment: 7 seconds → 4 frames
    # (rather than fixed 1 frame per 1.25 seconds)
    pass
```
Best for: Sports, instructional videos, surveillance footage where key events occur in bursts.

</details>

### Exercise 4: MusicGen Token Budget Calculation
MusicGen generates audio at 32kHz using EnCodec compression that produces 50 tokens/second. You want to generate a 60-second music clip.

A) Calculate the total tokens needed.
B) If the transformer has a max context length of 4096 tokens, can you generate the full 60 seconds in one pass? What is the maximum duration in one pass?
C) Propose a strategy to generate longer music (5 minutes) while maintaining musical coherence.

<details>
<summary>Show Answer</summary>

**A) Total tokens for 60 seconds**:
```python
duration_seconds = 60
tokens_per_second = 50  # EnCodec compression rate
total_tokens = 60 * 50 = 3000 tokens
```

**B) Can 60 seconds fit in 4096 context?**:
```python
max_context = 4096
max_duration = 4096 / 50 = 81.92 seconds
# YES: 3000 < 4096, so 60 seconds fits in a single pass.
# Maximum duration in one pass: ~81 seconds
```

**C) Strategy for 5-minute (300 seconds) generation**:
5 minutes × 50 tokens/sec = 15,000 tokens >> 4096 context limit.

**Approach: Continuation with overlap (similar to Whisper chunking)**:
```python
def generate_long_music(prompt: str, target_duration: float = 300.0):
    """Generate long music with continuation"""
    chunk_duration = 75.0    # ~3750 tokens, leaving room for context
    overlap_duration = 6.0   # 6-second overlap for coherence

    chunks = []
    generated_so_far = 0.0

    while generated_so_far < target_duration:
        if generated_so_far == 0:
            # First chunk: generate from text prompt
            chunk = model.generate(
                text_prompt=prompt,
                duration=chunk_duration
            )
        else:
            # Subsequent chunks: condition on last N seconds of previous chunk
            # The last overlap_duration of audio serves as musical context
            continuation_audio = chunks[-1][-overlap_duration:]
            chunk = model.generate_continuation(
                prompt_audio=continuation_audio,
                text_prompt=prompt,  # Maintain style consistency
                duration=chunk_duration - overlap_duration
            )

        chunks.append(chunk)
        generated_so_far += (chunk_duration - overlap_duration)

    # Crossfade between chunks at overlap regions to avoid hard cuts
    return crossfade_concatenate(chunks, overlap_duration)
```

Key insight: The text prompt stays constant across all chunks to maintain style consistency. The audio overlap provides musical continuity (key, tempo, instrument continuation). Crossfading at boundaries prevents audible clicks from discontinuities.

</details>
