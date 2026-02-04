# 18. Audio/Video Foundation Models

## 개요

Audio와 Video 도메인의 Foundation Model들은 음성 인식, 음악 생성, 비디오 이해 등 다양한 멀티미디어 태스크를 통합적으로 처리합니다.

---

## 1. Speech Foundation Models

### 1.1 Whisper

OpenAI의 범용 음성 인식 모델:

```
Whisper 아키텍처:
┌─────────────────────────────────────────────┐
│  Audio Input (30초 세그먼트)                  │
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

모델 크기:
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

# OpenAI whisper 사용
def transcribe_with_whisper():
    """OpenAI Whisper로 음성 인식"""
    model = whisper.load_model("base")

    # 기본 transcription
    result = model.transcribe("audio.mp3")
    print(result["text"])

    # 언어 감지 및 번역
    result = model.transcribe(
        "audio.mp3",
        task="translate",  # 영어로 번역
        language=None,     # 자동 감지
        fp16=torch.cuda.is_available()
    )

    # 타임스탬프 포함
    result = model.transcribe(
        "audio.mp3",
        word_timestamps=True
    )

    for segment in result["segments"]:
        print(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")

    return result


# HuggingFace Transformers 사용
def transcribe_with_hf_whisper():
    """HuggingFace Whisper 사용"""
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

    # 오디오 로드 (16kHz)
    import librosa
    audio, sr = librosa.load("audio.mp3", sr=16000)

    # 입력 처리
    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    # 생성
    predicted_ids = model.generate(
        input_features,
        language="korean",
        task="transcribe"
    )

    # 디코딩
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return transcription


# Whisper Fine-tuning
class WhisperFineTuner:
    """도메인 특화 Whisper Fine-tuning"""

    def __init__(self, model_name: str = "openai/whisper-small"):
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
            Seq2SeqTrainingArguments,
            Seq2SeqTrainer
        )

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

        # Freeze encoder (선택적)
        for param in self.model.model.encoder.parameters():
            param.requires_grad = False

    def prepare_dataset(self, dataset):
        """데이터셋 전처리"""
        def prepare_example(example):
            audio = example["audio"]["array"]

            # 입력 특징 추출
            input_features = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features[0]

            # 레이블 토큰화
            labels = self.processor.tokenizer(
                example["transcription"]
            ).input_ids

            return {
                "input_features": input_features,
                "labels": labels
            }

        return dataset.map(prepare_example)

    def train(self, train_dataset, eval_dataset):
        """Fine-tuning 실행"""
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
    """Coqui TTS로 음성 합성"""
    from TTS.api import TTS

    # 다국어 TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    # 음성 합성
    tts.tts_to_file(
        text="안녕하세요, Foundation Model 학습 자료입니다.",
        file_path="output.wav",
        speaker_wav="reference_voice.wav",  # Voice cloning
        language="ko"
    )


# Bark (Suno AI)
def text_to_speech_bark():
    """Bark로 음성 합성 (비언어적 표현 포함)"""
    from transformers import AutoProcessor, BarkModel
    import scipy.io.wavfile as wavfile

    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")

    # 텍스트 (비언어적 표현 포함 가능)
    text = "[laughs] Hello! This is amazing. [sighs]"

    inputs = processor(text, return_tensors="pt")
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()

    # 저장
    wavfile.write("bark_output.wav", 24000, audio_array)
```

---

## 2. Audio Generation Models

### 2.1 AudioLM

Google의 Audio Language Model:

```
AudioLM 구조:
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

3단계 생성:
1. Semantic → Semantic (continuation)
2. Semantic → Coarse Acoustic
3. Coarse → Fine Acoustic
```

### 2.2 MusicGen

Meta의 음악 생성 모델:

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile

class MusicGenerator:
    """MusicGen을 사용한 음악 생성"""

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
        """텍스트 프롬프트로 음악 생성"""
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # 토큰 수 계산 (32kHz, 50 tokens/second)
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
        """멜로디 조건부 생성 (melody 모델만)"""
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
        """오디오 저장"""
        wavfile.write(path, 32000, audio)


# 사용 예시
def music_generation_examples():
    """다양한 음악 생성 예시"""
    generator = MusicGenerator("small")

    # 텍스트 기반 생성
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
# AudioGen (사운드 효과 생성)
def generate_sound_effects():
    """AudioGen으로 사운드 효과 생성"""
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
VideoLLaMA 아키텍처:
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
    """비디오 이해 모델"""

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
        """비디오에서 프레임 추출"""
        import cv2

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if uniform:
            # 균일 샘플링
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            # 랜덤 샘플링
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
        """비디오 분석 및 질문 응답"""
        frames = self.extract_frames(video_path, num_frames)

        # 입력 준비
        inputs = self.processor(
            text=question,
            images=frames,
            return_tensors="pt"
        ).to(self.model.device)

        # 생성
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
    """비디오 캡셔닝"""

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
        """비디오 캡션 생성"""
        frames = self._extract_frames(video_path, num_frames)

        captions = []
        for frame in frames:
            inputs = self.processor(images=frame, return_tensors="pt")
            output = self.model.generate(**inputs, max_new_tokens=50)
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            captions.append(caption)

        # 캡션 통합
        summary = self._summarize_captions(captions)
        return summary

    def _summarize_captions(self, captions: list) -> str:
        """프레임별 캡션을 비디오 요약으로 통합"""
        # 간단한 통합 (실제로는 LLM 사용 권장)
        unique_elements = set()
        for caption in captions:
            unique_elements.update(caption.lower().split())

        return " → ".join(captions)
```

### 3.2 Video Generation 개념 (Sora)

```
Sora 핵심 개념:
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

핵심 기술:
1. Spacetime Patches: 시공간을 패치로 분할
2. DiT (Diffusion Transformer): Transformer 기반 diffusion
3. Variable Resolution: 다양한 해상도/길이 지원
4. Recaptioning: 상세한 캡션으로 재학습
```

```python
# 간단한 Video Diffusion 개념 구현
import torch
import torch.nn as nn
from einops import rearrange

class SpacetimePatchEmbed(nn.Module):
    """시공간 패치 임베딩"""

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

        # 3D 패치 임베딩
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(temporal_patch, patch_size, patch_size),
            stride=(temporal_patch, patch_size, patch_size)
        )

        # 패치 수 계산
        self.num_spatial_patches = (img_size // patch_size) ** 2
        self.num_temporal_patches = num_frames // temporal_patch
        self.num_patches = self.num_spatial_patches * self.num_temporal_patches

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) - 비디오 latent
        Returns:
            patches: (B, N, D) - 시공간 패치
        """
        # (B, D, t, h, w)
        x = self.proj(x)
        # (B, D, N) -> (B, N, D)
        x = x.flatten(2).transpose(1, 2)
        return x


class VideoTransformerBlock(nn.Module):
    """비디오용 Transformer 블록"""

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
            x: (B, T*S, D) - 시공간 패치
            t_emb: (B, D) - timestep 임베딩
        """
        B, N, D = x.shape
        T, S = self.num_temporal, self.num_spatial

        # Spatial attention (각 프레임 내)
        x_spatial = rearrange(x, 'b (t s) d -> (b t) s d', t=T, s=S)
        x_spatial = self.spatial_norm(x_spatial)
        attn_out, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = rearrange(attn_out, '(b t) s d -> b (t s) d', b=B, t=T)
        x = x + x_spatial

        # Temporal attention (같은 위치의 프레임 간)
        x_temporal = rearrange(x, 'b (t s) d -> (b s) t d', t=T, s=S)
        x_temporal = self.temporal_norm(x_temporal)
        attn_out, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        x_temporal = rearrange(attn_out, '(b s) t d -> b (t s) d', b=B, s=S)
        x = x + x_temporal

        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x


class SimpleDiT(nn.Module):
    """단순화된 Diffusion Transformer"""

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

## 4. 실용적 응용

### 4.1 Multimodal Pipeline

```python
class MultimodalPipeline:
    """오디오/비디오 통합 파이프라인"""

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
        """음성 인식 및 번역"""
        # 음성 인식
        result = self.whisper.transcribe(audio_path)
        original_text = result["text"]
        source_language = result["language"]

        # 번역 (영어로)
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
        """비디오 설명 기반 배경음악 생성"""
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
        """보이스오버 생성"""
        output_path = "voiceover.wav"

        self.tts.tts_to_file(
            text=script,
            file_path=output_path,
            speaker_wav=reference_voice,
            language=language
        )

        return output_path


# 사용 예시
def demo_pipeline():
    """파이프라인 데모"""
    pipeline = MultimodalPipeline()

    # 1. 음성 파일 전사 및 번역
    result = pipeline.transcribe_and_translate(
        "korean_speech.mp3",
        target_language="en"
    )
    print(f"Original: {result['original']}")
    print(f"Translation: {result['translation']}")

    # 2. 비디오용 배경음악 생성
    music = pipeline.generate_soundtrack(
        video_description="A documentary about ocean wildlife",
        mood="Calm and majestic",
        duration=60.0
    )

    # 3. 나레이션 생성
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
    """실시간 오디오 처리"""

    def __init__(self, buffer_size: float = 3.0):
        self.buffer_size = buffer_size
        self.sample_rate = 16000
        self.audio_buffer = deque(maxlen=int(buffer_size * self.sample_rate))

        # Whisper 모델 (작은 버전 사용)
        self.model = whisper.load_model("tiny")

    async def process_stream(self, audio_stream):
        """오디오 스트림 처리"""
        while True:
            # 오디오 청크 수신
            chunk = await audio_stream.receive()
            self.audio_buffer.extend(chunk)

            # 버퍼가 충분하면 처리
            if len(self.audio_buffer) >= self.sample_rate * 2:
                audio_array = np.array(self.audio_buffer)

                # 비동기 전사
                result = await asyncio.to_thread(
                    self.model.transcribe,
                    audio_array,
                    fp16=False
                )

                yield result["text"]

                # 버퍼 일부 유지 (오버랩)
                self.audio_buffer = deque(
                    list(self.audio_buffer)[self.sample_rate:],
                    maxlen=int(self.buffer_size * self.sample_rate)
                )


class StreamingVideoAnalyzer:
    """스트리밍 비디오 분석"""

    def __init__(self, frame_interval: int = 30):
        self.frame_interval = frame_interval
        self.frame_count = 0

        # CLIP for quick frame analysis
        from transformers import CLIPProcessor, CLIPModel
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def analyze_frame(self, frame, categories: list):
        """프레임 분류"""
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
        """비디오 스트림 처리"""
        import cv2

        while True:
            ret, frame = video_stream.read()
            if not ret:
                break

            self.frame_count += 1

            # 일정 간격으로만 분석
            if self.frame_count % self.frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                analysis = self.analyze_frame(frame_rgb, categories)
                yield self.frame_count, analysis
```

---

## 5. 모델 비교

### 5.1 Speech Models

| 모델 | 파라미터 | 특징 | 용도 |
|------|---------|------|------|
| Whisper Large | 1.55B | 다국어, 번역 | 범용 ASR |
| Whisper Large-v3 | 1.55B | 개선된 정확도 | 프로덕션 |
| wav2vec 2.0 | 300M | Self-supervised | Fine-tuning 베이스 |
| HuBERT | 300M-1B | Masked prediction | Speech representation |

### 5.2 Audio Generation

| 모델 | 크기 | 특징 | 출력 |
|------|------|------|------|
| MusicGen Small | 300M | 빠른 생성 | 음악 |
| MusicGen Large | 3.3B | 고품질 | 음악 |
| AudioGen | 300M-1.5B | 사운드 효과 | 오디오 |
| Bark | 1B+ | 비언어 표현 | TTS |

### 5.3 Video Models

| 모델 | 아키텍처 | 입력 | 태스크 |
|------|---------|------|--------|
| VideoLLaMA | LLaMA + Q-Former | Video + Audio | VQA, Captioning |
| Video-ChatGPT | LLaVA variant | Video | Conversation |
| TimeSformer | Divided attention | Video | Classification |
| ViViT | Factorized | Video | Classification |

---

## 핵심 정리

### Audio Foundation Models
```
Whisper: 범용 ASR + 번역
├── Encoder-Decoder Transformer
├── 680K 시간 학습 데이터
└── 다국어 (99개 언어)

MusicGen: 텍스트→음악
├── Autoregressive Transformer
├── EnCodec 토큰화
└── 텍스트/멜로디 조건부
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

### 실용 포인트
1. **Whisper**: Fine-tuning으로 도메인 특화 가능
2. **MusicGen**: guidance_scale로 품질/다양성 조절
3. **Video**: 프레임 샘플링 전략이 중요
4. **Real-time**: 작은 모델 + 스트리밍 버퍼

---

## 참고 자료

1. Radford et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper)
2. Copet et al. (2023). "Simple and Controllable Music Generation" (MusicGen)
3. Borsos et al. (2023). "AudioLM: a Language Modeling Approach to Audio Generation"
4. Zhang et al. (2023). "Video-LLaMA: An Instruction-tuned Audio-Visual Language Model"
5. OpenAI Sora Technical Report (2024)
