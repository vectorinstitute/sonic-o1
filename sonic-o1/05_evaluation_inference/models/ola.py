"""ola.py
OLA (Omni-modal Language Assistant) implementation following BaseModel pattern.

Author: SONIC-O1 Team
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from utils.audio_processor import sample_audio_chunks

from .base_model import BaseModel


logger = logging.getLogger(__name__)


class OLA(BaseModel):
    """OLA-7b wrapper following BaseModel pattern.

    Uses our own video frame sampling and audio chunking to handle
    long videos (up to 1 hour), overriding OLA's built-in 12.5-min
    audio truncation and fixed 64-frame video sampling.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]) -> None:
        super().__init__(model_name, config)

        self.model_path = config.get("model_path", "THUdyh/Ola-7b")
        self.ola_repo_path = config.get("ola_repo_path")

        # Frame config
        self.default_max_frames = config.get("max_frames", 64)
        self.default_min_frames = config.get("min_frames", 16)

        # Audio config
        self.audio_sample_rate = config.get("audio_sample_rate", 16000)
        self.audio_chunk_limit = 480000  # 30s at 16kHz — OLA's window size
       # self.max_speech_chunks = config.get("max_speech_chunks", 25)  # OLA hard cap

        # Generation config
        gen_config = config.get("generation_config", {})
        self.temperature = gen_config.get("temperature", 0.2)
        self.top_p = gen_config.get("top_p", None)
        self.num_beams = gen_config.get("num_beams", 1)
        self.max_new_tokens = gen_config.get("max_new_tokens", 1024)

        # Model components (loaded in load())
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None

        # Stats
        self.stats = {
            "total_samples": 0,
            "audio_chunks_sampled": 0,
        }
    def convert_av1_to_h264(
        self, video_path: Path, output_dir: Optional[Path] = None
    ) -> Path:
        """
        Convert AV1 video to H.264 for Decord compatibility.

        Args:
            video_path: Input video path
            output_dir: Output directory (default: creates 'converted' subdir)

        Returns:
            Path to converted video.
        """
        import subprocess

        if output_dir is None:
            output_dir = video_path.parent / "converted"
            output_dir.mkdir(exist_ok=True)

        output_path = output_dir / f"{video_path.stem}_h264{video_path.suffix}"

        # Check if already converted
        if output_path.exists():
            logger.info(f"Using cached converted video: {output_path.name}")
            return output_path

        logger.info(f"Converting AV1 to H.264: {video_path.name}")

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "copy",
            "-y",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"✓ Conversion successful: {output_path.name}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Conversion failed: {e.stderr}")
            raise RuntimeError(f"Failed to convert AV1 video: {e.stderr}")

    def _check_video_compatibility(self, video_path: Path) -> Optional[Path]:
        """
        Check if video is compatible with Decord.
        If AV1, automatically convert to H.264.

        Args:
            video_path: Original video path.

        Returns:
            Path to compatible video (original or converted).
        """
        import subprocess

        from decord import VideoReader, cpu

        try:
            vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
            frame_count = len(vr)
            del vr
            logger.info(f"✓ Video compatible: {video_path.name} ({frame_count} frames)")
            return video_path
        except Exception:
            logger.warning(f"⚠ Video incompatible with Decord: {video_path.name}")

            # Detect codec
            try:
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(video_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                codec = result.stdout.strip()

                if codec == "av1":
                    logger.info("  → Detected AV1 codec - converting to H.264...")
                    try:
                        return self.convert_av1_to_h264(video_path)
                    except Exception as conv_error:
                        logger.error(f"  ✗ Conversion failed: {conv_error}")
                        return None
                else:
                    logger.warning(f"  ✗ Unsupported codec '{codec}' - skipping")
                    return None
            except Exception as probe_error:
                logger.error(f"  ✗ Failed to detect codec: {probe_error}")
                return None
                
    def load(self) -> None:
        """Load OLA model."""
        if self.ola_repo_path:
            ola_path = os.path.expanduser(self.ola_repo_path)
            if ola_path not in sys.path:
                sys.path.insert(0, ola_path)
                logger.info(f"Added OLA repo to path: {ola_path}")

        # Set OLA's required env vars before importing
        os.environ.setdefault("LOWRES_RESIZE", "384x32")
        os.environ.setdefault("HIGHRES_BASE", "0x32")
        os.environ.setdefault("VIDEO_RESIZE", "0x64")
        os.environ.setdefault("VIDEO_MAXRES", "480")
        os.environ.setdefault("VIDEO_MINRES", "288")
        os.environ.setdefault("MAXRES", "1536")
        os.environ.setdefault("MINRES", "0")
        os.environ.setdefault("FORCE_NO_DOWNSAMPLE", "1")
        os.environ.setdefault("LOAD_VISION_EARLY", "1")
        os.environ.setdefault("PAD2STRIDE", "1")

        try:
            from ola.model.builder import load_pretrained_model
        except ImportError as e:
            raise ImportError(
                f"Failed to import OLA modules. Set 'ola_repo_path' in config.\n"
                f"Error: {e}"
            )

        logger.info(f"Loading OLA from {self.model_path}")

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(self.model_path, None)
        )
        self.model = self.model.to("cuda").eval().bfloat16()

        logger.info("OLA loaded successfully")

    def _sample_video_frames(self, video_path: str, max_frames: int, min_frames: int):
        video_path = Path(video_path)
        compatible_path = self._check_video_compatibility(video_path)
        if compatible_path is None:
            raise RuntimeError(f"Video codec incompatible with Decord: {video_path}")
        
        vreader = VideoReader(str(compatible_path), ctx=cpu(0))

        total_frames = len(vreader)

        num_frames = max(min_frames, min(max_frames, total_frames))
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vreader.get_batch(indices.tolist()).asnumpy()
        return [Image.fromarray(f) for f in frames], indices.tolist()

    def _load_audio_ola(
        self,
        audio_path: str,
        max_chunks: Optional[int] = None,
    ):
        """
        Load and chunk audio for OLA, overriding their 12.5-min truncation.

        Our approach: uniformly sample up to max_chunks windows from the full
        audio duration, instead of taking the first N chunks. This ensures
        coverage of the full video for long recordings.

        Returns:
            tuple: (mels, speech_lengths, speech_chunks, speech_wavs) — all on CPU.
        """
        import librosa
        import whisper

        speech_wav, _ = librosa.load(audio_path, sr=self.audio_sample_rate)
        if len(speech_wav.shape) > 1:
            speech_wav = speech_wav[:, 0]
        speech_wav = speech_wav.astype(np.float32)

        total_samples = len(speech_wav)
        chunk_lim = self.audio_chunk_limit  # 30s window

        # Build all 30s chunks
        all_chunks = []
        for i in range(0, total_samples, chunk_lim):
            chunk = speech_wav[i: i + chunk_lim]
            chunk = whisper.pad_or_trim(chunk)
            all_chunks.append(chunk)

        total_chunks = len(all_chunks)
        logger.info(f"Audio: {total_samples / self.audio_sample_rate:.1f}s → "
                    f"{total_chunks} chunks of 30s")

        # None = no limit, let all chunks through
        if max_chunks is not None and total_chunks > max_chunks:
            indices = np.linspace(0, total_chunks - 1, max_chunks, dtype=int)
            sampled_chunks = [all_chunks[i] for i in indices]
            logger.info(f"Uniformly sampled {max_chunks}/{total_chunks} audio chunks")
            self.stats["audio_chunks_sampled"] += 1
        else:
            sampled_chunks = all_chunks
            logger.info(f"Using all {total_chunks} audio chunks")

        # Build tensors
        mels = []
        speech_wavs = []
        for chunk in sampled_chunks:
            mel = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
            mels.append(mel)
            speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))

        mels = torch.cat(mels, dim=0)           # [N, 3000, 128]
        speech_wavs = torch.cat(speech_wavs, dim=0)  # [N, 480000]

        speech_lengths = torch.LongTensor([mels.shape[1]] * mels.shape[0])
        speech_chunks = torch.LongTensor([mels.shape[0]])

        return mels, speech_lengths, speech_chunks, speech_wavs

    def _process_video_frames(self, frames: list):
        """Process PIL frames through OLA's image processor."""
        from ola.mm_utils import process_anyres_video

        self.image_processor.do_resize = False
        self.image_processor.do_center_crop = False

        video_processed = []
        for frame in frames:
            frame_tensor = process_anyres_video(frame, self.image_processor)
            video_processed.append(frame_tensor.unsqueeze(0))

        video_processed = torch.cat(video_processed, dim=0).bfloat16().to("cuda")
        return video_processed

    def generate(
        self,
        frames: Union[str, Path],
        audio: Optional[Union[str, Path]] = None,
        prompt: str = "Describe what you see and hear.",
        fps: Optional[float] = None,
        video_category: Optional[Literal["short", "medium", "long"]] = None,
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate response from video and optional audio."""
        from ola.conversation import conv_templates, SeparatorStyle
        from ola.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_SPEECH_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from ola.datasets.preprocess import (
            tokenizer_image_token,
            tokenizer_speech_image_token,
        )
        from ola.mm_utils import KeywordsStoppingCriteria

        actual_max_frames = max_frames if max_frames is not None else self.default_max_frames
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        num_beams = kwargs.get("num_beams", self.num_beams)
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)

        try:
            self.stats["total_samples"] += 1

            # ── Audio ──────────────────────────────────────────────────────────
            has_audio = audio is not None and os.path.exists(str(audio))
            speechs, speech_lengths, speech_wavs, speech_chunks = [], [], [], []

            if has_audio:
                logger.info(f"Loading audio: {audio}")
                try:
                    mels, s_lengths, s_chunks, s_wavs = self._load_audio_ola(
                        str(audio), max_chunks=max_audio_chunks
                    )
                    speechs.append(mels.bfloat16().to("cuda"))
                    speech_lengths.append(s_lengths.to("cuda"))
                    speech_chunks.append(s_chunks.to("cuda"))
                    speech_wavs.append(s_wavs.to("cuda"))
                    logger.info(f"Audio loaded: {s_chunks[0].item()} chunks")
                except Exception as e:
                    logger.error(f"Audio processing failed: {e}")
                    logger.warning("Falling back to dummy audio")
                    has_audio = False

            if not has_audio:
                # Dummy audio — OLA handles this gracefully
                speechs = [torch.zeros(1, 3000, 128).bfloat16().to("cuda")]
                speech_lengths = [torch.LongTensor([3000]).to("cuda")]
                speech_wavs = [torch.zeros(1, 480000).to("cuda")]
                speech_chunks = [torch.LongTensor([1]).to("cuda")]

            # ── Video ──────────────────────────────────────────────────────────
            has_video = frames is not None and os.path.exists(str(frames))

            if has_video:
                logger.info(f"Loading video: {frames}")
                pil_frames, frame_idx = self._sample_video_frames(
                    str(frames),
                    max_frames=actual_max_frames,
                    min_frames=self.default_min_frames,
                )
                video_tensor = self._process_video_frames(pil_frames)
                video_data = (
                    (video_tensor, video_tensor),  # (images, images_highres)
                    (384, 384),
                    "video",
                )
                logger.info(f"Video frames sampled: {len(pil_frames)}")

                # Build prompt
                qs = DEFAULT_SPEECH_TOKEN + DEFAULT_IMAGE_TOKEN + "\n" + prompt

            else:
                # No video — audio only
                qs = DEFAULT_SPEECH_TOKEN + "\n" + prompt

            # ── Tokenize ───────────────────────────────────────────────────────
            conv_mode = "qwen_1_5"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            full_prompt = conv.get_prompt()

            if has_video:
                input_ids = (
                    tokenizer_speech_image_token(
                        full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .to("cuda")
                )
            else:
                from ola.datasets.preprocess import tokenizer_speech_token
                from ola.constants import SPEECH_TOKEN_INDEX
                input_ids = (
                    tokenizer_speech_token(
                        full_prompt, self.tokenizer, SPEECH_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .to("cuda")
                )

            pad_token_ids = 151643
            attention_masks = input_ids.ne(pad_token_ids).long().to("cuda")

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria(
                [stop_str], self.tokenizer, input_ids
            )

            # ── Generate ───────────────────────────────────────────────────────
            logger.info("Generating response...")
            with torch.inference_mode():
                if has_video:
                    output_ids = self.model.generate(
                        inputs=input_ids,
                        images=video_data[0][0],
                        images_highres=video_data[0][1],
                        modalities=video_data[2],
                        speech=speechs,
                        speech_lengths=speech_lengths,
                        speech_chunks=speech_chunks,
                        speech_wav=speech_wavs,
                        attention_mask=attention_masks,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                        do_sample=temperature > 0,
                        temperature=temperature,
                        top_p=top_p,
                        num_beams=num_beams,
                        max_new_tokens=max_new_tokens,
                    )
                else:
                    dummy_images = [
                        torch.zeros(1, 3, 224, 224).bfloat16().to("cuda")
                    ]
                    output_ids = self.model.generate(
                        inputs=input_ids,
                        images=dummy_images,
                        images_highres=dummy_images,
                        image_sizes=[(224, 224)],
                        modalities=["text"],
                        speech=speechs,
                        speech_lengths=speech_lengths,
                        speech_chunks=speech_chunks,
                        speech_wav=speech_wavs,
                        attention_mask=attention_masks,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                        do_sample=temperature > 0,
                        temperature=temperature,
                        top_p=top_p,
                        num_beams=num_beams,
                        max_new_tokens=max_new_tokens,
                    )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

            logger.info(f"Generated {len(outputs)} characters")
            return self.postprocess_output(outputs)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "size of tensor" in error_msg.lower():
                logger.error(f"OOM error: {error_msg[:200]}...")
                torch.cuda.empty_cache()
                raise RuntimeError(f"Out of memory: {e}")
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")

    def unload(self) -> None:
        """Unload model and free memory."""
        logger.info("Unloading OLA model...")

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.image_processor is not None:
            del self.image_processor
            self.image_processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("OLA model unloaded")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update(
            {
                "model_path": self.model_path,
                "backend": "HuggingFace Transformers",
                "native_video": True,
                "native_audio": True,
                "default_max_frames": self.default_max_frames,
                "default_min_frames": self.default_min_frames,
                "max_speech_chunks": self.max_speech_chunks,
                "context_length": self.context_len,
                "statistics": self.stats,
            }
        )
        return info
