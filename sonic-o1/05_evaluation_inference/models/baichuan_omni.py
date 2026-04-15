"""baichuan_omni.py
Baichuan-Omni-1.5 wrapper following BaseModel pattern.

Architecture (traced from source):
- LLM backbone:    Qwen2.5-7B (bfloat16)
- Vision encoder:  CLIP ViT-L/14 (patch=14, spatial_merge=2)
- Audio encoder:   Whisper-large (16kHz mel, max 30s window — same constraint as OLA)
- Video handling:  processor extracts frames internally at 1fps, saves jpgs to cache dir.
                   We patch max_frame_num after load to honour our config value.

Audio strategy:
  Inputs may be .m4a; we convert to 16 kHz mono wav with ffmpeg, then run the
  same 30s-chunk pipeline as below. Temporary wavs are removed after generate().
  Like OLA, Baichuan's audio encoder is Whisper-based (max_audio_seconds=30).
  Their processor hard-truncates at 30s. We override this by pre-processing the
  audio ourselves: uniformly sample up to max_chunks windows of 30s each,
  concatenate into a trimmed wav, write to cache dir, and pass that path to the
  processor. This gives us the same uniform-sampling coverage as OLA/VITA.

Input format (raw string — processor handles all tensor prep internally):
    <B_SYS>{system}<C_Q>
    <video_start_baichuan>{"local": "/abs/path.mp4"}<video_end_baichuan>
    <audio_start_baichuan>{"path": "/abs/path.wav"}<audio_end_baichuan>  (.m4a converted to wav first)
    <audiotext_start_baichuan>{prompt}<C_A>

Text-only output: stop at audiogen_start_token_id=151700. No vocoder loaded.

Author: SONIC-O1 Team
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import torch
import torchaudio
import re

from .base_model import BaseModel


logger = logging.getLogger(__name__)

# ── Role/tag constants (from web_demo/constants.py) ───────────────────────────
SYS_START   = "<B_SYS>"
USER_START  = "<C_Q>"
ASST_START  = "<C_A>"
AUDIOTEXT   = "<audiotext_start_baichuan>"
VIDEO_START = "<video_start_baichuan>"
VIDEO_END   = "<video_end_baichuan>"
AUDIO_START = "<audio_start_baichuan>"
AUDIO_END   = "<audio_end_baichuan>"

# Audio encoder constants (from config.json: audio_config)
AUDIO_SR      = 16000           # sampling_rate
CHUNK_SAMPLES = 30 * AUDIO_SR   # max_audio_seconds=30 → 480 000 samples


class BaichuanOmni(BaseModel):
    """Baichuan-Omni-1.5 wrapper following BaseModel pattern."""

    def __init__(self, model_name: str, config: Dict[str, Any]) -> None:
        super().__init__(model_name, config)

        self.model_path         = config.get("model_path", "baichuan-inc/Baichuan-Omni-1.5")
        self.baichuan_repo_path = config.get("baichuan_repo_path")

        # Cache dir — processor writes extracted video frame jpgs here.
        # Must be absolute path on persistent storage (not $SCRATCH).
        self.cache_dir = config.get(
            "cache_dir",
            "/projects/aixpert/users/ahmadradw/VideoQA-Agentic/sonic-o1/.cache",
        )

        # Frame config — patched into model.config.video_config.max_frame_num after load
        self.default_max_frames = config.get("max_frames", 32)
        self.default_min_frames = config.get("min_frames", 8)

        # Generation config
        gen_config = config.get("generation_config", {})
        self.temperature    = gen_config.get("temperature",    0.7)
        self.top_p          = gen_config.get("top_p",          0.95)
        self.num_beams      = gen_config.get("num_beams",       1)
        self.max_new_tokens = gen_config.get("max_new_tokens",  8192)

        # Model components (populated in load())
        self.tokenizer = None
        self.model     = None

        # Stats
        self.stats = {
            "total_samples":        0,
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
                
    # ─────────────────────────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load Baichuan-Omni-1.5 and bind processor."""
        # Add repo + model subdir to sys.path so trust_remote_code resolves
        # modeling_omni.py, processor_omni.py, configuration_omni.py etc.
        if self.baichuan_repo_path:
            repo_abs  = os.path.abspath(self.baichuan_repo_path)
            model_dir = os.path.join(repo_abs, "baichuan-omni", "model")
            for p in [repo_abs, model_dir]:
                if p not in sys.path:
                    sys.path.insert(0, p)
                    logger.info(f"Added to sys.path: {p}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = os.path.abspath(self.model_path)
        logger.info(f"Loading Baichuan-Omni-1.5 from {model_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.model.training = False

        # bind_processor attaches OmniMMProcessor to model.processor.
        # relative_path = cache dir where processor saves video frame jpgs.
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model.bind_processor(
            self.tokenizer,
            training=False,
            relative_path=self.cache_dir,
        )

        # Patch frame cap to honour our config (config.json default is 32).
        self.model.config.video_config.max_frame_num = self.default_max_frames
        logger.info(f"Patched video_config.max_frame_num = {self.default_max_frames}")

        self.model.eval()
        logger.info("Baichuan-Omni-1.5 loaded successfully")

    # ─────────────────────────────────────────────────────────────────────────
    # Audio pre-processing
    # ─────────────────────────────────────────────────────────────────────────

    def _m4a_to_wav(self, m4a_path: str) -> str:
        """
        Decode m4a to a temporary wav (16 kHz mono) via ffmpeg.
        torchaudio often cannot load m4a reliably; Baichuan expects a wav path.
        Caller must delete the returned path when done.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        out_name = f"baichuan_m4a_conv_{os.getpid()}_{abs(hash(m4a_path)) % 10**8}.wav"
        out_path = os.path.join(self.cache_dir, out_name)
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            m4a_path,
            "-ac",
            "1",
            "-ar",
            str(AUDIO_SR),
            "-f",
            "wav",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "ffmpeg is required to convert .m4a audio; install ffmpeg and retry."
            ) from e
        except subprocess.CalledProcessError as e:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            err = (e.stderr or e.stdout or "").strip()
            raise RuntimeError(f"ffmpeg m4a→wav failed: {err or e}") from e
        logger.info(f"Converted m4a → wav: {out_path}")
        return out_path

    def _maybe_convert_m4a(
        self, audio_path: str
    ) -> tuple[str, Optional[str]]:
        """
        If ``audio_path`` is .m4a, convert to wav and return
        (path_to_wav, temp_path_to_delete). Otherwise return (audio_path, None).
        """
        if Path(audio_path).suffix.lower() != ".m4a":
            return audio_path, None
        converted = self._m4a_to_wav(audio_path)
        return converted, converted

    def _audio_has_stream(self, audio_path: str) -> bool:
        """Return True if ffprobe sees at least one audio stream."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=nw=1:nk=1",
            audio_path,
        ]
        try:
            out = subprocess.run(
                cmd, check=False, capture_output=True, text=True
            ).stdout.strip()
            return bool(out)
        except Exception:
            return False

    def _prepare_audio(
        self,
        audio_path: str,
        max_chunks: Optional[int],
    ) -> str:
        """
        Uniformly sample up to max_chunks × 30s windows from the full audio,
        concatenate, write a trimmed wav to cache dir, return its absolute path.

        Like OLA, we use 30s windows because Baichuan's audio encoder is
        Whisper-based (max_audio_seconds=30 in config.json). Their processor
        would hard-truncate at 30s — we override that here.

        ``audio_path`` should be a format ``torchaudio.load`` can read (e.g. wav);
        use :meth:`_maybe_convert_m4a` first for .m4a inputs.
        """
        waveform, sr = torchaudio.load(audio_path)

        # Resample to 16 kHz if needed
        if sr != AUDIO_SR:
            waveform = torchaudio.functional.resample(waveform, sr, AUDIO_SR)

        # Downmix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)  # [samples]
        total_samples = waveform.shape[0]
        logger.info(f"Audio: {total_samples / AUDIO_SR:.1f}s total")

        # Build non-overlapping 30s chunks
        all_chunks = []
        for start in range(0, total_samples, CHUNK_SAMPLES):
            chunk = waveform[start: start + CHUNK_SAMPLES]
            if chunk.shape[0] < CHUNK_SAMPLES:
                chunk = torch.nn.functional.pad(
                    chunk, (0, CHUNK_SAMPLES - chunk.shape[0])
                )
            all_chunks.append(chunk)

        total_chunks = len(all_chunks)
        logger.info(f"Audio: {total_chunks} chunk(s) of 30s")

        # Uniform sampling if over budget
        if max_chunks is not None and total_chunks > max_chunks:
            indices = np.linspace(0, total_chunks - 1, max_chunks, dtype=int)
            sampled = [all_chunks[i] for i in indices]
            logger.info(f"Uniformly sampled {max_chunks}/{total_chunks} audio chunks")
            self.stats["audio_chunks_sampled"] += 1
        else:
            sampled = all_chunks
            logger.info(f"Using all {total_chunks} audio chunk(s)")

        # Concatenate and write to cache
        trimmed  = torch.cat(sampled, dim=0).unsqueeze(0)  # [1, samples]
        out_name = f"baichuan_audio_{os.getpid()}_{abs(hash(audio_path)) % 10**8}.wav"
        out_path = os.path.join(self.cache_dir, out_name)
        torchaudio.save(out_path, trimmed, AUDIO_SR)
        logger.info(f"Trimmed audio → {out_path}")
        return out_path

    # ─────────────────────────────────────────────────────────────────────────
    # Message builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_message(
        self,
        video_path: Optional[str],
        audio_path: Optional[str],
        prompt: str,
        system: str = "You are a helpful assistant.",
    ) -> str:
        """
        Build the raw input string the OmniMMProcessor expects.

        Format (from traced web_demo/s2s_gradio_demo_cosy_multiturn.py):
            <B_SYS>{system}<C_Q>
            <video_start_baichuan>{"local": "..."}<video_end_baichuan>   ← optional
            <audio_start_baichuan>{"path":  "..."}<audio_end_baichuan>   ← optional
            <audiotext_start_baichuan>{prompt}<C_A>
        """
        msg = SYS_START + system + USER_START

        if video_path is not None:
            msg += VIDEO_START + json.dumps({"local": video_path}) + VIDEO_END

        if audio_path is not None:
            msg += AUDIO_START + json.dumps({"path": audio_path}) + AUDIO_END

        # <audiotext_start_baichuan> signals the model to respond in text mode.
        # Must be present even for text-only output — matches training format.
        msg += AUDIOTEXT + prompt + ASST_START
        return msg

    # ─────────────────────────────────────────────────────────────────────────
    # Generate
    # ─────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        frames: Union[str, Path, None] = None,
        audio: Optional[Union[str, Path]] = None,
        prompt: str = "Describe what you see and hear.",
        fps: Optional[float] = None,
        video_category: Optional[Literal["short", "medium", "long"]] = None,
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate a text response from video and/or audio.

        Args:
            frames:           Path to video file (.mp4), or None.
            audio:            Path to audio file (.m4a / .wav), or None.
            prompt:           Text question/prompt.
            fps:              Unused — processor samples at 1fps internally.
            video_category:   Unused; reserved for future frame budgeting.
            max_frames:       Per-call frame cap (patches model config).
            max_audio_chunks: Max 30s audio chunks (None = no limit).
            **kwargs:         temperature, top_p, num_beams, max_new_tokens.
        """
        temperature    = kwargs.get("temperature",    self.temperature)
        top_p          = kwargs.get("top_p",          self.top_p)
        num_beams      = kwargs.get("num_beams",       self.num_beams)
        max_new_tokens = kwargs.get("max_new_tokens",  self.max_new_tokens)

        # Per-call frame cap override
        if max_frames is not None and max_frames != self.default_max_frames:
            self.model.config.video_config.max_frame_num = max_frames
            logger.info(f"Per-call frame override: max_frame_num={max_frames}")

        self.stats["total_samples"] += 1
        tmp_audio_path = None       # trimmed wav from _prepare_audio
        m4a_converted_path = None   # intermediate wav from m4a conversion

        try:
            # ── 1. Video ──────────────────────────────────────────────────────
            has_video  = frames is not None and os.path.exists(str(frames))
            video_path = os.path.abspath(str(frames)) if has_video else None
            if has_video:
                logger.info(f"Video: {Path(frames).name}")

                compatible = self._check_video_compatibility(Path(video_path))
                if compatible is None:
                    raise RuntimeError(f"Video codec incompatible with Decord: {video_path}")

                video_path = str(compatible)  # use converted path going into processor
            # ── 2. Audio ──────────────────────────────────────────────────────
            has_audio = audio is not None and os.path.exists(str(audio))
            if has_audio and not self._audio_has_stream(str(audio)):
                logger.warning(
                    f"Audio file has no decodable stream, falling back to video-only: {audio}"
                )
                has_audio = False
            if has_audio:
                audio_for_prepare, m4a_converted_path = self._maybe_convert_m4a(
                    str(audio)
                )
                tmp_audio_path = self._prepare_audio(
                    audio_for_prepare, max_audio_chunks
                )

            # ── 3. Build message string and run processor ──────────────────────
            message = self._build_message(video_path, tmp_audio_path, prompt)
            logger.info("Running processor...")

            # processor([str]) → batch mode → OmniProcessorOutput
            ret = self.model.processor([message])

            # ── 4. Move tensors to GPU ────────────────────────────────────────
            input_ids      = ret.input_ids.cuda()
            attention_mask = ret.attention_mask.cuda()

            audios         = ret.audios.cuda()         if ret.audios         is not None else None
            encoder_length = ret.encoder_length.cuda() if ret.encoder_length is not None else None
            bridge_length  = ret.bridge_length.cuda()  if ret.bridge_length  is not None else None

            # Static images: not used (video frames handled via videos= field)
            images      = None
            patch_nums  = None
            images_grid = None

            videos = (
                [torch.tensor(v, dtype=torch.float32).cuda() for v in ret.videos]
                if ret.videos is not None else None
            )
            videos_patch_nums = ret.videos_patch_nums if ret.videos_patch_nums is not None else None
            videos_grid       = ret.videos_grid       if ret.videos_grid       is not None else None

            # ── 5. Generate (text only — stop before audio generation) ─────────
            logger.info(
                f"Generating: video={'yes' if has_video else 'no'}, "
                f"audio={'yes' if has_audio else 'no'}"
            )

            with torch.inference_mode():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audios=audios,
                    images=images,
                    patch_nums=patch_nums,
                    images_grid=images_grid,
                    videos=videos,
                    videos_patch_nums=videos_patch_nums,
                    videos_grid=videos_grid,
                    encoder_length=encoder_length,
                    bridge_length=bridge_length,
                    tokenizer=self.tokenizer,
                    # Stop before TTS generation — text output only
                    stop_strings=["<audiogen_start_baichuan>"],
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    num_beams=num_beams,
                    return_dict_in_generate=True,
                    use_cache=True,
                )

            # ── 6. Decode new tokens only ─────────────────────────────────────
            input_len = input_ids.shape[1]
            response  = self.tokenizer.decode(
                output.sequences[0, input_len:],
                skip_special_tokens=True,
            ).strip()

            # Clean control characters that break JSON parsing
            response = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', response)
            # Truncate if excessively long (Baichuan sometimes runs away)
            if len(response) > 15000:
                # Find last complete JSON structure
                last_brace = response.rfind('}')
                if last_brace > 0:
                    response = response[:last_brace + 1]

            logger.info(f"Generated {len(response)} characters")
            return self.postprocess_output(response)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "size of tensor" in error_msg.lower():
                logger.error(f"OOM: {error_msg[:200]}...")
                torch.cuda.empty_cache()
                raise RuntimeError(f"Out of memory: {e}")
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")

        finally:
            # Remove trimmed wav from _prepare_audio and any m4a→wav intermediate
            for p in (tmp_audio_path, m4a_converted_path):
                if p is not None and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    def unload(self) -> None:
        """Unload model and free GPU memory."""
        logger.info("Unloading Baichuan-Omni model...")

        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Baichuan-Omni unloaded")

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "model_path":         self.model_path,
            "backbone":           "Qwen2.5-7B (bfloat16)",
            "vision_encoder":     "CLIP ViT-L/14 (1fps, max_frame_num patched from config)",
            "audio_encoder":      "Whisper-large (30s window, uniform sampling — same as OLA)",
            "native_video":       True,
            "native_audio":       True,
            "default_max_frames": self.default_max_frames,
            "cache_dir":          self.cache_dir,
            "statistics":         self.stats,
        })
        return info
