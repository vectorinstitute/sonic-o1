"""omnivinci.py
OmniVinci (nvidia/omnivinci) implementation following BaseModel pattern.

Accepts separate video (.mp4) and audio (.m4a) paths, merges them into a
single mp4 via ffmpeg (stream-copy video, re-encode audio to aac), then
passes the merged file to OmniVinci's processor.

Audio is handled entirely internally by OmniVinci — no custom chunking needed.
Frame count is controlled via model.config / processor.config before each call
so that retry fallbacks (frame_count_fallback) take effect correctly.

Author: SONIC-O1 Team
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import torch

from .base_model import BaseModel


logger = logging.getLogger(__name__)


class OmniVinci(BaseModel):
    """
    OmniVinci wrapper following BaseModel pattern.

    - Video + audio are merged into a single mp4 before inference.
    - Frame count is set on model/processor config before each call.
    - Audio length is capped at max_3600 (1 hour) at load time — no chunking.
    - No repo path injection needed (pure HF trust_remote_code model).
    """

    def __init__(self, model_name: str, config: Dict[str, Any]) -> None:
        super().__init__(model_name, config)

        self.model_path = config.get("model_path", "nvidia/omnivinci")

        # modeling_vila.py does eval(torch_dtype) so must be a torch.* object
        self.device_map = config.get("device_map", "auto")
        dtype_cfg = config.get("torch_dtype", "float16")
        self.torch_dtype = getattr(torch, dtype_cfg) if isinstance(dtype_cfg, str) else dtype_cfg

        # Frame limits
        self.default_max_frames = config.get("max_frames", 128)
        self.default_min_frames = config.get("min_frames", 8)

        # Generation config
        gen_config = config.get("generation_config", {})
        self.temperature = gen_config.get("temperature", 0.7)
        self.top_p = gen_config.get("top_p", 0.95)
        self.max_new_tokens = gen_config.get("max_new_tokens", 8192)

        self.model = None
        self.processor = None
        self.generation_config = None
        self._temp_dir = None

    # ------------------------------------------------------------------ #
    #  Load / Unload                                                       #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load OmniVinci model and processor."""
        from transformers import AutoModel, AutoProcessor

        logger.info(f"Loading OmniVinci from {self.model_path}")

        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )
        self.model = self.model.to("cuda")
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.generation_config = self.model.default_generation_config
        self.generation_config.update(
            max_new_tokens=self.max_new_tokens,
            max_length=99999999,
        )

        # Audio: trust OmniVinci's internal pipeline, cap at 1 hour
        self.model.config.load_audio_in_video = True
        self.processor.config.load_audio_in_video = True
        self.model.config.audio_chunk_length = "max_3600"
        self.processor.config.audio_chunk_length = "max_3600"

        self._temp_dir = tempfile.mkdtemp(prefix="omnivinci_merged_")
        logger.info(f"OmniVinci loaded successfully (device_map={self.device_map})")

    def unload(self) -> None:
        """Unload model and free memory."""
        logger.info("Unloading OmniVinci model...")

        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if self._temp_dir and Path(self._temp_dir).exists():
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp dir: {self._temp_dir}")

        logger.info("OmniVinci unloaded")

    # ------------------------------------------------------------------ #
    #  Video helpers                                                       #
    # ------------------------------------------------------------------ #

    def convert_av1_to_h264(self, video_path: Path, output_dir: Optional[Path] = None) -> Path:
        """Convert AV1 video to H.264 for decoder compatibility."""
        if output_dir is None:
            output_dir = Path(self._temp_dir) / "converted"
            output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{video_path.stem}_h264{video_path.suffix}"
        if output_path.exists():
            logger.info(f"Using cached converted video: {output_path.name}")
            return output_path

        logger.info(f"Converting AV1 to H.264: {video_path.name}")
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy", "-y", str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"✓ Conversion successful: {output_path.name}")
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert AV1 video: {e.stderr[-200:]}")

    def _ensure_video_compatibility(self, video_path: Path) -> Path:
        """Convert AV1 → H.264 if needed; return compatible path."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                check=True, capture_output=True, text=True,
            )
            codec = result.stdout.strip().lower()
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not detect codec for {video_path.name}: {e}")
            return video_path

        if codec == "av1":
            logger.info("Detected AV1 codec — converting to H.264 before inference")
            return self.convert_av1_to_h264(video_path)
        return video_path

    def _has_usable_audio_stream(self, audio_path: Path) -> bool:
        """
        True if ffprobe finds at least one audio stream with positive duration.

        Segments can be saved as .m4a with no audio track (silent source,
        failed extract, or empty mux) — ffmpeg then errors on -map 1:a:0.
        Some broken extracts report a container but Duration 00:00:00.00;
        we require format duration > 0 as well.
        """
        try:
            if not audio_path.is_file() or audio_path.stat().st_size == 0:
                return False
        except OSError:
            return False
        r = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return False
        dur = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
        if dur.returncode != 0:
            return False
        try:
            if float((dur.stdout.strip() or "0").split()[0]) <= 0:
                return False
        except (ValueError, IndexError):
            return False
        return True

    def _merge_video_audio(self, video_path: Path, audio_path: Path) -> Path:
        """
        Merge video + audio into a single mp4 (cached across retries).

        On any ffmpeg failure (no audio stream, codec issue, corrupt segment, etc.),
        logs a warning and returns ``video_path`` so inference runs video-only.
        """
        output_path = Path(self._temp_dir) / f"merged_{video_path.stem}.mp4"
        if output_path.exists():
            logger.info(f"Using cached merged file: {output_path.name}")
            return output_path

        logger.info(f"Merging {video_path.name} + {audio_path.name} → {output_path.name}")
        cmd = [
            "ffmpeg",
            "-i", str(video_path), "-i", str(audio_path),
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0",
            "-y", str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"✓ Merge successful: {output_path.name}")
            return output_path
        except subprocess.CalledProcessError as e:
            err = (e.stderr or "") + (e.stdout or "")
            if output_path.exists():
                try:
                    output_path.unlink()
                except OSError:
                    pass
            tail = (err[-400:] if err else str(e))[:400]
            logger.warning(
                f"ffmpeg merge failed ({audio_path.name}); using video without merged audio. "
                f"Last output: {tail!r}"
            )
            return video_path

    # ------------------------------------------------------------------ #
    #  Inference helpers                                                   #
    # ------------------------------------------------------------------ #

    def _set_frame_count(self, num_frames: int) -> None:
        """Must be called before processor([text]) so retry fallback takes effect."""
        self.model.config.num_video_frames = num_frames
        self.processor.config.num_video_frames = num_frames
        logger.info(f"num_video_frames set to: {num_frames}")

    def _inference_device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def _move_nested_to_device(obj: Any, device: torch.device) -> Any:
        """Recursively move tensors in nested dicts/lists/tuples to device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: OmniVinci._move_nested_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            moved = [OmniVinci._move_nested_to_device(x, device) for x in obj]
            return type(obj)(moved)
        return obj

    @staticmethod
    def _patch_media(media: Optional[Any]) -> Optional[Any]:
        """
        modeling_vila.__embed_media_tokens unconditionally accesses
        media['speech'] regardless of unified_audio_encoder flag.
        When audio comes from video the processor only fills 'sound',
        leaving 'speech' absent → KeyError: 'speech'.
        Fix: ensure every expected key exists with an empty fallback.
        """
        if not isinstance(media, dict):
            return media

        patched = dict(media)

        # Keys VILA's embed loop iterates over — ensure they exist
        for key in ("speech", "sound", "image", "video", "vision", "frames", "audio"):
            if key not in patched:
                patched[key] = []

        logger.info(f"media keys after patch: {list(patched.keys())}")
        return patched

    # ------------------------------------------------------------------ #
    #  generate                                                            #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        frames: Union[str, Path],
        audio: Optional[Union[str, Path]] = None,
        prompt: str = "",
        fps: Optional[float] = None,
        video_category: Optional[Literal["short", "medium", "long"]] = None,
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,   # ignored — audio handled internally
        **kwargs,
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        actual_max_frames = max_frames if max_frames is not None else self.default_max_frames

        video_path = Path(frames)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        try:
            compatible_video = self._ensure_video_compatibility(video_path)

            if audio is not None:
                audio_path = Path(audio)
                if not audio_path.exists():
                    logger.warning(f"Audio not found: {audio_path} — proceeding video-only")
                    input_video = compatible_video
                elif not self._has_usable_audio_stream(audio_path):
                    logger.warning(
                        f"No audio stream in {audio_path.name} (empty/silent extract) — "
                        "proceeding video-only"
                    )
                    input_video = compatible_video
                else:
                    input_video = self._merge_video_audio(compatible_video, audio_path)
            else:
                logger.warning("No audio provided — proceeding video-only")
                input_video = compatible_video

            # Must be set BEFORE processor call
            self._set_frame_count(actual_max_frames)
            logger.info(f"Running OmniVinci on: {input_video.name} (frames={actual_max_frames})")

            conversation = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": str(input_video)},
                    {"type": "text", "text": prompt},
                ],
            }]

            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.processor([text])

            # Processor returns input_ids on CPU; model weights are on cuda.
            # media / media_config are handled internally by model.generate.
            inputs.input_ids = inputs.input_ids.to("cuda")

            max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
            self.generation_config.update(max_new_tokens=max_new_tokens)

            logger.info(f"Generating (max_tokens={max_new_tokens})...")

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids=inputs.input_ids,
                    media=getattr(inputs, "media", None),
                    media_config=getattr(inputs, "media_config", None),
                    generation_config=self.generation_config,
                )

            response = self.processor.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True,
            )[0].strip()

            logger.info(f"Generated response ({len(response)} chars)")
            return self.postprocess_output(response)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                logger.error(f"OOM: {error_msg[:200]}")
                torch.cuda.empty_cache()
                raise RuntimeError(f"Out of memory: {e}")
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")

    # ------------------------------------------------------------------ #
    #  Info                                                                #
    # ------------------------------------------------------------------ #

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "model_path": self.model_path,
                "model_type": "Omni Multimodal (VILAForCausalLM)",
                "device_map": self.device_map,
                "torch_dtype": str(self.torch_dtype),
                "default_max_frames": self.default_max_frames,
                "audio_handling": "Internal (max_3600) — no custom chunking",
                "input_format": "Merged mp4 (ffmpeg stream-copy video + aac audio)",
            }
        )
        return info