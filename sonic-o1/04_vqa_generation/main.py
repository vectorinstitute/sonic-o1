"""main.py.

Main VQA generation script. Generates summarization, MCQ, and temporal
localization tasks using Gemini-based multimodal models.

Usage:
    python main.py --topics 1,2,3
    python main.py --all
    python main.py --topics 1 --task summarization

Author: SONIC-O1 Team
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
from utils.config_utils import Config, load_config


# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logging.info("Loaded environment variables from .env file")
except ImportError:
    pass

from models import MCQModel, SummarizationModel, TemporalLocalizationModel


# Setup logging
logger = logging.getLogger(__name__)

# Registry of all tasks: (cli_key, model_class, output_dir_name, multi_entry)
# multi_entry=True  -- model.process_video() returns List[Dict] (one per segment)
# multi_entry=False -- model.process_video() returns Dict    (one per video)
TASKS = [
    ("summarization", SummarizationModel, "task1_summarization", False),
    ("mcq", MCQModel, "task2_mcq", True),
    ("temporal", TemporalLocalizationModel, "task3_temporal_localization", True),
]


def load_metadata_for_topic(topic_dir: Path) -> List[Dict[str, Any]]:
    """
    Load metadata_enhanced.json for a topic directory.

    Args:
        topic_dir: Path to topic directory (e.g. dataset/videos/01_.../)

    Returns
    -------
        List of video metadata dicts
    """
    metadata_file = topic_dir / "metadata_enhanced.json"

    if not metadata_file.exists():
        logger.warning("No metadata_enhanced.json found in %s", topic_dir)
        return []

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)

        logger.info("Loaded %d videos from %s", len(metadata_list), topic_dir.name)
        return metadata_list

    except Exception as e:
        logger.error("Failed to load metadata from %s: %s", topic_dir, e)
        return []


def get_file_paths(video_meta: Dict[str, Any], topic_dir: Path) -> Dict[str, Path]:
    """
    Get file paths for video, audio, and transcript.

    Args:
        video_meta: Video metadata dict
        topic_dir: Topic directory path

    Returns
    -------
        Dict with keys: video_path, audio_path, transcript_path
    """
    video_number = video_meta.get("video_number", video_meta.get("video_id", "001"))

    # Video path
    video_filename = f"video_{video_number}.mp4"
    video_path = topic_dir / video_filename

    # Audio path (in parent audios directory)
    audio_dir = topic_dir.parent.parent / "audios" / topic_dir.name
    audio_filename = f"audio_{video_number}.m4a"
    audio_path = audio_dir / audio_filename

    # Transcript path (in parent captions directory)
    captions_dir = topic_dir.parent.parent / "captions" / topic_dir.name
    transcript_filename = f"caption_{video_number}.srt"
    transcript_path = captions_dir / transcript_filename

    return {
        "video_path": video_path if video_path.exists() else None,
        "audio_path": audio_path if audio_path.exists() else None,
        "transcript_path": (transcript_path if transcript_path.exists() else None),
    }


def get_confidence(entry: Dict[str, Any]) -> float:
    """Return the confidence score of a VQA entry."""
    return float(entry.get("confidence", 0))


def get_summary_detailed(entry: Dict[str, Any]) -> str:
    """Return the detailed summary text of a Task 1 entry."""
    return entry.get("summary_detailed", "")


_SUMMARY_FAIL_PATTERNS = (
    "unavailable",
    "summary generation failed",
    "could not be generated",
    "summary failed",
)


def get_summary_failed(entry: Dict[str, Any]) -> bool:
    """Return True if a Task 1 entry contains known failure markers."""
    for item in entry.get("summary_short", []):
        if isinstance(item, str):
            lower = item.lower()
            if any(p in lower for p in _SUMMARY_FAIL_PATTERNS):
                return True
            if "first segment" in lower and "failed" in lower:
                return True

    lower_detailed = get_summary_detailed(entry).lower()
    return (
        "could not be generated" in lower_detailed
        or "summary generation failed" in lower_detailed
        or "parsing error" in lower_detailed
        or "explicitly reported a failure" in lower_detailed
        or ("failed to" in lower_detailed and "summary" in lower_detailed)
    )


def skip_task(task_name: str, existing: Dict, video_id: str) -> bool:
    """Return True if this video already has valid, complete entries for the task."""
    if video_id not in existing:
        return False

    if task_name == "task1_summarization":
        entry = existing[video_id]
        if get_summary_failed(entry):
            logger.info("Reprocessing Task 1 for %s (previous failure)", video_id)
            return False
        if get_confidence(entry) == 0:
            logger.info("Reprocessing Task 1 for %s (confidence was 0)", video_id)
            return False
        logger.info("Skipping Task 1 for %s (already processed)", video_id)
        return True

    # Task 2 / Task 3 — list of segment entries per video
    segment_entries = existing[video_id]
    if segment_entries and all(get_confidence(e) > 0 for e in segment_entries):
        logger.info("Skipping %s for %s (already processed)", task_name, video_id)
        return True
    return False


def _apply_rate_limit(video_category: str, config: Config) -> None:
    """Sleep between videos according to config rate limits."""
    delay = int(getattr(config.rate_limit, "delay_between_videos", 15))
    if video_category == "long":
        delay += int(getattr(config.rate_limit, "delay_after_long_video", 60))
        logger.info("Long video — waiting %ss before next", delay)
    else:
        logger.info("Waiting %ss before next video (rate limit)", delay)
    time.sleep(delay)


def process_topic(
    topic_id: int,
    topic_name: str,
    topic_dir: Path,
    task_name: str,
    model,
    existing: Dict,
    config: Config,
    multi_entry: bool = False,
) -> List[Dict[str, Any]]:
    """Process all videos in a topic for a single task.

    Args:
        topic_id: Topic ID (1-13).
        topic_name: Human-readable topic name.
        topic_dir: Path to topic video directory.
        task_name: Output subdirectory name, e.g. "task1_summarization".
        model: Instantiated task model.
        existing: Pre-loaded existing entries indexed by video_id.
        config: Configuration object.
        multi_entry: True when model.process_video() returns List[Dict]
            (MCQ/temporal); False when it returns a single Dict (summarization).

    Returns
    -------
        List of VQA entry dicts for this task across all videos.
    """
    metadata_list = load_metadata_for_topic(topic_dir)
    if not metadata_list:
        logger.warning("No videos found for topic %s", topic_id)
        return []

    entries: List[Dict[str, Any]] = []

    for video_meta in tqdm(metadata_list, desc=f"Topic {topic_id} / {task_name}"):
        video_id = video_meta.get("video_id", video_meta.get("video_number", "unknown"))
        duration = video_meta.get("duration_seconds", 0)
        video_category = video_meta.get("duration_category", "")
        if video_category not in ("short", "medium", "long"):
            video_category = (
                "short" if duration <= 300 else "medium" if duration <= 1800 else "long"
            )

        if skip_task(task_name, existing, video_id):
            cached = existing[video_id]
            entries.extend([cached] if not multi_entry else cached)
            continue

        try:
            video_meta["topic_id"] = topic_id
            video_meta["topic_name"] = topic_name
            file_paths = get_file_paths(video_meta, topic_dir)

            if not file_paths["video_path"] and not file_paths["audio_path"]:
                logger.warning("No video or audio for %s, skipping", video_id)
                continue

            logger.info("Generating %s for %s", task_name, video_id)
            new = model.process_video(
                video_path=file_paths["video_path"],
                audio_path=file_paths["audio_path"],
                transcript_path=file_paths["transcript_path"],
                metadata=video_meta,
            )

            if multi_entry:
                new_list = new if isinstance(new, list) else [new]
                if video_id in existing and existing[video_id]:
                    merged, kept, replaced = merge_entries_keep_good(
                        existing[video_id], new_list
                    )
                    entries.extend(merged)
                    logger.info(
                        "%s for %s: kept %d good, replaced %d failed",
                        task_name,
                        video_id,
                        kept,
                        replaced,
                    )
                else:
                    entries.extend(new_list)
            else:
                entries.append(new)

            _apply_rate_limit(video_category, config)

        except Exception as e:
            logger.error("Failed to process %s: %s", video_id, e, exc_info=True)

    logger.info(
        "Completed %s for Topic %s: %d entries", task_name, topic_id, len(entries)
    )
    return entries


def get_topic_output_path(
    output_dir: Path, task_name: str, topic_id: int, topic_name: str
) -> Path:
    """
    Build the output JSON path for a given task and topic.

    Args:
        output_dir: Root output directory (e.g., vqa/)
        task_name: e.g. "task1_summarization"
        topic_id: Topic ID
        topic_name: Topic name (spaces allowed; converted to underscores)

    Returns
    -------
        Full Path to the output JSON file.
    """
    filename = f"{topic_id:02d}_{topic_name.replace(' ', '_')}.json"
    return output_dir / task_name / filename


def init_task(
    task_filter_key: str,
    model_class,
    task_name: str,
    config: "Config",
    output_dir: Path,
    topic_id: int,
    topic_name: str,
    task_filter: str,
    list_per_video: bool = False,
    dry_run: bool = False,
) -> tuple:
    """
    Initialize a task model and load any pre-existing output entries.

    Args:
        task_filter_key: Filter string for this task, e.g. "summarization".
        model_class: Model class to instantiate, e.g. SummarizationModel.
        task_name: Output subdirectory name, e.g. "task1_summarization".
        config: Configuration object.
        output_dir: Root output directory.
        topic_id: Topic ID.
        topic_name: Topic name.
        task_filter: Active CLI filter (None = run all tasks).
        list_per_video: If True, existing entries are indexed as
            dict[video_id -> list]; if False, dict[video_id -> entry].

    Returns
    -------
        Tuple of (model_instance_or_None, existing_entries_dict).
        model is None when this task is excluded by task_filter.
    """
    if task_filter is not None and task_filter != task_filter_key:
        return None, {}

    model = model_class(config, dry_run=dry_run)
    existing = {}

    output_file = get_topic_output_path(output_dir, task_name, topic_id, topic_name)
    if output_file.exists():
        with open(output_file, "r") as f:
            data = json.load(f)
        for entry in data.get("entries", []):
            vid = entry["video_id"]
            if list_per_video:
                existing.setdefault(vid, []).append(entry)
            else:
                existing[vid] = entry
        n = sum(len(v) for v in existing.values()) if list_per_video else len(existing)
        logger.info("Loaded %d existing %s entries", n, task_name)

    return model, existing


def save_task_results(
    task_name: str,
    topic_id: int,
    topic_name: str,
    entries: List[Dict[str, Any]],
    output_dir: Path,
):
    """
    Save VQA entries to JSON file.

    Args:
        task_name: e.g. "task1_summarization" or "task2_mcq"
        topic_id: Topic ID
        topic_name: Topic name
        entries: List of VQA entry dicts
        output_dir: Output directory (e.g., vqa/)
    """
    if not entries:
        logger.warning("No entries to save for %s - %s", task_name, topic_name)
        return

    output_file = get_topic_output_path(output_dir, task_name, topic_id, topic_name)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "task": task_name.split("_", 1)[1] if "_" in task_name else task_name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_entries": len(entries),
        "entries": entries,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d entries to %s", len(entries), output_file)


def get_all_topic_dirs(dataset_root: Path) -> List[tuple]:
    """
    Get all topic directories.

    Returns
    -------
        List of tuples: (topic_id, topic_name, topic_dir_path)
    """
    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        return []

    topics = []
    for topic_dir in sorted(videos_dir.iterdir()):
        if topic_dir.is_dir() and topic_dir.name[0].isdigit():
            # Extract topic ID and name from dir (e.g. 01_Patient-Doctor_...)
            parts = topic_dir.name.split("_", 1)
            if len(parts) == 2:
                topic_id = int(parts[0])
                topic_name = parts[1].replace("_", " ")
                topics.append((topic_id, topic_name, topic_dir))

    return topics


def merge_entries_keep_good(
    existing_entries: List[Dict], new_entries: List[Dict]
) -> List[Dict]:
    """
    Merge existing and new entries.

    - Keeping existing entries with confidence > 0
    - Replacing existing entries with confidence 0.0 with matching new entries
    - Adding new entries for segments that don't exist in existing.

    Args:
        existing_entries: List of existing entries for a video_id
        new_entries: List of newly generated entries

    Returns
    -------
        Merged list with good existing + new replacements for failed
    """
    merged = []

    # Keep all existing entries with confidence > 0
    for existing in existing_entries:
        if existing.get("confidence", 0) > 0:
            merged.append(existing)

    # Replace failed entries (confidence 0) with new if segment matches
    failed_segments = {
        (e.get("segment", {}).get("start"), e.get("segment", {}).get("end")): e
        for e in existing_entries
        if e.get("confidence", 0) == 0.0
    }

    new_segments_used = set()

    # Replace failed segments with new entries
    for new in new_entries:
        new_seg = (
            new.get("segment", {}).get("start"),
            new.get("segment", {}).get("end"),
        )

        if new_seg in failed_segments:
            merged.append(new)
            new_segments_used.add(new_seg)
        else:
            exist_segs = [
                (e.get("segment", {}).get("start"), e.get("segment", {}).get("end"))
                for e in existing_entries
            ]
            if new_seg not in exist_segs:
                merged.append(new)
                new_segments_used.add(new_seg)

    # Log what happened
    replaced = len(set(failed_segments.keys()).intersection(new_segments_used))
    kept_good = sum(1 for e in existing_entries if e.get("confidence", 0) > 0)

    return merged, kept_good, replaced


def main():
    """Run main entry point."""
    parser = argparse.ArgumentParser(description="VQA Generation System")
    parser.add_argument("--config", type=str, default="vqa_config.yaml")
    parser.add_argument(
        "--topics",
        type=str,
        default=None,
        help='Comma-separated topic IDs (e.g., "1,2,3")',
    )
    parser.add_argument("--all", action="store_true", help="Process all topics")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["summarization", "mcq", "temporal"],
        help="Process only a specific task (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without API calls; generates stub outputs",
    )
    args = parser.parse_args()

    config = load_config(args.config, base_dir=Path(__file__).parent)

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("[DRY-RUN] No API calls will be made; outputs are stubs")
        logger.info("=" * 60)

    output_dir = Path(args.output) if args.output else Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(config.paths.dataset_root)
    if not dataset_root.exists():
        logger.error("Dataset root not found: %s", dataset_root)
        return

    all_topics = get_all_topic_dirs(dataset_root)
    logger.info("Found %d topics in dataset", len(all_topics))

    if args.topics:
        topic_ids = {int(t.strip()) for t in args.topics.split(",")}
        topics_to_process = [t for t in all_topics if t[0] in topic_ids]
    elif args.all:
        topics_to_process = all_topics
    else:
        logger.error("Must specify either --topics or --all")
        return

    logger.info("Processing %d topics", len(topics_to_process))

    totals = {task_name: 0 for _, _, task_name, _ in TASKS}

    for topic_id, topic_name, topic_dir in topics_to_process:
        for task_key, model_class, task_name, multi_entry in TASKS:
            if args.task and args.task != task_key:
                continue
            try:
                model, existing = init_task(
                    task_key,
                    model_class,
                    task_name,
                    config,
                    output_dir,
                    topic_id,
                    topic_name,
                    task_filter=None,
                    list_per_video=multi_entry,
                    dry_run=args.dry_run,
                )
                entries = process_topic(
                    topic_id,
                    topic_name,
                    topic_dir,
                    task_name,
                    model,
                    existing,
                    config,
                    multi_entry,
                )
                if not args.dry_run:
                    save_task_results(
                        task_name, topic_id, topic_name, entries, output_dir
                    )
                else:
                    logger.info(
                        "[DRY-RUN] Would save %d entries for %s",
                        len(entries),
                        task_name,
                    )
                totals[task_name] += len(entries)
            except Exception as e:
                logger.error(
                    "Failed %s for topic %s: %s", task_name, topic_id, e, exc_info=True
                )

    logger.info("=" * 60)
    logger.info("VQA Generation Complete!")
    logger.info("Topics processed: %d", len(topics_to_process))
    for _, _, task_name, _ in TASKS:
        logger.info("  %s: %d entries", task_name, totals[task_name])
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
