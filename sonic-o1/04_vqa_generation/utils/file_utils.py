"""file_utils.py.

Shared file I/O helpers for VQA generation scripts.

Author: SONIC-O1 Team
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict


logger = logging.getLogger(__name__)


def save_json_with_backup(
    data: Dict[str, Any],
    json_path: Path,
    backup_suffix: str = ".json.backup",
) -> None:
    """Write a backup of *json_path* then overwrite it with *data*.

    Args:
        data: JSON-serialisable data to write.
        json_path: Destination path (original file to overwrite).
        backup_suffix: Suffix appended to *json_path* for the backup file.
            Defaults to ".json.backup".

    Raises
    ------
        Exception: Re-raises any I/O error after logging it.
    """
    try:
        backup_path = json_path.with_suffix(backup_suffix)
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Created backup: %s", backup_path.name)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to save %s: %s", json_path, e)
        raise
