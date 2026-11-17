from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from football_action_valuation.constants import (
    WORLD_CUP_2022_COMPETITION_ID,
    WORLD_CUP_2022_SEASON_ID,
)

DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
)
USER_AGENT = "football-action-valuation/0.1 (educational open-data project)"


class DownloadError(RuntimeError):
    """Raised when a StatsBomb file cannot be downloaded."""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _download_file(url: str, destination: Path, *, force: bool = False) -> Path:
    """Download one file atomically, skipping existing files by default."""
    if destination.exists() and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise DownloadError(f"Unable to download {url}: {exc}") from exc

    return destination


def matches_path(raw_dir: Path, competition_id: int, season_id: int) -> Path:
    return raw_dir / "matches" / str(competition_id) / f"{season_id}.json"


def events_path(raw_dir: Path, match_id: int) -> Path:
    return raw_dir / "events" / f"{match_id}.json"


def download_competition_events(
    raw_dir: Path,
    *,
    competition_id: int = WORLD_CUP_2022_COMPETITION_ID,
    season_id: int = WORLD_CUP_2022_SEASON_ID,
    limit: int | None = None,
    workers: int = 4,
    force: bool = False,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, int]:
    """Download match metadata and event files for one open competition.

    Files are streamed to disk and matches are downloaded concurrently. The
    worker count stays deliberately small for laptops with limited memory.
    """
    raw_dir = raw_dir.resolve()
    metadata_path = matches_path(raw_dir, competition_id, season_id)
    metadata_url = f"{base_url}/matches/{competition_id}/{season_id}.json"
    _download_file(metadata_url, metadata_path, force=force)

    matches = read_json(metadata_path)
    if not isinstance(matches, list):
        raise ValueError(f"Expected a list in {metadata_path}")

    selected = matches[:limit] if limit is not None else matches
    downloaded = 0
    skipped = 0

    def fetch(match: dict[str, Any]) -> tuple[Path, bool]:
        match_id = int(match["match_id"])
        destination = events_path(raw_dir, match_id)
        already_present = destination.exists() and not force
        url = f"{base_url}/events/{match_id}.json"
        return _download_file(url, destination, force=force), already_present

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(fetch, match) for match in selected]
        for completed, future in enumerate(as_completed(futures), start=1):
            path, already_present = future.result()
            if already_present:
                skipped += 1
            else:
                downloaded += 1
            print(f"[{completed:>3}/{len(futures)}] {path.name}")

    return {
        "matches_available": len(matches),
        "matches_selected": len(selected),
        "downloaded": downloaded,
        "skipped": skipped,
    }

