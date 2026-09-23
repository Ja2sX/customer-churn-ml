"""Download and load the IBM Telco Customer Churn sample.

The only network call in this project is an unauthenticated HTTPS GET of that
public CSV. No API key, account, or paid dataset service is required.
"""

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from churn.config import DATA_PATH, DATA_SHA256, DATA_URL


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(
    path: Path = DATA_PATH,
    url: str = DATA_URL,
    expected_sha256: str = DATA_SHA256,
) -> Path:
    """Download the CSV when it is missing or does not match the pinned digest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and file_sha256(path) == expected_sha256:
        return path

    partial = path.with_suffix(path.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, partial)
        actual = file_sha256(partial)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Downloaded file sha256 {actual} does not match pinned {expected_sha256}. "
                "The upstream sample may have changed; update DATA_SHA256 after review."
            )
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()
    return path


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    path = download_dataset(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"Dataset at {path} has no rows.")
    return frame
