"""Download the public gazeRE dataset.

The dataset is hosted on GitHub (https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset, GPL-3.0).
By default the archive of a fixed commit is downloaded, so everybody analyses exactly the same files:

>>> from gaze_relevance.download import download_gazere
>>> download_gazere()            # -> data/gazeRE-dataset-<commit>/  (about 100 MB download, 265 MB unpacked)

No git installation is needed. If the download fails (e.g. no internet access), clone the repository manually and
point GAZERE_DATA at it; see data/README.md.
"""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from .config import REPO

GAZERE_REPOSITORY = "https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset"
GAZERE_COMMIT = "9af27c6ed771fd8ac5619d020f688d70cd446d26"      # the version used for all results in this repository


_last = [0]


def _progress(block, block_size, total):
    """Prints the progress every 20 MB (single lines, so that it also reads well in a saved notebook)."""
    done = block * block_size
    if block == 0:
        _last[0] = 0
    if done - _last[0] >= 20e6:
        _last[0] = done
        print(f"  {done / 1e6:.0f} MB" + (f" of {total / 1e6:.0f} MB" if total > 0 else ""), flush=True)


def download_gazere(dest: str | Path | None = None, commit: str = GAZERE_COMMIT, force: bool = False) -> Path:
    """Download and unpack the gazeRE repository at `commit` into `dest` (default: <repository>/data).
    Returns the path of the unpacked repository. Does nothing if it already exists (unless force=True)."""
    dest = Path(dest) if dest else REPO / "data"
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"gazeRE-dataset-{commit}"
    if target.exists() and not force:
        print(f"gazeRE dataset already present: {target.relative_to(dest.parent) if dest.parent in target.parents else target}")
        return target
    if target.exists():
        shutil.rmtree(target)
    url = f"{GAZERE_REPOSITORY}/archive/{commit}.zip"
    archive = dest / f"gazeRE-dataset-{commit}.zip"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, archive, _progress)
    print("  unpacking ...")
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    archive.unlink()
    print(f"  done: {target.name}")
    return target
