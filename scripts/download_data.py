"""Download the public gazeRE dataset into data/ (no git needed).

    python scripts/download_data.py              # pinned commit used for all results of this repository
    python scripts/download_data.py --dest /some/folder
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gaze_relevance.data import find_gazere_data          # noqa: E402
from gaze_relevance.download import download_gazere       # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=None, help="target folder (default: <repository>/data)")
    ap.add_argument("--force", action="store_true", help="download again even if present")
    a = ap.parse_args()
    root = download_gazere(a.dest, force=a.force)
    print("participant folders found at:", find_gazere_data(root))
