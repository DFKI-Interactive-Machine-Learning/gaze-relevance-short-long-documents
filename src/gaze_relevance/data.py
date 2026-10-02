"""Data access: find the gazeRE recordings anywhere below a folder, extract the 17 features with the official gazeRE
code, and build the per-paragraph visit table (including paragraphs with visits shorter than 3 s).

Typical use
-----------
>>> from gaze_relevance import data
>>> root = data.find_gazere_data()                 # searches $GAZERE_DATA and the usual places, recursively
>>> grel, nq = data.feature_tables(root)            # 288 g-REL trials, 1504 GoogleNQ paragraphs (cached)
>>> visits = data.paragraph_visits(root)            # all 1680 GoogleNQ paragraphs + 288 g-REL trials (cached)
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR, F17, MIN_VISIT_SECONDS, REPO, SCREEN_H, SCREEN_W
from .evaluation import add_reader_normalisation

PARTICIPANT = re.compile(r"^[AB]\d{2}$")          # gazeRE participant folders: A01 ... B13
CORPUS_DIRS = {"g-rel", "googlenq"}               # case-insensitive corpus sub-folders of each participant
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- recursive discovery
def _is_data_dir(path: Path) -> bool:
    """A gazeRE data folder contains participant folders, each with g-rel/ and GoogleNQ/ sub-folders that hold
    reading recordings (*.csv) and a User_Rating file."""
    try:
        parts = [p for p in path.iterdir() if p.is_dir() and PARTICIPANT.match(p.name)]
    except (PermissionError, OSError):
        return False
    if not parts:
        return False
    for p in parts:
        subs = {s.name.lower(): s for s in p.iterdir() if s.is_dir()}
        if not CORPUS_DIRS <= set(subs):
            return False
        if not all((subs[c] / "User_Rating").exists() for c in CORPUS_DIRS):
            return False
    return True


def find_gazere_data(*roots, max_depth: int = 6) -> Path:
    """Search recursively for the gazeRE data folder.

    Looks in the given roots (if any), else in $GAZERE_DATA, then <repo>/data, <repo>/gazeRE-dataset,
    <repo>/../gazeRE-dataset (i.e. next to this repository), and finally the current folder. Each root is searched
    breadth-first up to `max_depth` levels, so you can pass the cloned
    repository, its data/ folder or any parent folder. Returns the folder that directly contains A01/, A03/, ...
    """
    if not roots:
        env = os.environ.get("GAZERE_DATA")
        roots = tuple(r for r in [env, REPO / "data", REPO / "gazeRE-dataset", REPO.parent / "gazeRE-dataset", "."] if r)
    tried = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        tried.append(str(root))
        if not root.exists():
            continue
        frontier = [(root, 0)]
        while frontier:
            path, depth = frontier.pop(0)
            if _is_data_dir(path):
                log.info("gazeRE data found at %s", path)
                return path
            if depth < max_depth:
                try:
                    frontier += [(c, depth + 1) for c in sorted(path.iterdir())
                                 if c.is_dir() and not c.name.startswith(".") and c.name != "__pycache__"]
                except (PermissionError, OSError):
                    pass
    raise FileNotFoundError(
        "Could not find the gazeRE data. Run gaze_relevance.download.download_gazere() (or `python scripts/download_data.py`), "
        "or clone https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset and set GAZERE_DATA to its path. "
        "Searched: " + ", ".join(tried))


def load_study(data_dir: Path):
    """Load all recordings with the official gazeRE loader. Returns (grel, google_nq) dictionaries
    {participant: {document: {..., 'dataframe': gaze samples}}}."""
    from ._gazere.data_loading import gazeRE_DataLoader
    loader = gazeRE_DataLoader(data_dir=str(data_dir), googleNQ=True, gREL=True)
    return loader.grel, loader.google_nq


# ----------------------------------------------------------------------------- 17 features (official extractor)
def _cached(name, fn, cache=True):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.csv"
    if cache and path.exists():
        return pd.read_csv(path)
    df = fn()
    df.to_csv(path, index=False)
    return df


def _features_from_study(study):
    from ._gazere.features.feature_extractor import LongestVisitFeatureExtractor, extract_training_data
    extractor = LongestVisitFeatureExtractor(min_visit_duration=MIN_VISIT_SECONDS, min_fixations=0,
                                             screen_width=SCREEN_W, screen_height=SCREEN_H)
    logging.getLogger("gaze_relevance._gazere").setLevel(logging.WARNING)
    df = extract_training_data(study_data=study, feature_extractor=extractor).dataframe
    return df


def feature_tables(data_dir: Path | None = None, cache: bool = True):
    """The 17 eye-tracking features (plus f_total_time) of the longest visit > 3 s per paragraph, exactly as in
    the gazeRE dataset paper. Returns (g-REL, GoogleNQ) data frames with columns
    user_id, document, corpus, paragraph, system_label, label, gREL_label (g-REL only), f_*, f_*__pz, y."""
    def build():
        root = data_dir or find_gazere_data()
        grel, nq = load_study(root)
        out = []
        for corpus, study in [("g-rel", grel), ("nq", nq)]:
            d = _features_from_study(study).rename(columns={"user": "user_id", "perceived_relevance": "label",
                                                             "system_relevance": "system_label"})
            d["gREL_label"] = d["system_relevance_type"] if corpus == "g-rel" else np.nan
            d = d.drop(columns=["visit", "method", "system_relevance_type"])
            out.append(d)
        return pd.concat(out, ignore_index=True)

    allf = _cached("features_17", build, cache)
    for c in ["label", "system_label"]:
        allf[c] = allf[c].astype(bool)
    allf["y"] = allf["label"].astype(int)
    # per-reader normalisation (<feature>__pz): per corpus, over ALL of a reader's trials, before any subset is taken
    grel = add_reader_normalisation(allf[allf.corpus == "g-rel"].reset_index(drop=True))
    nq = add_reader_normalisation(allf[allf.corpus == "nq"].drop(columns=["gREL_label"]).reset_index(drop=True))
    assert len(grel) == 288 and len(nq) == 1504, f"unexpected sample counts {len(grel)} / {len(nq)}"
    assert all(c in allf for c in F17)
    return grel, nq


# ----------------------------------------------------------------------------- paragraph visits (all paragraphs)
def paragraph_visits(data_dir: Path | None = None, cache: bool = True) -> pd.DataFrame:
    """One row per (participant, document, paragraph) - including paragraphs that were never visited for > 3 s -
    with the longest visit, total dwell time and number of visits (visits: consecutive gaze samples on the paragraph,
    gaps < 0.2 s merged, as in gazeRE). Used for the analysis of the 3-second visit filter."""
    def build():
        from ._gazere.data_loading import extract_paragraph_visits_vectorized
        root = data_dir or find_gazere_data()
        grel, nq = load_study(root)
        rows = []
        for corpus, study in [("g-rel", grel), ("nq", nq)]:
            for user, docs in study.items():
                for doc, s in docs.items():
                    visits = extract_paragraph_visits_vectorized(s["dataframe"].copy(), doc, min_visit_duration=0.2000001,
                                                                 max_gap_duration=0.2)
                    for p in range(s["num_paragraphs"]):
                        vs = [v for v in visits if v.paragraph_id == p]
                        rows.append(dict(corpus=corpus, user_id=user, document=doc, paragraph=p, doc_index=s["index"],
                                         label=bool(s["perceived_relevance"][p]), system_label=bool(s["system_relevance"][p]),
                                         gREL_label=s.get("g-rel_relevance", np.nan), n_paragraphs=s["num_paragraphs"],
                                         longest_visit=max([v.duration for v in vs], default=0.0),
                                         total_dwell=sum(v.duration for v in vs), n_visits=len(vs)))
        return pd.DataFrame(rows)

    v = _cached("paragraph_visits", build, cache)
    v["y"] = v["label"].astype(int)
    return v
