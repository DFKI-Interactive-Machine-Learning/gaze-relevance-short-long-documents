"""Image and time-series representations of the gaze data, ported from the paper's data-generation notebooks
(Heatmap_Data_Generation, Scanpath_Data_Generation, Timeseries_Data_Generation). These build on the CNN stimulus
generation of the GNN study, https://github.com/DFKI-Interactive-Machine-Learning/GNN-Scanpath-Analysis-ICMI2024
(DataGeneration/CNN_Data_Generation.ipynb, GPL-3.0; Mohamed Selim et al., ICMI '24, doi:10.1145/3678957.3685736).

  heatmap      2D histogram of the gaze samples on the 2560x1440 screen, Gaussian smoothing (sigma = 25 px),
               viridis colour map, black background; for GoogleNQ paragraphs the y coordinates are shifted so that
               every paragraph starts at the top of the canvas (paragraph location is not visible).
  scanpath     fixations (mean gaze position per fixation_id) drawn by duration level - 110-250 ms red circle,
               250-400 ms purple star, 400-550 ms yellow pentagon, >= 550 ms white cross - and saccades as straight
               lines coloured with the 'winter' colour map in temporal order, black background.
  time series  x and y gaze coordinates of every sample, min-max scaled within the trial (VTNet's GRU input).

Each generator writes the files to `out_dir/<representation>/` and an index CSV (`gREL.csv` / `GoogleNQ.csv`) with
the columns user_id, corpus, stimulus, [paragraph_id], label, system_label, [gREL_label], img_path | csv_path.
The original images were rendered at dpi=300 (7680x4320 px); the networks resize them to 150x150 (VTNet) or
256x256 (VGG19), so a lower `dpi` gives practically the same network input and is much faster.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import SCREEN_H, SCREEN_W

FIXATION_LEVELS = [((110, 250), "o", "red", 6), ((250, 400), "*", "purple", 12),
                   ((400, 550), "p", "yellow", 18), ((550, np.inf), "x", "white", 24)]
NQ_PADDING = 100


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def heatmap(df, path, nq=False, dpi=300):
    from scipy.ndimage import gaussian_filter
    plt = _plt()
    x, y = df["gaze_x"], df["gaze_y"]
    if nq:
        y = SCREEN_H - (y.max() - y) - NQ_PADDING
    h, _, _ = np.histogram2d(x, y, bins=(SCREEN_W, SCREEN_H), range=[[0, SCREEN_W], [0, SCREEN_H]])
    h = gaussian_filter(h, sigma=25)
    plt.figure(figsize=(25.6, 14.4))
    plt.axis("off")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    plt.imshow(h.T, origin="lower", cmap=cmap, interpolation="bilinear")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(path, bbox_inches="tight", pad_inches=0, facecolor="black", dpi=dpi)
    plt.close("all")


def fixations(df):
    """One row per fixation: start time and duration in ms, mean gaze position."""
    ev = df.groupby("fixation_id", as_index=False).agg(timestamp=("timestamp", "min"), max_timestamp=("timestamp", "max"),
                                                       avg_gaze_x=("gaze_x", "mean"), avg_gaze_y=("gaze_y", "mean"))
    ev["duration"] = (ev.max_timestamp - ev.timestamp) * 1000
    ev["timestamp"] *= 1000
    return ev.drop(columns=["max_timestamp"])


def scanpath(ev, path, nq=False, dpi=300):
    plt = _plt()
    ev = ev.copy()
    plt.figure(figsize=(25.6, 14.4), dpi=100, facecolor="black")
    plt.axis("off")
    if nq:
        ev["avg_gaze_y"] = SCREEN_H - (ev.avg_gaze_y.max() - ev.avg_gaze_y) - NQ_PADDING
    n = len(ev)
    for i in range(n - 1):
        # g-REL: every consecutive pair; GoogleNQ: only consecutive fixation ids (the paragraph may be left and re-entered)
        if not nq or ev.fixation_id.iloc[i + 1] - ev.fixation_id.iloc[i] == 1:
            plt.plot(ev.avg_gaze_x.iloc[i:i + 2], ev.avg_gaze_y.iloc[i:i + 2], color=plt.cm.winter(i / n))
    for _, f in ev.iterrows():
        for (lo, hi), marker, color, size in FIXATION_LEVELS:
            if lo <= f.duration < hi:
                plt.plot(f.avg_gaze_x, f.avg_gaze_y, marker=marker, color=color, markersize=size)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Scanpath")
    plt.xlim(0, SCREEN_W)
    plt.ylim(0, SCREEN_H)
    plt.savefig(path, facecolor="black", bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close("all")


def time_series(df):
    t = df[["timestamp", "gaze_x", "gaze_y"]].rename(columns={"timestamp": "t", "gaze_x": "x", "gaze_y": "y"})
    for c in ["x", "y"]:
        t[c] = (t[c] - t[c].min()) / (t[c].max() - t[c].min())
    return t


def _trials(study, nq):
    """Yields (meta, gaze data frame) per g-REL trial or per GoogleNQ paragraph."""
    for user, docs in study.items():
        for doc, s in docs.items():
            if not nq:
                yield dict(user_id=user, corpus="g-rel", stimulus=doc, label=bool(s["perceived_relevance"][0]),
                           system_label=bool(s["system_relevance"][0]), gREL_label=s["g-rel_relevance"][0]), s["dataframe"]
            else:
                df = s["dataframe"]
                for p in range(s["num_paragraphs"]):
                    yield dict(user_id=user, corpus="nq", stimulus=doc, paragraph_id=p, label=bool(s["perceived_relevance"][p]),
                               system_label=bool(s["system_relevance"][p])), df[df.paragraph_id == p]


def generate(out_dir, data_dir=None, corpora=("g-rel",), kinds=("heatmap", "scanpath", "time_series"), dpi=300, overwrite=False):
    """Render the representations for the requested corpora. Returns {kind: {corpus: index data frame}}."""
    from .data import find_gazere_data, load_study
    out_dir = Path(out_dir)
    grel, nq = load_study(data_dir or find_gazere_data())
    folders = {"heatmap": "Heatmaps_Image_Data", "scanpath": "Scanpath_Image_Data", "time_series": "Time_Series_Data"}
    result = {}
    for kind in kinds:
        folder = out_dir / folders[kind]
        (folder / "event_data").mkdir(parents=True, exist_ok=True)
        result[kind] = {}
        for corpus, study in [("g-rel", grel), ("nq", nq)]:
            if corpus not in corpora:
                continue
            rows, is_nq = [], corpus == "nq"
            for meta, df in _trials(study, is_nq):
                stem = f"{meta['user_id']}_{meta['stimulus']}" + (f"_{meta['paragraph_id']}" if is_nq else "") + f"_{meta['label']}_{meta['system_label']}"
                if kind == "time_series":
                    path = folder / "event_data" / f"{stem}.csv"
                    if overwrite or not path.exists():
                        time_series(df).to_csv(path, index=False)
                    rows.append({**meta, "csv_path": str(path)})
                    continue
                path = folder / "event_data" / f"{stem}.png"
                if kind == "scanpath":
                    ev = fixations(df)
                    if ev.empty:                      # paragraphs without fixations have no scanpath (as in the paper)
                        continue
                    if overwrite or not path.exists():
                        scanpath(ev, path, nq=is_nq, dpi=dpi)
                elif overwrite or not path.exists():
                    heatmap(df, path, nq=is_nq, dpi=dpi)
                rows.append({**meta, "img_path": str(path)})
            idx = pd.DataFrame(rows)
            idx.to_csv(folder / ("gREL.csv" if corpus == "g-rel" else "GoogleNQ.csv"), index=False)
            result[kind][corpus] = idx
    return result
