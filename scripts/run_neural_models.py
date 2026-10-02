"""Neural-model analyses (g-REL; fixed hyperparameters so that all splits are comparable).

  python scripts/run_neural_models.py representations   # render heatmaps, scanpaths, time series (CPU, once)
  python scripts/run_neural_models.py vtnet-unseen      # VTNet: known vs unseen documents       -> Table S7, Fig. 6
  python scripts/run_neural_models.py vtnet-ablation    # full / CNN-only / GRU-only, 3 seeds    -> per-reader ablation tests
  python scripts/run_neural_models.py vgg19             # VGG19: unseen documents + document identification
  python scripts/run_neural_models.py all

Environment variables: GAZERE_DATA (data folder, otherwise searched), GAZERE_RESULTS (output, default results/),
REPRESENTATIONS (default results/representations), DEVICE (default cuda), CUDA_VISIBLE_DEVICES (choose the GPU),
EPOCHS_VTNET (30), EPOCHS_VGG (15), SEEDS (0,1,2 for the ablation), DPI (rendering resolution, default 300 as in
the paper). Outputs are written to results/gpu/ and read by the notebook (section 10).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as BA
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gaze_relevance import representations                      # noqa: E402
from gaze_relevance.analyses import split_schemes                 # noqa: E402
from gaze_relevance.config import RESULTS                         # noqa: E402
from gaze_relevance.evaluation import grel_subsets, participant_folds   # noqa: E402

REP_DIR = Path(os.environ.get("REPRESENTATIONS", RESULTS / "representations"))
OUT = RESULTS / "gpu"
DEVICE = os.environ.get("DEVICE", "cuda")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]


def index(rep):
    """g-REL index with img_path (heatmap or scanpath) and csv_path (time series), plus the columns analyses expect."""
    if not (REP_DIR / "Time_Series_Data" / "gREL.csv").exists():
        build_representations()
    img = pd.read_csv(REP_DIR / f"{rep}_Image_Data" / "gREL.csv")
    ts = pd.read_csv(REP_DIR / "Time_Series_Data" / "gREL.csv")
    d = img.merge(ts[["user_id", "stimulus", "csv_path"]], on=["user_id", "stimulus"])
    d = d.rename(columns={"stimulus": "document"})
    d["y"] = d.label.astype(int)
    return d


def subset(d, sub):
    return grel_subsets(d)[sub].reset_index(drop=True)


def build_representations():
    representations.generate(REP_DIR, corpora=("g-rel",), dpi=int(os.environ.get("DPI", 300)))


def vtnet_unseen():
    from gaze_relevance.neural import VTNetDataset, train_predict_vtnet
    rows = []
    for rep in ["Scanpath", "Heatmaps"]:
        for sub in ["All", "Agree"]:
            d = subset(index(rep), sub)
            ds = VTNetDataset(d)
            for scheme, sp in split_schemes(d).items():
                prob = np.full(len(d), np.nan)
                for tr, te in sp:
                    if len(te) == 0:
                        continue
                    prob[te] = train_predict_vtnet(ds, tr, te, seed=0, epochs=int(os.environ.get("EPOCHS_VTNET", 30)), device=DEVICE) \
                        if d.y.iloc[tr].nunique() == 2 else float(d.y.iloc[tr].mode()[0])
                m = ~np.isnan(prob)
                r = dict(model="VTNet", representation=rep, subset=sub, split=scheme, n=int(m.sum()),
                         BA=float(BA(d.y[m], prob[m] >= 0.5)), AUC=float(roc_auc_score(d.y[m], prob[m])))
                rows.append(r)
                print(r, flush=True)
                pd.DataFrame(rows).to_csv(OUT / "vtnet_unseen_documents.csv", index=False)


def vtnet_ablation():
    """Per-trial probabilities of the full VTNet and of each branch on the participant folds (scanpaths, 3 seeds)."""
    from gaze_relevance.neural import VTNetDataset, train_predict_vtnet
    rows = []
    for sub in ["All", "Agree"]:
        d = subset(index("Scanpath"), sub)
        ds = VTNetDataset(d)
        for branches in ["full", "cnn_only", "gru_only"]:
            for seed in SEEDS:
                prob = np.zeros(len(d))
                for tr, te in participant_folds(d):
                    prob[te] = train_predict_vtnet(ds, tr, te, branches, seed, epochs=int(os.environ.get("EPOCHS_VTNET", 30)), device=DEVICE)
                rows += [dict(subset=sub, model=branches, seed=seed, user_id=u, document=s, y=int(t), prob=float(p))
                         for u, s, t, p in zip(d.user_id, d.document, d.y, prob)]
                print(dict(subset=sub, model=branches, seed=seed, BA=round(BA(d.y, prob >= 0.5), 3)), flush=True)
                pd.DataFrame(rows).to_csv(OUT / "vtnet_ablation_predictions.csv", index=False)


def vgg19():
    from gaze_relevance.neural import load_images_rgb, train_predict_vgg19
    rows = []
    ep = int(os.environ.get("EPOCHS_VGG", 15))
    for rep in ["Scanpath", "Heatmaps"]:
        d = index(rep).reset_index(drop=True)
        X = load_images_rgb(d.img_path)
        docs = sorted(d.document.unique())
        dix = d.document.map({s: i for i, s in enumerate(docs)}).values
        p = np.zeros(len(d), int)
        for tr, te in participant_folds(d):
            p[te] = train_predict_vgg19(X, dix, tr, te, len(docs), epochs=ep, device=DEVICE)
        rows.append(dict(model="VGG19", representation=rep, subset="All", split="participant", task="identify document",
                         accuracy=float((p == dix).mean()), chance=1 / len(docs)))
        print(rows[-1], flush=True)
        for sub in ["All", "Agree"]:
            mask = d.index.isin(grel_subsets(d)[sub].index)
            dd, Xs = d[mask].reset_index(drop=True), X[np.where(mask)[0]]
            y = dd.y.values
            for scheme, sp in split_schemes(dd).items():
                P = np.full(len(dd), -1)
                for tr, te in sp:
                    if len(te):
                        P[te] = train_predict_vgg19(Xs, y, tr, te, 2, epochs=ep, device=DEVICE) if len(set(y[tr])) == 2 else np.bincount(y[tr]).argmax()
                m = P >= 0
                rows.append(dict(model="VGG19", representation=rep, subset=sub, split=scheme, task="relevance", BA=float(BA(y[m], P[m])), n=int(m.sum())))
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(OUT / "vgg19_unseen_documents.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {"representations": build_representations, "vtnet-unseen": vtnet_unseen, "vtnet-ablation": vtnet_ablation, "vgg19": vgg19}
    for name in (jobs if task == "all" else [task]):
        print(f"== {name}", flush=True)
        jobs[name]()
    (OUT / "run_info.json").write_text(json.dumps(dict(task=task, device=DEVICE, seeds=SEEDS), indent=1))
