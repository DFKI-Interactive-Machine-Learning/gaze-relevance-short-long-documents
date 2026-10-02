# Perceived text relevance from eye movements: short vs long documents

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23103752.svg)](https://doi.org/10.5281/zenodo.23103752)

Can we tell from someone's eye movements whether they find a text relevant, on short documents that fit on the
screen and on long documents that require scrolling? This repository contains the complete, documented analysis
behind the article

> **A Comparative Analysis of Gaze-Based Representations for Perceived Text Relevance Estimation Across Short and
> Long Documents.** Abdulrahman Mohamed Selim, Omair Shahzad Bhatti, Cristina Conati, Michael Barz, Daniel Sonntag.
> *Scientific Reports.*

Everything starts from the **public gazeRE eye-tracking dataset**: 24 readers, 12 short g-REL documents and 12 long
GoogleNQ documents. A single Jupyter notebook downloads the data and explains and runs every step, from the raw gaze
samples to the tables and figures of the paper.

## Run it in three steps

```bash
# 1. get the code and install the requirements (Python >= 3.10; tested with 3.13)
git clone https://github.com/DFKI-Interactive-Machine-Learning/gaze-relevance-short-long-documents
cd gaze-relevance-short-long-documents
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (optional) download the dataset beforehand - otherwise the notebook does it
python scripts/download_data.py

# 3. open the notebook and run all cells
jupyter lab notebooks/gaze_relevance_analysis.ipynb
```

To run it without opening Jupyter:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/gaze_relevance_analysis.ipynb
```

The full run takes about 5 minutes on a laptop CPU. The first run also downloads the data (about 100 MB) and
extracts the features (about 30 s). All tables (CSV) and figures (PDF) are written to `results/`.

## Sources this work builds on

This repository contains **no data** and re-uploads nothing. It builds on two public repositories, both under
GPL-3.0:

| Source | What we use | Reference |
|---|---|---|
| [gazeRE dataset](https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset) | the eye-tracking recordings and relevance ratings (downloaded, never copied), the official data loader and the feature extractor (included in `src/gaze_relevance/_gazere/`) | Barz, Bhatti & Sonntag, *Frontiers in Computer Science*, 2021 |
| [GNN-Scanpath-Analysis-ICMI2024](https://github.com/DFKI-Interactive-Machine-Learning/GNN-Scanpath-Analysis-ICMI2024) | the generation of the image representations (heatmaps, scanpaths; `DataGeneration/CNN_Data_Generation.ipynb`), adapted in `src/gaze_relevance/representations.py`, and the nested cross-validation protocol of the neural models | Mohamed Selim, Bhatti, Barz & Sonntag, *ICMI '24* |

## The dataset

The gazeRE dataset (Barz, Bhatti & Sonntag, 2021) is hosted at
<https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset> under GPL-3.0. It is **not** redistributed
here. There are three ways to provide it:

| | How | When |
|---|---|---|
| **automatic** | do nothing; the notebook calls `download_gazere()` | default |
| **script** | `python scripts/download_data.py [--dest FOLDER]` | to download once, e.g. on a server |
| **own copy** | `git clone` the dataset, then `export GAZERE_DATA=/path/to/it` (or put it in `data/` or next to this repository) | offline machines, existing copies |

The automatic download fetches the archive of commit `9af27c6`, the exact version used for the published
results. The code searches the possible locations **recursively** for the folder that contains the participant
folders `A01/ … B13/`. See [`data/README.md`](data/README.md) for the expected file structure.

## What the notebook covers

| § | Step | Paper |
|---|---|---|
| 2 | Download the dataset, inspect the raw files, plot a short and a long reading trial | Methods: Dataset |
| 3 | Visits, the longest-visit rule, the 17 eye-tracking features, the All / Agree / Topical subsets | Table 1 |
| 4 | Participant folds, per-reader normalisation, metrics, bootstrap and permutation tests, protocol checks | Table S2 |
| 5 | Feature-based models with the algorithm selected inside a nested cross-validation | Tables S3, S4 |
| 6 | How the 3-second visit filter affects relevant and irrelevant paragraphs | Table S1 |
| 7 | Transfer from short to long documents with held-out readers; comparison with models trained on long documents; paragraph position | Tables S5, S6, S8 |
| 8 | Which features carry the transfer: permutation importance, SHAP, distributions, PCA | Figures 7, 8, 9 |
| 9 | Generalisation to documents not seen during training | Figure 6, Table S7 |
| 10 | Neural models (VTNet, VGG19) and statistical tests, from optional inputs | Results, Discussion |
| 11 | Summary of the key numbers | |

Each step has a short explanation of what is computed, why, and how to read the result.

## Repository layout

```
├── notebooks/
│   └── gaze_relevance_analysis.ipynb   the documented analysis (start here)
├── src/gaze_relevance/
│   ├── download.py         download of the gazeRE dataset (pinned commit)
│   ├── data.py             recursive data discovery, feature extraction, paragraph visits (cached in .cache/)
│   ├── evaluation.py       subsets, participant folds, per-reader normalisation, metrics and statistical tests
│   ├── models.py           classical algorithm pool and transfer models
│   ├── analyses.py         one function per analysis of the paper
│   ├── figures.py          paper figures and explanatory plots
│   ├── representations.py  heatmaps, scanpaths and time series for the neural models
│   ├── neural.py           VTNet (full / CNN branch / GRU branch) and VGG19
│   ├── config.py           constants (screen size, 3-s threshold, fold seed, feature names)
│   └── _gazere/            official gazeRE loader and feature extractor (unchanged apart from small fixes)
├── scripts/
│   ├── download_data.py    download the dataset from the command line
│   └── run_neural_models.py  GPU analyses of VTNet and VGG19
├── data/                   the dataset is downloaded here (not under version control)
└── results/                generated tables and figures (not under version control)
```

## Methods in brief

- **Features.** These are the 17 features of the gazeRE paper (fixations, saccades, bounding box and convex hull
  of the gaze). They are computed with the official gazeRE code on the longest visit (> 3 s) to each g-REL
  document and GoogleNQ paragraph.
- **Readers are always held out.** Outer folds use `StratifiedGroupKFold(5, shuffle=True, random_state=256)`,
  grouped by participant; model and algorithm choices happen in inner loops on the training readers. The
  participant assignment of this splitter depends on the scikit-learn version, so `requirements.txt` pins
  **1.9.1**.
- **Per-reader normalisation.** Each feature is z-scored against the same reader's texts in the same corpus. This
  uses no labels and is computed before any subset is taken.
- **Transfer.** Train on the g-REL trials of some readers, test on the GoogleNQ paragraphs of the other readers.
  The configuration is selected on g-REL only.
- **Metrics.** Balanced accuracy (primary), F1, recall and precision (relevant = positive class) from pooled
  predictions, with 95% participant-bootstrap confidence intervals, within-participant permutation tests,
  per-reader Wilcoxon tests and corrected resampled t-tests (Nadeau & Bengio) with Holm adjustment.

## Neural models (optional, GPU)

```bash
pip install -r requirements-gpu.txt
CUDA_VISIBLE_DEVICES=0 python scripts/run_neural_models.py all
```

This renders the g-REL heatmaps, scanpaths and time series from the raw data, with the rendering code of the paper, which is adapted from the GNN-Scanpath-Analysis-ICMI2024 repository. It then trains VTNet and VGG19 with fixed hyperparameters on known and on unseen documents, and runs the
VTNet branch ablation with per-trial predictions. Expect a few GPU hours; single steps are `representations`,
`vtnet-unseen`, `vtnet-ablation` and `vgg19`. Rendering at the paper's resolution (`DPI=300`) is slow, and a lower
`DPI` gives nearly the same 150×150 / 256×256 network input. The notebook picks up the outputs from
`results/gpu/`.

Section 10.2 of the notebook can also compare the fold-level results of the original neural-model training runs.
To use it, set `ORIGINAL_RESULTS` to a folder with `<model>/gREL_<subset>.csv` files.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `GAZERE_DATA` | searched automatically | location of the gazeRE dataset (any parent folder works) |
| `GAZERE_CACHE` | `.cache/` | cached feature and visit tables |
| `GAZERE_RESULTS` | `results/` | output folder |
| `ORIGINAL_RESULTS` | – | fold-level results of the original neural models (Section 10.2) |
| `ABLATION_PREDICTIONS` | `results/gpu/vtnet_ablation_predictions.csv` | per-trial VTNet predictions (Section 10.1) |

## Tested with

Python 3.13.4, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, statsmodels 0.15.0, matplotlib 3.11.2,
shap 0.52.0; for the neural models, torch 2.14.1 and torchvision 0.29.1. The notebook records the versions of
each run in `results/environment.json`.

## Ethics

This work is a secondary analysis of the publicly available gazeRE dataset. All participants of the original study
gave written informed consent (see Barz et al., 2021).

## Citation

Please cite the article, this code, the dataset and the GNN study whose code we build on.

The code is archived on Zenodo:
- version 1.0.0, the code used for the article: [10.5281/zenodo.23103752](https://doi.org/10.5281/zenodo.23103752);
- all versions, always resolving to the latest: [10.5281/zenodo.23103751](https://doi.org/10.5281/zenodo.23103751).

```bibtex
@software{mohamed_selim_gaze_relevance_2026,
  title     = {gaze-relevance: perceived text relevance from eye movements on short and long documents},
  author    = {Mohamed Selim, Abdulrahman and Bhatti, Omair Shahzad and Conati, Cristina and Barz, Michael and Sonntag, Daniel},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23103752},
  url       = {https://github.com/DFKI-Interactive-Machine-Learning/gaze-relevance-short-long-documents}
}
```

```bibtex
@article{barz_implicit_2021,
  title   = {Implicit Estimation of Paragraph Relevance from Eye Movements},
  author  = {Barz, Michael and Bhatti, Omair Shahzad and Sonntag, Daniel},
  journal = {Frontiers in Computer Science},
  year    = {2021},
  doi     = {10.3389/fcomp.2021.808507}
}

@inproceedings{mohamed_selim_perceived_2024,
  title     = {Perceived Text Relevance Estimation Using Scanpaths and GNNs},
  author    = {Mohamed Selim, Abdulrahman and Bhatti, Omair Shahzad and Barz, Michael and Sonntag, Daniel},
  booktitle = {Proceedings of the 26th International Conference on Multimodal Interaction (ICMI '24)},
  pages     = {418--427},
  year      = {2024},
  publisher = {ACM},
  doi       = {10.1145/3678957.3685736}
}
```

## Licence

GPL-3.0 (see `LICENSE`), the same licence as the two sources above: the gazeRE dataset and code (included in
`src/gaze_relevance/_gazere/`) and the GNN-Scanpath-Analysis-ICMI2024 code (adapted in
`src/gaze_relevance/representations.py`). The gazeRE data themselves are not part of this repository and remain
under the terms of the gazeRE repository.
