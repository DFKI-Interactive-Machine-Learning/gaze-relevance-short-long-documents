# Data folder

This folder is empty in the repository. The data are the public **gazeRE** recordings
(<https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset>, GPL-3.0), which are not redistributed here.

## Option 1: automatic (recommended)

Nothing to do. The first code cells of the notebook download the dataset into this folder. You can also download
it yourself:

```bash
python scripts/download_data.py
```

This downloads the archive of commit `9af27c6` (about 100 MB, 265 MB unpacked), the exact version used for all
results, and unpacks it to `data/gazeRE-dataset-9af27c6.../`. No git is needed.

## Option 2: use your own copy

```bash
git clone https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset
cd gazeRE-dataset && git checkout 9af27c6     # optional: the version used for the published results
```

Clone it into this folder or next to the repository, or anywhere else and point the code to it:

```bash
export GAZERE_DATA=/path/to/gazeRE-dataset          # Windows (PowerShell): $env:GAZERE_DATA="C:\path\to\gazeRE-dataset"
```

The code searches `$GAZERE_DATA`, this folder and `../gazeRE-dataset` **recursively** for the folder that contains
the participant folders, so any parent folder works.

## Expected structure

```
gazeRE-dataset/data/
├── A01/
│   ├── g-rel/      0_g-rel_q116-1_r.csv, 1_g-rel_q076-1_r.csv, ..., User_Rating
│   └── GoogleNQ/   0_nq_7p_a1_Mzgy.csv, ..., User_Rating
├── A03/
└── ...             24 participants: A01, A03, ..., A13, B01, ..., B13
```

Each CSV holds the gaze samples of one document (`|`-separated: timestamp, gaze_x, gaze_y, gaze_y_abs,
fixation_id, scroll_y, paragraph_id). `User_Rating` holds the participant's relevance ratings, one line per
document in reading order. Section 2 of the notebook explains the files in detail.

When you use the data, please cite: M. Barz, O. S. Bhatti, D. Sonntag. *Implicit Estimation of Paragraph Relevance
from Eye Movements.* Frontiers in Computer Science, 2021. <https://doi.org/10.3389/fcomp.2021.808507>
