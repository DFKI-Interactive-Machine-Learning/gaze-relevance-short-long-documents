"""Folds, subsets, per-reader normalisation, metrics and statistical tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from statsmodels.stats.multitest import multipletests

from .config import F17, FOLD_SEED, N_FOLDS


# ----------------------------------------------------------------------------- subsets
def grel_subsets(g):
    """g-REL subsets as in the paper: Agree = perceived label matches the system label and the document is not
    topical; Topical = on-topic documents without the answer."""
    return {"All": g, "Agree": g[(g.label == g.system_label) & (g.gREL_label != "t")], "Topical": g[g.gREL_label == "t"]}


def nq_subsets(n):
    return {"All": n, "Agree": n[n.label == n.system_label]}


# ----------------------------------------------------------------------------- folds
def participant_folds(df, k: int = N_FOLDS):
    """Participant-grouped, label-stratified folds: StratifiedGroupKFold(5, shuffle=True, random_state=256).
    NOTE: the participant assignment of StratifiedGroupKFold depends on the scikit-learn version; this repository uses
    scikit-learn 1.9.1 (see requirements.txt). The assignments are written to results/ (Supplementary Table S2)."""
    df = df.reset_index(drop=True)
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=FOLD_SEED).split(df, df["y"], df["user_id"]))


def fold_table(df, name):
    """Which participants are held out in each fold (Supplementary Table S2)."""
    df = df.reset_index(drop=True)
    return pd.DataFrame([dict(subset=name, fold=i + 1, participants=", ".join(sorted(df.user_id.iloc[te].unique())),
                              n_samples=len(te), n_relevant=int(df.y.iloc[te].sum()))
                         for i, (_, te) in enumerate(participant_folds(df))])


# ----------------------------------------------------------------------------- per-reader normalisation
def add_reader_normalisation(df, cols=F17):
    """Adds <feature>__pz: each feature z-scored against the SAME reader's trials in the SAME corpus. Computed over all
    of a reader's trials (before any label-defined subset is taken) and without labels."""
    df = df.copy()
    grp = df.groupby("user_id")[cols]
    z = (df[cols] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    for c in cols:
        df[c + "__pz"] = z[c].fillna(0.0)
    return df


def reader_normalised(df, cols):
    """Copy of df whose `cols` are replaced by their per-reader normalised versions."""
    out = df.copy()
    out[cols] = df[[c + "__pz" for c in cols]].values
    return out


# ----------------------------------------------------------------------------- metrics
def metrics(y, p):
    """Balanced accuracy (primary), F1, recall, precision with 'perceived relevant' as the positive class, computed
    from one pooled confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(BA=balanced_accuracy_score(y, p), F1=f1_score(y, p, zero_division=0), recall=recall_score(y, p, zero_division=0),
                precision=precision_score(y, p, zero_division=0), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp), n=int(len(y)))


def bootstrap_ci(y, p, users, B: int = 2000, seed: int = 0):
    """95% CI of balanced accuracy, resampling participants (2000 resamples)."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "u": np.asarray(users)})
    groups = [x for _, x in d.groupby("u")]
    vals = []
    for _ in range(B):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if s.y.nunique() == 2:
            vals.append(balanced_accuracy_score(s.y, s.p))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def permutation_p(y, p, users, B: int = 500, seed: int = 0):
    """Permutation test: labels shuffled within each participant (500 permutations). The smallest possible value is
    1/501 = 0.002."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "u": np.asarray(users)})
    obs = balanced_accuracy_score(d.y, d.p)
    hits = sum(balanced_accuracy_score(d.groupby("u").y.transform(lambda s: rng.permutation(s.values)), d.p) >= obs for _ in range(B))
    return (hits + 1) / (B + 1)


def full_report(y, p, users, perm: bool = True):
    r = metrics(y, p)
    r["CI_low"], r["CI_high"] = bootstrap_ci(y, p, users)
    if perm:
        r["perm_p"] = permutation_p(y, p, users)
    return r


def per_reader_ba(y, p, users):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "u": np.asarray(users)})
    return d.groupby("u").apply(lambda x: balanced_accuracy_score(x.y, x.p) if x.y.nunique() == 2 else np.nan)


def wilcoxon_readers(y, p_a, p_b, users):
    """Paired comparison of two prediction sets with the readers as units (Wilcoxon signed-rank test)."""
    a, b = per_reader_ba(y, p_a, users), per_reader_ba(y, p_b, users)
    ok = a.notna() & b.notna()
    d = a[ok] - b[ok]
    p = wilcoxon(a[ok], b[ok]).pvalue if (d != 0).any() else 1.0
    return dict(mean_diff=d.mean(), a_better=int((d > 0).sum()), b_better=int((d < 0).sum()), equal=int((d == 0).sum()),
                readers=int(ok.sum()), p=p)


# ----------------------------------------------------------------------------- fold-level tests
def corrected_ttest(a, b, k: int = N_FOLDS, test_train_ratio: float = 1 / 4):
    """Nadeau & Bengio (2003) corrected resampled t-test for k-fold CV (n_test/n_train = 1/4 for 5 folds)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    m, s = d.mean(), d.std(ddof=1)
    se = np.sqrt((1 / k + test_train_ratio) * s ** 2)
    t = m / se
    half = stats.t.ppf(0.975, k - 1) * se
    naive_t = m / (s / np.sqrt(k))
    return dict(diff=m, CI_low=m - half, CI_high=m + half, t_corrected=t, p_corrected=2 * stats.t.sf(abs(t), k - 1),
                t_naive=naive_t, p_naive=2 * stats.t.sf(abs(naive_t), k - 1), folds_a_better=int((d > 0).sum()))


def holm(pvalues):
    return multipletests(np.asarray(pvalues, float), method="holm")[1]
