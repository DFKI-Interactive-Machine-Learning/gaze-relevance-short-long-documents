"""The analyses of the paper. Every function returns plain pandas objects; the notebook prints and saves them.

Function                     Paper
---------------------------  -----------------------------------------------------------------------------
fold_assignments             Supplementary Table S2 (participant folds)
nested_selection             Supplementary Tables S3/S4, feature-based models (nested algorithm selection)
visit_filter                 Supplementary Table S1 (3-second visit filter), dwell-time baseline on all paragraphs
transfer_grid                Supplementary Table S6 (all transfer configurations)
selected_transfer            Supplementary Table S5 (transfer g-REL -> GoogleNQ, held-out readers)
within_googlenq,
transfer_vs_within           transfer vs models trained on GoogleNQ (per-reader Wilcoxon test)
transfer_by_position         Supplementary Table S8 (transfer by paragraph position)
transfer_importance          Figure 7 (permutation importance, SHAP)
pca_separability             Figure 9 (PCA of both corpora)
unseen_documents,
document_identification      Figure 6 / Supplementary Table S7 (feature-based models on unseen documents)
fold_level_tests             corrected resampled t-tests with Holm adjustment (fold-level comparisons)
reader_level_tests           per-reader Wilcoxon tests (VTNet ablation, VTNet vs LDA)
leakage_checks               protocol checks
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as BA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import F17, HULL2
from .evaluation import (corrected_ttest, full_report, grel_subsets, holm, metrics, nq_subsets, participant_folds,
                         per_reader_ba, reader_normalised, wilcoxon_readers)
from .models import CLASSICAL_POOL, TRANSFER_MODELS, fit_predict_classical, minmax_pipeline, within_googlenq_svm

FEATURE_SETS = {"17 features": F17, "2 convex hull features": HULL2}
NORMS = ["Min-Max", "Min-Max + per-reader"]


# ============================================================================ 3-second visit filter (Table S1)
def visit_filter(visits: pd.DataFrame, thresholds=(0, 0.5, 1, 2, 3, 4, 5)):
    """Exclusion rate per class for several minimum-visit thresholds (Supplementary Table S1) and a dwell-time-only
    baseline on all paragraphs vs on paragraphs with a visit > 3 s (post-hoc analysis)."""
    d = visits[visits.corpus == "nq"]
    subs = {"All": d, "Agree": d[d.label == d.system_label]}
    rows, counts, dwell = [], [], []
    for name, x in subs.items():
        R, I = int(x.y.sum()), int((1 - x.y).sum())
        for t in thresholds:
            k = x[x.longest_visit > t]
            er, ei = 100 * (R - k.y.sum()) / R, 100 * (I - (1 - k.y).sum()) / I
            rows.append(dict(subset=name, threshold_s=t, kept=len(k), kept_relevant=int(k.y.sum()), kept_irrelevant=int((1 - k.y).sum()),
                             excluded_relevant_pct=er, excluded_irrelevant_pct=ei, ratio=ei / er if er > 0 else np.nan))
        k = x[x.longest_visit > 3]
        counts.append(dict(subset=name, paragraphs=len(x), relevant=R, irrelevant=I, kept=len(k),
                           removed_relevant=R - int(k.y.sum()), removed_irrelevant=I - int((1 - k.y).sum())))
        dwell.append(dict(subset=name, all_paragraphs_BA=dwell_baseline(x), visits_over_3s_BA=dwell_baseline(k),
                          n_all=len(x), n_over_3s=len(k)))
    return pd.DataFrame(counts), pd.DataFrame(rows), pd.DataFrame(dwell)


def dwell_baseline(d):
    """Dwell time only: log total dwell, log longest visit, number of visits; class-weighted logistic regression on
    the participant folds."""
    d = d.reset_index(drop=True)
    X = np.c_[np.log1p(d.total_dwell), np.log1p(d.longest_visit), d.n_visits]
    p = np.zeros(len(d), int)
    for tr, te in participant_folds(d):
        p[te] = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(X[tr], d.y[tr]).predict(X[te])
    return BA(d.y, p)


# ============================================================================ participant folds and nested selection (Tables S2-S4)
def fold_assignments(grel, nq):
    """Held-out participants of each outer fold for every subset (Supplementary Table S2)."""
    from .evaluation import fold_table
    parts = [fold_table(d, f"g-REL {k}") for k, d in grel_subsets(grel).items()]
    parts += [fold_table(d, f"GoogleNQ {k}") for k, d in nq_subsets(nq).items()]
    return pd.concat(parts, ignore_index=True)


def nested_selection(grel, nq, features=F17):
    """The twelve classical algorithms are compared in an inner participant-grouped 5-fold CV on the training part of
    each outer fold; the winner is refitted on the training part and tested on the outer fold."""
    rows, folds = [], []
    for corpus, subs in [("g-REL", grel_subsets(grel)), ("GoogleNQ", nq_subsets(nq))]:
        for sub, d in subs.items():
            d = d.reset_index(drop=True)
            X, y = d[features].values, d.y.values
            outer, chosen = [], []
            for i, (tr, te) in enumerate(participant_folds(d)):
                inner_d = d.iloc[tr].reset_index(drop=True)
                Xi, yi = inner_d[features].values, inner_d.y.values
                inner = {m: np.mean([BA(yi[b], fit_predict_classical(m, Xi[a], yi[a], Xi[b]))
                                     for a, b in participant_folds(inner_d) if len(set(yi[a])) == 2]) for m in CLASSICAL_POOL}
                best = max(inner, key=inner.get)
                p = fit_predict_classical(best, X[tr], y[tr], X[te])
                outer.append(BA(y[te], p))
                chosen.append(best)
                folds.append(dict(corpus=corpus, subset=sub, fold=i + 1, selected=best, inner_BA=inner[best], **metrics(y[te], p)))
            rows.append(dict(corpus=corpus, subset=sub, BA_mean=np.mean(outer), BA_SD=np.std(outer, ddof=1),
                             selected_per_fold=", ".join(chosen)))
    return pd.DataFrame(rows), pd.DataFrame(folds)


# ============================================================================ transfer from g-REL to GoogleNQ (Tables S5, S6)
def grel_sources(grel):
    return {"g-REL All": grel, "g-REL Agree": grel_subsets(grel)["Agree"], "g-REL without topical": grel[grel.gREL_label != "t"]}


def _prep(df, cols, norm):
    return reader_normalised(df, cols) if norm == "Min-Max + per-reader" else df


def _fit(src, cols, norm, model):
    s = _prep(src, cols, norm)
    return minmax_pipeline(TRANSFER_MODELS[model]).fit(s[cols], s.y)


def transfer_predict(src, target, cols, norm, model, held_out_readers=True):
    """Train on g-REL, predict GoogleNQ. With held_out_readers, for each GoogleNQ participant fold the model is trained
    on the g-REL trials of the OTHER readers only (participant-disjoint transfer, the main protocol)."""
    T = target.reset_index(drop=True)
    if not held_out_readers:
        return _fit(src, cols, norm, model).predict(_prep(T, cols, norm)[cols])
    p = np.zeros(len(T), int)
    for _, te in participant_folds(T):
        held = T.user_id.iloc[te].unique()
        p[te] = _fit(src[~src.user_id.isin(held)], cols, norm, model).predict(_prep(T.iloc[te], cols, norm)[cols])
    return p


def grel_cv(src, cols, norm, model):
    """Selection criterion: participant-grouped 5-fold CV balanced accuracy on g-REL only."""
    s = src.reset_index(drop=True)
    p = np.zeros(len(s), int)
    for tr, te in participant_folds(s):
        p[te] = _fit(s.iloc[tr], cols, norm, model).predict(_prep(s.iloc[te], cols, norm)[cols])
    return BA(s.y, p)


def transfer_grid(grel, nq):
    """All 3 x 2 x 2 x 5 = 60 transfer configurations (Supplementary Table S6): g-REL CV score (used for selection)
    and participant-disjoint transfer BA to GoogleNQ All and Agree (reported, never used for selection)."""
    targets = {k: v.reset_index(drop=True) for k, v in nq_subsets(nq).items()}
    rows = []
    for (sn, S), (fn, cols), norm, model in itertools.product(grel_sources(grel).items(), FEATURE_SETS.items(), NORMS, TRANSFER_MODELS):
        r = dict(source=sn, features=fn, normalisation=norm, model=model, grel_cv_BA=grel_cv(S, cols, norm, model))
        for tn, T in targets.items():
            r[f"transfer_BA_{tn}"] = BA(T.y, transfer_predict(S, T, cols, norm, model))
        rows.append(r)
    return pd.DataFrame(rows)


def evaluate_transfer(grel, nq, source, features, norm, model, label, held_out_readers=True):
    """Pooled metrics, confusion matrix, participant-bootstrap CI and within-participant permutation test."""
    S, cols = grel_sources(grel)[source], FEATURE_SETS[features]
    out, preds = [], {}
    for tn, T in nq_subsets(nq).items():
        T = T.reset_index(drop=True)
        p = transfer_predict(S, T, cols, norm, model, held_out_readers)
        r = full_report(T.y, p, T.user_id)
        pr = per_reader_ba(T.y, p, T.user_id)
        out.append(dict(configuration=label, protocol="held-out readers" if held_out_readers else "same readers", target=f"GoogleNQ {tn}",
                        source=source, features=features, normalisation=norm, model=model, **r,
                        readers_above_chance=f"{int((pr > 0.5).sum())}/{int(pr.notna().sum())}"))
        preds[tn] = (T, p)
    return pd.DataFrame(out), preds


def selected_transfer(grel, nq, grid):
    """Selected configuration = highest g-REL CV (selection on g-REL only), plus the baseline
    configuration and the hypothesis-driven convex-hull LDA (selected subset and normalisation), all evaluated once."""
    best = grid.sort_values("grel_cv_BA", ascending=False).iloc[0]
    configs = [("selected on g-REL", best.source, best.features, best.normalisation, best.model),
               ("baseline (LDA, g-REL All, Min-Max)", "g-REL All", "17 features", "Min-Max", "LDA"),
               ("convex hull LDA (hypothesis-driven)", best.source, "2 convex hull features", best.normalisation, "LDA")]
    tables, preds = [], {}
    for label, *cfg in configs:
        t, p = evaluate_transfer(grel, nq, *cfg, label=label)
        tables.append(t)
        preds[label] = p
    t, _ = evaluate_transfer(grel, nq, *configs[0][1:], label=configs[0][0] + " (same readers, for comparison)", held_out_readers=False)
    tables.append(t)
    return best, pd.concat(tables, ignore_index=True), preds


def within_googlenq(nq, cols=F17):
    """Models trained and tested on GoogleNQ (participant folds): class-balanced SVM on per-reader normalised features."""
    out = {}
    for tn, T in nq_subsets(nq).items():
        T = T.reset_index(drop=True)
        Tz = reader_normalised(T, cols)
        p = np.zeros(len(T), int)
        for tr, te in participant_folds(T):
            p[te] = within_googlenq_svm().fit(Tz.iloc[tr][cols], T.y.iloc[tr]).predict(Tz.iloc[te][cols])
        out[tn] = (T, p)
    return out


def transfer_vs_within(transfer_preds, within_preds):
    """Per-reader Wilcoxon signed-rank tests: transfer vs the within-GoogleNQ model, 24 readers as paired units."""
    rows = []
    for tn, (T, p_w) in within_preds.items():
        T2, p_t = transfer_preds[tn]
        assert (T2.index == T.index).all() and (T2.user_id.values == T.user_id.values).all()
        w = wilcoxon_readers(T.y, p_t, p_w, T.user_id)
        rows.append(dict(target=f"GoogleNQ {tn}", transfer_BA=BA(T.y, p_t), within_BA=BA(T.y, p_w),
                         readers_transfer_better=w["a_better"], readers_within_better=w["b_better"], readers_equal=w["equal"],
                         mean_reader_diff=w["mean_diff"], wilcoxon_p=w["p"]))
    return pd.DataFrame(rows)


# ============================================================================ paragraph position (Table S8)
POSITIONS = {"first two paragraphs": lambda q: q <= 1, "next two paragraphs": lambda q: (q >= 2) & (q <= 3),
             "remaining paragraphs": lambda q: q >= 4}


def transfer_by_position(preds):
    """Transfer BA by paragraph position in the GoogleNQ document (Supplementary Table S8)."""
    rows = []
    for tn, (T, p) in preds.items():
        for pos, f in POSITIONS.items():
            m = f(T.paragraph).values
            r = full_report(T.y[m], p[m], T.user_id[m], perm=False)
            rows.append(dict(target=f"GoogleNQ {tn}", position=pos, **r))
    return pd.DataFrame(rows)


# ============================================================================ feature analysis (Figure 7)
def transfer_importance(grel, nq, source, features, norm, model, repeats=20, seed=0, shap_samples=100):
    """Permutation importance ON GoogleNQ All (drop in BA when a feature is shuffled; participant-disjoint transfer
    model of each fold), and SHAP values of the transfer model trained on all g-REL readers (KernelExplainer, on a
    sample of GoogleNQ paragraphs). SHAP is skipped if the shap package is not installed."""
    rng = np.random.default_rng(seed)
    S, cols = grel_sources(grel)[source], FEATURE_SETS[features]
    T = nq_subsets(nq)["All"].reset_index(drop=True)
    drops = {c: [] for c in cols}
    for _, te in participant_folds(T):
        held = T.user_id.iloc[te].unique()
        m = _fit(S[~S.user_id.isin(held)], cols, norm, model)
        X = _prep(T.iloc[te], cols, norm)[cols].reset_index(drop=True)
        y = T.y.iloc[te].values
        base = BA(y, m.predict(X))
        for c in cols:
            vals = []
            for _ in range(repeats):
                Xp = X.copy()
                Xp[c] = rng.permutation(Xp[c].values)
                vals.append(base - BA(y, m.predict(Xp)))
            drops[c].append(np.mean(vals))
    imp = pd.DataFrame({"feature": cols, "BA_drop": [np.mean(drops[c]) for c in cols],
                        "BA_drop_SD_folds": [np.std(drops[c], ddof=1) for c in cols]}).sort_values("BA_drop", ascending=False)
    shap_df = None
    try:
        import shap
        m = _fit(S, cols, norm, model)
        Xs = _prep(S, cols, norm)[cols]
        Xt = _prep(T, cols, norm)[cols].sample(min(shap_samples, len(T)), random_state=seed)
        f = m.decision_function if hasattr(m, "decision_function") else (lambda x: m.predict_proba(x)[:, 1])
        ex = shap.KernelExplainer(lambda x: f(pd.DataFrame(x, columns=cols)), shap.kmeans(Xs, 10))
        sv = ex.shap_values(Xt.values, silent=True)
        shap_df = pd.DataFrame(sv, columns=cols).assign(**{f"value__{c}": Xt[c].values for c in cols})
    except ImportError:
        pass
    return imp.reset_index(drop=True), shap_df


# ============================================================================ PCA (Figure 9)
def pca_separability(grel, nq):
    """Two-component PCA of the 17 features of both corpora (a) after Min-Max fitted on both corpora, (b) after
    per-reader normalisation; a logistic regression separates the corpora on the two components (balanced accuracy)."""
    out = {}
    for lab, (G, N) in [("Min-Max", (grel, nq)), ("per-reader", (reader_normalised(grel, F17), reader_normalised(nq, F17)))]:
        X = MinMaxScaler().fit_transform(pd.concat([G[F17], N[F17]]))
        Z = PCA(2).fit_transform(X)
        y = np.r_[np.zeros(len(G)), np.ones(len(N))]
        clf = LogisticRegression(class_weight="balanced").fit(Z, y)
        out[lab] = dict(Z=Z, corpus=y, clf=clf, BA=BA(y, clf.predict(Z)))
    return out


# ============================================================================ unseen documents (Figure 6, Table S7)
def document_folds(d, k=4, seed=1):
    """k folds of held-out documents (4 folds: 3 documents per fold for All, 2 for Agree)."""
    docs = np.array(sorted(d.document.unique()))
    return [(np.where(~d.document.isin(f))[0], np.where(d.document.isin(f))[0])
            for f in np.array_split(np.random.default_rng(seed).permutation(docs), k)]


def split_schemes(d):
    """participant: the 5 participant folds; document: 4 document folds; both: held-out readers AND documents
    (each of 4 participant folds crossed with each of the 4 document folds)."""
    d = d.reset_index(drop=True)
    from sklearn.model_selection import StratifiedGroupKFold
    from .config import FOLD_SEED
    ufold = [d.user_id.iloc[te].unique() for _, te in StratifiedGroupKFold(4, shuffle=True, random_state=FOLD_SEED).split(d, d.y, d.user_id)]
    dfold = [d.document.iloc[te].unique() for _, te in document_folds(d)]
    both = [(np.where(~d.user_id.isin(u) & ~d.document.isin(f))[0], np.where(d.user_id.isin(u) & d.document.isin(f))[0])
            for u in ufold for f in dfold]
    return {"participant": participant_folds(d), "document": document_folds(d), "both": both}


def unseen_documents(grel):
    """Feature-based models for new readers of known documents vs unseen documents (Supplementary Table S7).
    The class-balanced models matter here: with 2-3 documents per test fold, the class ratio of a document fold
    differs strongly from that of the training documents."""
    from sklearn.naive_bayes import GaussianNB
    models = {"LDA, 17 features": (F17, False, lambda: minmax_pipeline(LinearDiscriminantAnalysis)),
              "LDA, 2 convex hull features": (HULL2, False, lambda: minmax_pipeline(LinearDiscriminantAnalysis)),
              "LDA, 2 convex hull features, per-reader": (HULL2, True, lambda: minmax_pipeline(LinearDiscriminantAnalysis)),
              "balanced LR (C=0.1), 2 convex hull features, per-reader":
                  (HULL2, True, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))),
              "balanced NB, 2 convex hull features, per-reader": (HULL2, True, lambda: make_pipeline(StandardScaler(), GaussianNB(priors=[0.5, 0.5])))}
    rows = []
    for sub in ["All", "Agree"]:
        d = grel_subsets(grel)[sub].reset_index(drop=True)
        for name, (cols, norm, make) in models.items():
            X = (reader_normalised(d, cols) if norm else d)[cols].values
            r = dict(subset=sub, model=name)
            for scheme, sp in split_schemes(d).items():
                p = np.full(len(d), -1)
                for tr, te in sp:
                    if len(te) and d.y.iloc[tr].nunique() == 2:
                        p[te] = make().fit(X[tr], d.y.iloc[tr]).predict(X[te])
                ok = p >= 0
                r[f"BA_{scheme}"] = BA(d.y[ok], p[ok])
            rows.append(r)
    return pd.DataFrame(rows)


def document_identification(grel):
    """How well do the 17 features identify WHICH g-REL document was read (held-out readers, random forest)?"""
    d = grel.reset_index(drop=True)
    acc = []
    for tr, te in participant_folds(d):
        rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(d.iloc[tr][F17], d.document.iloc[tr])
        acc.append((rf.predict(d.iloc[te][F17]) == d.document.iloc[te]).mean())
    return dict(accuracy=float(np.mean(acc)), chance=1 / d.document.nunique(), documents=d.document.nunique())


# ============================================================================ statistical tests
def fold_level_tests(fold_ba: dict, comparisons):
    """fold_ba: {(subset, model): array of 5 fold BAs}. Corrected resampled t-test + Holm over all comparisons."""
    rows = []
    for sub, a, b in comparisons:
        A, B = np.asarray(fold_ba[(sub, a)]), np.asarray(fold_ba[(sub, b)])
        rows.append(dict(subset=sub, A=a, B=b, mean_A=A.mean(), mean_B=B.mean(), **corrected_ttest(A, B)))
    t = pd.DataFrame(rows)
    t["p_corrected_Holm"] = holm(t.p_corrected)
    return t


def reader_level_tests(pred: pd.DataFrame, pairs):
    """pred: one row per trial with columns subset, user_id, y and one 0/1 prediction column per model.
    Wilcoxon signed-rank test over readers + Holm."""
    rows = []
    for sub, a, b in pairs:
        d = pred[pred.subset == sub]
        w = wilcoxon_readers(d.y, d[a], d[b], d.user_id)
        rows.append(dict(subset=sub, A=a, B=b, BA_A=BA(d.y, d[a]), BA_B=BA(d.y, d[b]), **w))
    t = pd.DataFrame(rows)
    t["p_Holm"] = holm(t.p)
    return t


# ============================================================================ protocol checks
def leakage_checks(grel, nq):
    """Assertions that the protocol has no reader or label leakage. Returns a table of (check, passed)."""
    out = []
    # 1. outer folds are participant-disjoint and cover every trial once
    for name, d in list(grel_subsets(grel).items()) + list(nq_subsets(nq).items()):
        d = d.reset_index(drop=True)
        f = participant_folds(d)
        ok = all(not set(d.user_id.iloc[a]) & set(d.user_id.iloc[b]) for a, b in f) and sorted(np.concatenate([b for _, b in f])) == list(range(len(d)))
        out.append((f"participant-disjoint folds covering all trials ({name})", ok))
    # 2. per-reader normalisation does not depend on labels: recompute with shuffled labels
    from .evaluation import add_reader_normalisation
    shuffled = grel.assign(label=np.random.default_rng(0).permutation(grel.label.values))
    a = add_reader_normalisation(grel[["user_id"] + F17])
    b = add_reader_normalisation(shuffled[["user_id"] + F17])
    out.append(("per-reader normalisation is label-free", np.allclose(a.filter(like="__pz"), b.filter(like="__pz"))))
    # 3. per-reader statistics are computed over all trials of the corpus, not within the (label-defined) Agree subset
    ag = grel_subsets(grel)["Agree"]
    out.append(("Agree uses statistics of all g-REL trials of the reader",
                np.allclose(ag.filter(like="__pz").values, grel.loc[ag.index].filter(like="__pz").values)))
    # 4. in the transfer, the held-out GoogleNQ readers are never in the g-REL training data
    T = nq.reset_index(drop=True)
    ok = True
    for _, te in participant_folds(T):
        held = set(T.user_id.iloc[te])
        ok &= not held & set(grel[~grel.user_id.isin(held)].user_id)
    out.append(("transfer: held-out readers excluded from g-REL training", ok))
    return pd.DataFrame(out, columns=["check", "passed"])
