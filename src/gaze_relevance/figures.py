"""Figures of the paper (7-9) and explanatory figures of the notebook. Fonts are embedded (TrueType) and features\ncarry readable names."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .config import FEATURE_LABELS

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 9})
COLORS = {"g-REL": "#3b6fb6", "GoogleNQ": "#d9822b"}


def feature_analysis(imp, shap_df, path=None):
    """Figure 7: (a) permutation importance of the transfer model on GoogleNQ, (b) SHAP values (beeswarm-like)."""
    has_shap = shap_df is not None
    fig, axes = plt.subplots(1, 2 if has_shap else 1, figsize=(11 if has_shap else 5.5, 0.32 * len(imp) + 1.2), squeeze=False)
    ax = axes[0, 0]
    o = imp.iloc[::-1]
    ax.barh([FEATURE_LABELS[f] for f in o.feature], o.BA_drop, xerr=o.BA_drop_SD_folds, color="#3b6fb6", alpha=0.85)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Drop in balanced accuracy on GoogleNQ when shuffled")
    ax.set_title("(a) Permutation importance", loc="left")
    if has_shap:
        ax = axes[0, 1]
        feats = list(imp.feature)[::-1]
        rng = np.random.default_rng(0)
        for i, f in enumerate(feats):
            v, x = shap_df[f].values, shap_df[f"value__{f}"].values
            c = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)
            ax.scatter(v, i + rng.uniform(-0.25, 0.25, len(v)), c=c, cmap="coolwarm", s=6, lw=0)
        ax.set_yticks(range(len(feats)), [FEATURE_LABELS[f] for f in feats])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("SHAP value (impact on the decision function)")
        ax.set_title("(b) SHAP values (colour: feature value, low to high)", loc="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def distributions(grel, nq, features=("f_hull_area_per_time", "f_fixns_per_hull_area", "f_fixn_n"), path=None):
    """Figure 8: feature distributions of both corpora, before (top) and after (bottom) per-reader normalisation."""
    fig, axes = plt.subplots(2, len(features), figsize=(3.6 * len(features), 5.4))
    for j, f in enumerate(features):
        for i, suffix in enumerate(["", "__pz"]):
            ax = axes[i, j]
            for name, d in [("g-REL", grel), ("GoogleNQ", nq)]:
                for lab, ls in [(1, "-"), (0, "--")]:
                    v = d.loc[d.y == lab, f + suffix].values
                    if i == 0:
                        v = np.log10(v[v > 0])
                    ax.hist(v, bins=30, density=True, histtype="step", ls=ls, color=COLORS[name],
                            label=f"{name}, {'relevant' if lab else 'irrelevant'}")
            ax.set_title(FEATURE_LABELS[f], fontsize=9)
            ax.set_xlabel("log10(value)" if i == 0 else "per-reader z-score")
    axes[0, 0].set_ylabel("density (raw values)")
    axes[1, 0].set_ylabel("density (per-reader normalised)")
    axes[0, -1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def pca(pca_result, path=None):
    """Figure 9: two-component PCA of the 17 features, (a) Min-Max, (b) per-reader normalised, with the boundary of the
    logistic regression that separates the corpora."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (key, title) in zip(axes, [("Min-Max", "(a) Min-Max normalisation"), ("per-reader", "(b) per-reader normalisation")]):
        r = pca_result[key]
        Z, c = r["Z"], r["corpus"]
        for val, name in [(0, "g-REL"), (1, "GoogleNQ")]:
            ax.scatter(Z[c == val, 0], Z[c == val, 1], s=6, alpha=0.5, color=COLORS[name], label=name, lw=0)
        w, b = r["clf"].coef_[0], r["clf"].intercept_[0]
        xs = np.linspace(Z[:, 0].min(), Z[:, 0].max(), 50)
        if abs(w[1]) > 1e-9:
            ys = -(w[0] * xs + b) / w[1]
            keep = (ys > Z[:, 1].min()) & (ys < Z[:, 1].max())
            ax.plot(xs[keep], ys[keep], "k--", lw=1, label="logistic regression boundary")
        ax.set_title(f"{title}\ncorpus separability: balanced accuracy {r['BA']:.2f}", loc="left")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig





# ----------------------------------------------------------------------------- explanatory figures of the notebook
def gaze_trial(df, title="", ax=None, absolute=False):
    """Raw gaze samples of one recording, coloured by time; fixations (mean position per fixation_id) as circles
    whose size is proportional to the fixation duration. absolute=True plots the position on the (scrolled) page."""
    ycol = "gaze_y_abs" if absolute else "gaze_y"
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    t = df.timestamp - df.timestamp.min()
    ax.scatter(df.gaze_x, df[ycol], c=t, cmap="viridis", s=1, alpha=0.4)
    fx = df.dropna(subset=["fixation_id"]).groupby("fixation_id").agg(x=("gaze_x", "mean"), y=(ycol, "mean"),
                                                                     t0=("timestamp", "min"), t1=("timestamp", "max"))
    ax.scatter(fx.x, fx.y, s=800 * (fx.t1 - fx.t0), facecolors="none", edgecolors="k", lw=0.6)
    ax.set_xlim(0, 2560)
    if not absolute:
        ax.set_ylim(1440, 0)
    else:
        ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y on the page (px)" if absolute else "y on the screen (px)")
    ax.set_title(title, loc="left", fontsize=9)
    return ax


def paragraphs_of_document(df, labels, ax=None):
    """GoogleNQ document: gaze on the page (absolute y), coloured by the paragraph it falls on; paragraphs the reader
    rated relevant are marked."""
    ax = ax or plt.subplots(figsize=(5, 6))[1]
    cmap = plt.get_cmap("tab10")
    for p, g in df[df.paragraph_id >= 0].groupby("paragraph_id"):          # negative ids: gaze outside the paragraphs
        p = int(p)
        ax.scatter(g.gaze_x, g.gaze_y_abs, s=1, color=cmap(p % 10), alpha=0.5)
        ax.text(2580, g.gaze_y_abs.median(), f"P{p} {'relevant' if labels[p] else 'irrelevant'}", fontsize=8,
                color=cmap(p % 10), va="center", fontweight="bold" if labels[p] else "normal")
    ax.set_xlim(0, 3300)
    ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y on the page (px)")
    return ax


def normalisation_demo(grel, nq, feature="f_fixns_per_hull_area", readers=4, path=None):
    """Why per-reader normalisation: raw feature values per reader and corpus (left) vs per-reader z-scores (right)."""
    users = sorted(grel.user_id.unique())[:readers]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, col, lab in [(axes[0], feature, "raw value"), (axes[1], feature + "__pz", "per-reader z-score")]:
        for i, u in enumerate(users):
            for j, (name, d) in enumerate([("g-REL", grel), ("GoogleNQ", nq)]):
                v = d.loc[d.user_id == u, col]
                ax.boxplot(v, positions=[i * 3 + j], widths=0.8, patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=COLORS[name], alpha=0.6), medianprops=dict(color="k"))
        ax.set_xticks([i * 3 + 0.5 for i in range(len(users))], users)
        ax.set_xlabel("reader")
        ax.set_ylabel(lab)
    axes[0].set_title(f"(a) {FEATURE_LABELS[feature]}: raw", loc="left")
    axes[1].set_title("(b) after per-reader normalisation", loc="left")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=COLORS[k], alpha=0.6, label=k) for k in COLORS], frameon=False, fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def confusion_matrices(table, path=None):
    """Pooled confusion matrices (rows: true class, columns: predicted class) of a table with TN, FP, FN, TP."""
    n = len(table)
    cols = min(n, 3 if n > 4 else n)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.8 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, table.iterrows()):
        m = np.array([[r.TN, r.FP], [r.FN, r.TP]])
        ax.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for k in range(2):
                ax.text(k, i, int(m[i, k]), ha="center", va="center", fontsize=10)
        ax.set_xticks([0, 1], ["irrelevant", "relevant"], fontsize=8)
        ax.set_yticks([0, 1], ["irrelevant", "relevant"], fontsize=8)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("perceived (true)", fontsize=8)
        ax.set_title(f"{r.get('short', '')}\n{r.target}: BA {r.BA:.3f}", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig
