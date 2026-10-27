"""
14_feature_selection.py: from 16,333 features to a small, stable and interpretable panel.

Learned on the training part of a design only (default: balanced); the test part is not read.

    1 quality      drop features missing in > 20% of training samples or constant
    2 relevance    single-feature AUC (Mann-Whitney) for cancer vs control, Benjamini-Hochberg FDR < 0.05
    3 redundancy   greedy pruning: features ranked by |AUC - 0.5|; a feature is kept only if its Spearman
                   correlation with every kept feature is below 0.9 (the families overlap strongly)
    4 stability    stability selection (Meinshausen & Buehlmann 2010): L1 logistic regression (C = 0.2,
                   ~80 features per fit) on 200 stratified half-samples; the panel is every feature
                   selected in >= 50% of them

The same steps run inside the cross-validation folds of 15 and 16, so the model evaluation never sees a
panel chosen on its own test samples. This script reports what the selection keeps and why.

Outputs (results/modelling/14_features/): funnel.json, features.tsv (all features with their statistics),
panel.tsv, family_funnel.tsv, fig_funnel.png, fig_families.png, fig_volcano.png, fig_stability.png
"""

import argparse
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from ml_common import FAMILIES, OUT, SEED, designs, family, load, log, logreg, out_dir, region, strata  # noqa: E402

warnings.filterwarnings("ignore")


def column_auc(X, y):
    """AUC and two-sided Mann-Whitney p of every column, NaN-aware."""
    R = X.rank().to_numpy()
    seen = ~np.isnan(R)
    R = np.nan_to_num(R)
    pos = np.asarray(y) == 1
    n1, n0 = seen[pos].sum(0), seen[~pos].sum(0)
    U = R[pos].sum(0) - n1 * (n1 + 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = U / (n1 * n0)
        z = (U - n1 * n0 / 2) / np.sqrt(n1 * n0 * (n1 + n0 + 1) / 12)
    return auc, 2 * stats.norm.sf(np.abs(z))


def standardise(X):
    Z = X.fillna(X.median())
    return ((Z - Z.mean()) / Z.std(ddof=0).replace(0, 1)).to_numpy()


def prune_correlated(X, order, thr=0.9):
    """Greedy: walk the features in `order`, keep one unless |Spearman| >= thr with an already kept one."""
    R = X[order].rank()
    R = R.fillna(R.mean())
    Z = ((R - R.mean()) / R.std(ddof=0).replace(0, 1)).to_numpy() / np.sqrt(len(R))
    kept = []
    for j in range(Z.shape[1]):
        if kept and np.max(np.abs(Z[:, kept].T @ Z[:, j])) >= thr:
            continue
        kept.append(j)
    return [order[j] for j in kept]


def select(X, y, max_missing=0.2, fdr=0.05, corr=0.9):
    """Steps 1-3 on (X, y); returns the pruned feature list and the per-feature statistics."""
    miss = X.isna().mean()
    sd = X.std()
    q_ok = (miss <= max_missing) & (sd > 1e-9)
    Xq = X.loc[:, q_ok]
    auc, p = column_auc(Xq, y)
    q = multipletests(np.nan_to_num(p, nan=1.0), method="fdr_bh")[1]
    st = pd.DataFrame({"auc": auc, "p": p, "q": q}, index=Xq.columns)
    st["effect"] = (st["auc"] - 0.5).abs()
    sig = st.index[st["q"] < fdr]
    order = list(st.loc[sig].sort_values("effect", ascending=False).index)
    kept = prune_correlated(Xq, order, corr) if order else []
    return kept, st, q_ok


def l1_once(Z, y, idx, C):
    m = logreg(l1_ratio=1.0, C=C)
    m.fit(Z[idx], y[idx])
    return m.coef_[0] != 0


def stability(X, y, key, n=200, C=0.2, jobs=-1):
    Z = standardise(X)
    split = StratifiedShuffleSplit(n_splits=n, train_size=0.5, random_state=SEED)
    sel = Parallel(n_jobs=jobs)(delayed(l1_once)(Z, y, tr, C) for tr, _ in split.split(Z, key))
    return pd.Series(np.mean(sel, axis=0), index=X.columns)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--design", default="balanced", choices=["balanced", "study"])
    ap.add_argument("--subsamples", type=int, default=200)
    ap.add_argument("--min-freq", type=float, default=0.5)
    ap.add_argument("--panel-size", type=int, default=25, help="Minimum panel size (most stable first).")
    ap.add_argument("--C", type=float, default=0.2, help="L1 strength of the stability selection.")
    args = ap.parse_args()
    out = out_dir("14_features")
    samples, X, _ = load()
    tr, _ = designs(samples)[args.design]
    Xt, st_s = X.loc[tr], samples.loc[tr]
    y = st_s["ml_label"].to_numpy()
    log(f"design {args.design}: {len(tr)} training samples, {X.shape[1]} features")

    kept, st, q_ok = select(Xt, y)
    log(f"quality {int(q_ok.sum())}, FDR<0.05 {int((st['q'] < 0.05).sum())}, non-redundant {len(kept)}")
    freq = stability(Xt[kept], y, strata(st_s, with_demographics=False).to_numpy(), n=args.subsamples, C=args.C)
    n_stable = int((freq >= args.min_freq).sum())
    panel = freq.sort_values(ascending=False).head(max(n_stable, args.panel_size))
    log(f"stable panel: {len(panel)} features ({n_stable} with selection frequency >= {args.min_freq})")

    feats = pd.DataFrame(index=X.columns)
    feats["family"] = [family(c) for c in X.columns]
    feats["region"] = [region(c) for c in X.columns]
    feats["quality_ok"] = q_ok.reindex(X.columns).fillna(False)
    feats = feats.join(st)
    feats["significant"] = feats["q"] < 0.05
    feats["non_redundant"] = feats.index.isin(kept)
    feats["stability"] = freq.reindex(feats.index)
    feats["in_panel"] = feats.index.isin(panel.index)
    feats["direction"] = np.where(feats["auc"] >= 0.5, "higher in cancer", "lower in cancer")
    ann_path = OUT / "12_annotation/region_genes.tsv"
    if ann_path.exists():
        ann = pd.read_csv(ann_path, sep="\t", index_col="region_id")
        feats = feats.join(ann[["gene", "tss_distance", "promoter"]], on="region")
    feats.to_csv(out / "features.tsv", sep="\t")
    feats.loc[panel.index].sort_values("stability", ascending=False).to_csv(out / "panel.tsv", sep="\t")

    steps = ["all", "quality_ok", "significant", "non_redundant", "in_panel"]
    fam = pd.DataFrame({s: (feats.groupby("family").size() if s == "all" else feats[feats[s] == True]  # noqa: E712
                            .groupby("family").size()) for s in steps}).fillna(0).astype(int)
    fam = fam.loc[[f for f in FAMILIES if f in fam.index]]
    fam.to_csv(out / "family_funnel.tsv", sep="\t")
    funnel = {"design": args.design, "n_train": len(tr), "cancer_train": int(y.sum()),
              **{s: int(fam[s].sum()) for s in steps}, "min_freq": args.min_freq, "n_freq_ge_min": n_stable,
              "subsamples": args.subsamples, "C": args.C}
    (out / "funnel.json").write_text(json.dumps(funnel, indent=2))
    log("family funnel\n" + fam.to_string())

    labels = ["All features", "Quality filter", "Significant\n(FDR < 5%)", "Non-redundant\n(|rho| < 0.9)",
              "Stable panel\n(most stable)"]
    vals = [funnel[s] for s in steps]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(range(5)[::-1], vals, color=["#0A2F5C", "#00579B", "#0077B6", "#00A19A", "#E4572E"])
    for i, v in enumerate(vals):
        ax.text(v, 4 - i, f"  {v:,}", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(range(5)[::-1], labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Number of features (log scale)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_funnel.png", dpi=170)
    plt.close(fig)

    share = fam.div(fam.sum(axis=0), axis=1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    left = np.zeros(len(steps))
    cmap = plt.get_cmap("tab10")
    for k, f in enumerate(share.index):
        ax.barh(range(len(steps))[::-1], share.loc[f], left=left, color=cmap(k), label=FAMILIES[f])
        left += share.loc[f].to_numpy()
    ax.set_yticks(range(len(steps))[::-1], [lab.replace("\n", " ") for lab in labels], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of the features at each step")
    ax.legend(fontsize=7.5, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_families.png", dpi=170)
    plt.close(fig)

    v = feats[feats["quality_ok"] == True]  # noqa: E712
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(v["auc"], -np.log10(v["q"].clip(lower=1e-30)), s=5, color="#9CA3AF", alpha=0.5, linewidth=0)
    pv = v[v["in_panel"]]
    ax.scatter(pv["auc"], -np.log10(pv["q"].clip(lower=1e-30)), s=18, color="#E4572E", label="stable panel")
    ax.axhline(-np.log10(0.05), color="#6B7280", linestyle=":", linewidth=1)
    ax.axvline(0.5, color="#6B7280", linewidth=1)
    ax.set_xlabel("Single-feature AUC (cancer vs control)")
    ax.set_ylabel("-log10 FDR")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_volcano.png", dpi=170)
    plt.close(fig)

    top = feats.loc[panel.index].head(25)
    lab = [f"{FAMILIES[f].split(' (')[0]} · {r}" + (f" ({g})" if "gene" in top and isinstance(g, str) else "")
           for f, r, g in zip(top["family"], top["region"], top.get("gene", pd.Series([None] * len(top))))]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(top))[::-1], top["stability"], color=["#E4572E" if a >= 0.5 else "#2A78D6" for a in top["auc"]])
    ax.set_yticks(range(len(top))[::-1], lab, fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Selection frequency over half-samples (orange: higher in cancer)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_stability.png", dpi=170)
    plt.close(fig)
    log(f"-> {out}")


if __name__ == "__main__":
    main()
