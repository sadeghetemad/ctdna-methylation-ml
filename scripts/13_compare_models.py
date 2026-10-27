"""
13_compare_models.py: all models of 11 (classical) and 12 (deep learning) side by side, on one design.

    table        CV AUC, test AUC with bootstrap 95% CI, partial AUC (FPR <= 5%), sensitivity and
                 specificity at the threshold fixed on the training part, balanced accuracy
    paired test  bootstrap of the test-AUC difference between the best model (by CV AUC) and every other
    benchmark    study design only: the published classifier's calls on the same test patients (Supplementary
                 Table 7) against each model at the published specificity (threshold matched on the test
                 controls, which favours our models); on the other designs the published classifier was
                 trained on some of the test patients, so it is not compared
    fairness     test AUC per subgroup (sex, age band, sequencing batch) for every model
    figures      ROC curves, AUC bars, fairness heatmap, calibration of the best model

Outputs (results/modelling/13_compare/<design>/): comparison.tsv, paired.tsv, fairness.tsv,
benchmark.tsv, fig_roc.png, fig_auc.png, fig_fairness.png, fig_calibration.png, fig_benchmark.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from ml_common import DESIGNS, OUT, SEED, log, out_dir, published_calls  # noqa: E402

KIND_COLOR = {"linear": "#2A78D6", "kernel": "#8A5CC2", "tree": "#1BAF7A", "deep": "#E4572E"}


def main():
    ap = argparse.ArgumentParser(description="Compare classical and deep-learning models on one design.")
    ap.add_argument("--design", default="balanced", choices=list(DESIGNS))
    args = ap.parse_args()
    out = out_dir(f"13_compare/{args.design}")
    res, pred, sub = [], None, []
    for part in ("11_classical", "12_deep"):
        d = OUT / part / args.design
        if not (d / "results.tsv").exists():
            log(f"missing {d}; skipped")
            continue
        res.append(pd.read_csv(d / "results.tsv", sep="\t"))
        p = pd.read_csv(d / "predictions.tsv", sep="\t").set_index("sample_id")
        pred = p if pred is None else pred.join(p.drop(columns="ml_label"))
        sub.append(pd.read_csv(d / "subgroups.tsv", sep="\t"))
    R = pd.concat(res).sort_values("cv_auc", ascending=False).reset_index(drop=True)
    S = pd.concat(sub)
    y = pred["ml_label"].to_numpy()
    best = R[R["family"] != "reference"].iloc[0]["model"]

    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
    boots = [b for b in boots if 0 < y[b].sum() < len(b)]
    pb = pred[best].to_numpy()
    rows = []
    for m in R["model"]:
        pm = pred[m].to_numpy()
        diff = np.array([roc_auc_score(y[b], pb[b]) - roc_auc_score(y[b], pm[b]) for b in boots])
        rows.append({"model": m, "vs": best, "auc_diff": roc_auc_score(y, pb) - roc_auc_score(y, pm),
                     "ci_lo": np.percentile(diff, 2.5), "ci_hi": np.percentile(diff, 97.5),
                     "p_best_not_better": float((diff <= 0).mean())})
    P = pd.DataFrame(rows)
    R.to_csv(out / "comparison.tsv", sep="\t", index=False)
    P.to_csv(out / "paired.tsv", sep="\t", index=False)
    fair = S.pivot_table(index="model", columns="group", values="auc").reindex(R["model"])
    fair.to_csv(out / "fairness.tsv", sep="\t")
    log("comparison\n" + R[["model", "family", "cv_auc", "test_auc", "test_auc_lo", "test_auc_hi", "test_pauc",
                            "test_sens", "test_spec"]].round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for _, r in R.iterrows():
        fpr, tpr, _ = roc_curve(y, pred[r["model"]])
        ax.plot(fpr, tpr, color=KIND_COLOR.get(r["family"], "#6B7280"), alpha=1 if r["model"] == best else 0.55,
                linewidth=2.6 if r["model"] == best else 1.3, label=f"{r['model']} ({r['test_auc']:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", linewidth=1)
    ax.axvline(0.05, color="#9CA3AF", linestyle=":", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_roc.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(R) + 1.2))
    yy = np.arange(len(R))[::-1]
    ax.errorbar(R["test_auc"], yy, xerr=[R["test_auc"] - R["test_auc_lo"], R["test_auc_hi"] - R["test_auc"]],
                fmt="none", ecolor="#CBD5E1", elinewidth=3)
    ax.scatter(R["test_auc"], yy, c=[KIND_COLOR.get(k, "#6B7280") for k in R["family"]], s=60, zorder=3, label="test")
    ax.scatter(R["cv_auc"], yy, marker="|", s=160, color="#0A2F5C", zorder=4, label="cross-validation")
    ax.set_yticks(yy, R["model"], fontsize=8.5)
    ax.axvline(0.5, color="#9CA3AF", linestyle=":")
    ax.set_xlim(0.45, 1)
    ax.set_xlabel("AUC (dot: test with 95% CI; bar: cross-validation)")
    ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_auc.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(1.0 * fair.shape[1] + 2.5, 0.4 * len(fair) + 1.3))
    im = ax.imshow(fair.to_numpy(dtype=float), cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(fair.shape[1]), fair.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(fair)), fair.index, fontsize=8)
    for i in range(fair.shape[0]):
        for j in range(fair.shape[1]):
            v = fair.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="test AUC")
    fig.tight_layout()
    fig.savefig(out / "fig_fairness.png", dpi=170)
    plt.close(fig)

    frac, mean = calibration_curve(y, pred[best], n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF")
    ax.plot(mean, frac, marker="o", color="#00A19A")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed cancer share")
    ax.set_title(f"Calibration: {best}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_calibration.png", dpi=170)
    plt.close(fig)
    if args.design == "study":
        pub = published_calls().reindex(pred.index).to_numpy()
        ok = ~np.isnan(pub)
        yy, pc = y[ok], pub[ok].astype(int)
        p_sens, p_spec = pc[yy == 1].mean(), 1 - pc[yy == 0].mean()
        rows = [{"model": "Published ELSA-seq classifier", "family": "published", "sens": p_sens, "spec": p_spec}]
        for _, r in R.iterrows():
            sc = pred[r["model"]].to_numpy()[ok]
            thr = np.quantile(sc[yy == 0], p_spec)
            rows.append({"model": r["model"], "family": r["family"], "sens": float((sc[yy == 1] > thr).mean()),
                         "spec": float((sc[yy == 0] <= thr).mean())})
        B = pd.DataFrame(rows)
        B.to_csv(out / "benchmark.tsv", sep="\t", index=False)
        fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(B) + 1.2))
        yb = np.arange(len(B))[::-1]
        ax.barh(yb, B["sens"], color=["#0A2F5C" if f == "published" else KIND_COLOR.get(f, "#9CA3AF") for f in B["family"]])
        for yv, v in zip(yb, B["sens"]):
            ax.text(v + 0.01, yv, f"{v:.0%}", va="center", fontsize=8)
        ax.set_yticks(yb, B["model"], fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"Stage I cancers found at the published specificity ({p_spec:.1%})")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "fig_benchmark.png", dpi=170)
        plt.close(fig)
        log(f"published classifier: sens {p_sens:.3f} at spec {p_spec:.3f}; best of ours "
            f"{B.iloc[1:]['sens'].max():.3f}")
    log(f"best by CV AUC: {best} -> {out}")


if __name__ == "__main__":
    main()
