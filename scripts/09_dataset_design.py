"""
09_dataset_design.py: train / test designs and their balance audit.

Three designs:
    balanced   hold-out of 30%, stratified on label x sequencing batch x sex x age band, so cancers and
               controls of both batches, both sexes and every age band are split in the same proportions
    study      the study's own groups (TV -> SBT), kept for comparison with the publication
    lobo       leave one batch out: train on the larger sequencing batch, test on the smaller one

For each design: class balance, batch, sex and age composition, and standardised mean differences
(SMD) of the technical covariates between train and test, and between cancers and controls inside the
training set (|SMD| > 0.2 flags an imbalance worth correcting or reporting).

Outputs (results/modelling/09_design/): splits.tsv, balance.tsv, smd.tsv, fig_balance.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml_common import age_band, designs, load, log, out_dir  # noqa: E402

COVARIATES = ["age", "dna_input_ng", "median_region_depth", "n_fragments", "r2_end_ch_meth"]


def smd(a, b):
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    sd = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def main():
    out = out_dir("09_design")
    samples, X, _ = load()
    D = designs(samples)
    split = pd.DataFrame(index=samples.index)
    rows, smd_rows = [], []
    for name, (tr, te) in D.items():
        split[name] = "train"
        split.loc[te, name] = "test"
        for part, idx in (("train", tr), ("test", te)):
            s = samples.loc[idx]
            rows.append({"design": name, "part": part, "n": len(s), "cancer": int(s["ml_label"].sum()),
                         "cancer_share": s["ml_label"].mean(), "female_share": (s["gender"] == "F").mean(),
                         "age_mean": s["age"].mean(), "benign_controls": int((s["stage"] == "Benign").sum()),
                         **{f"batch_{b}": int((s["submission_series"] == b).sum()) for b in
                            sorted(samples["submission_series"].unique())},
                         **{f"cancer_batch_{b}": int(((s["submission_series"] == b) & (s["ml_label"] == 1)).sum())
                            for b in sorted(samples["submission_series"].unique())}})
        s_tr = samples.loc[tr]
        for c in COVARIATES:
            smd_rows.append({"design": name, "covariate": c, "train_vs_test": smd(samples.loc[tr, c], samples.loc[te, c]),
                             "cancer_vs_control_in_train": smd(s_tr.loc[s_tr["ml_label"] == 1, c],
                                                               s_tr.loc[s_tr["ml_label"] == 0, c])})
    bal = pd.DataFrame(rows)
    smd_t = pd.DataFrame(smd_rows)
    split.to_csv(out / "splits.tsv", sep="\t")
    bal.to_csv(out / "balance.tsv", sep="\t", index=False)
    smd_t.to_csv(out / "smd.tsv", sep="\t", index=False)
    log("balance\n" + bal.round(3).to_string(index=False))
    log("SMD\n" + smd_t.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, (title, col) in zip(axes, [("Label", "ml_label"), ("Sequencing batch", "submission_series"),
                                        ("Sex", "gender"), ("Age band", None)]):
        labels, data = [], []
        for name in D:
            for part in ("train", "test"):
                idx = split.index[split[name] == part]
                v = age_band(samples.loc[idx, "age"]) if col is None else samples.loc[idx, col].astype(str)
                data.append(v.value_counts(normalize=True))
                labels.append(f"{name}\n{part}")
        frame = pd.DataFrame(data, index=labels).fillna(0)
        frame.plot.bar(stacked=True, ax=ax, width=0.75, colormap="tab20c", legend=True)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=0, labelsize=8)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_balance.png", dpi=160)
    log(f"-> {out}")


if __name__ == "__main__":
    main()
