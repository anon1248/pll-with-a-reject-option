""" Script to create all plots and tables. """

import random
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm.notebook import tqdm
from typing import List
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
from glob import glob


dataset_kind_to_idx = {
    "uci": 0,
    "rl": 1,
    "mnistlike": 2,
}
idx_to_dataset_kind = {v: k for k, v in dataset_kind_to_idx.items()}

aug_type_to_idx = {
    "rl": 0,
    "uniform": 1,
    "class-dependent": 2,
    "instance-dependent": 3,
}
idx_to_aug_type = {v: k for k, v in aug_type_to_idx.items()}

algos_to_idx = {
    # ML methods
    "pl-knn-2005": 0,
    "pl-svm-2008": 1,
    "ipal-2015": 2,
    "pl-ecoc-2017": 3,
    # Deep Learning methods
    "proden-2020": 4,
    "rc-2020": 5,
    "cc-2020": 6,
    "valen-2021": 7,
    "pop-2023": 8,
    # Our method
    "dst-pll-2024": 9,
}
idx_to_algos = {v: k for k, v in algos_to_idx.items()}

datasets_to_idx = {
    "ecoli": (0, "uci"),
    "first-order-theorem": (1, "uci"),
    "mfeat-fourier": (2, "uci"),
    "pendigits": (3, "uci"),
    "semeion": (4, "uci"),
    "statlog-landsat-satellite": (5, "uci"),
    "flare": (6, "uci"),
    # Real-world datasets
    "bird-song": (7, "rl"),
    "mir-flickr": (8, "rl"),
    "msrc-v2": (9, "rl"),
    "yahoo-news": (10, "rl"),
    # MNIST datasets
    "mnist": (11, "mnistlike"),
    "fmnist": (12, "mnistlike"),
    "kmnist": (13, "mnistlike"),
}
idx_to_datasets = {v: k for k, (v, _) in datasets_to_idx.items()}

split_to_idx = {
    "train": 0,
    "test": 1,
    "holdout": 2,
}
idx_to_split = {v: k for k, v in split_to_idx.items()}

# Load data
concat_df = []
algo_count = {}
all_keys = []
for fname in sorted(glob("./results/*.parquet.gz")):
    single_exp_df = pd.read_parquet(fname)
    algo_idx = int(single_exp_df.loc[0, "algo"])
    key_tup = list(
        single_exp_df.loc[0, ["dataset", "datasetkind", "algo", "seed", "augmenttype"]])
    algo_count[algo_idx] = algo_count.get(algo_idx, 0) + 1

    if -1 not in list(single_exp_df["predlabel"].unique()):
        concat_df.append(single_exp_df)
        all_keys.append(key_tup)
    else:
        print(f"Failed Algo: {idx_to_algos[algo_idx]}, \"{fname}\"")

runall_data = pd.concat(concat_df)
group_res = runall_data.query("split == 1").groupby(
    ["augmenttype", "dataset", "datasetkind", "algo", "seed"],
    as_index=False,
)["correct"].mean()

all_res = []
for i, (augmenttype_name, augmenttype_idx) in enumerate(aug_type_to_idx.items()):
    for datasetkind_name, datasetkind_idx in dataset_kind_to_idx.items():
        # DST-PLL results
        dst_res = group_res.query(
            f"algo == {algos_to_idx['dst-pll-2024']} and augmenttype == {augmenttype_idx} " +
            f"and datasetkind == {datasetkind_idx}"
        ).sort_values("dataset")["correct"]
        if dst_res.shape[0] == 0:
            continue
        j = 0 if datasetkind_idx == 0 else (1 if datasetkind_idx == 2 else 2)

        # Algo results
        for k, (algo_name, algo_idx) in enumerate(algos_to_idx.items()):
            if algo_name == "dst-pll-2024":
                continue
            algo_res = group_res.query(
                f"algo == {algo_idx} and augmenttype == {augmenttype_idx} " +
                f"and datasetkind == {datasetkind_idx}"
            ).sort_values("dataset")["correct"]

            stat_test = ttest_ind(dst_res, algo_res).pvalue

            all_res.append((
                k * 100 + j * 10 + i, augmenttype_name, datasetkind_name, algo_name,
                f"{float(algo_res.mean() * 100):.2f}", float(algo_res.std()
                                                             ), stat_test < 0.05, float(f"{stat_test:.6f}"),
            ))

        all_res.append((
            9 * 100 + j * 10 + i, augmenttype_name, datasetkind_name, "dst-pll-2024",
            f"{float(dst_res.mean() * 100):.2f}", float(dst_res.std()), False, 1,
        ))

all_stat_res_df = pd.DataFrame(all_res, columns=[
    "idx", "augmenttype", "datasetkind", "algo",
    "mean", "std", "signif_diff_to_dst_pll", "pval",
])

print("Accuracy table:")
for i, tup in enumerate(all_stat_res_df.sort_values("idx")[["mean", "std"]].itertuples()):
    if (i % 7) in (0, 3):
        continue
    std_str = f"{float(tup[2]) * 100:.1f}"
    if len(std_str.split(".")[0]) == 1:
        phan = "\\phantom{0}"
        print(
            f"{float(tup[1]):.1f} ($\\pm$ {phan}{float(tup[2]) * 100:.1f})", end="")
    else:
        print(
            f"{float(tup[1]):.1f} ($\\pm$ {float(tup[2]) * 100:.1f})", end="")
    if i % 7 == 6:
        print()
    else:
        print(" & ", end="")
print()
print()


def get_reject_trajectory(
    data: pd.DataFrame, augmenttype_name: str, datasetkind_name: str, dataset_names: List[str], algo_name: str,
):
    algo_idx = algos_to_idx[algo_name]
    augmenttype_idx = aug_type_to_idx[augmenttype_name]
    datasetkind_idx = dataset_kind_to_idx[datasetkind_name]

    xs = []
    frac_rejects = []
    std_frac = []
    mean_acc_no_rejects = []
    std_accs = []

    base_data = data.query(
        f"algo == {algo_idx} and augmenttype == {augmenttype_idx} and datasetkind == {datasetkind_idx} and split == 1"
    )

    low = -1.0 if algo_name == "dst-pll-2024" else 0.0
    for thresh in np.arange(low, 1.01, 0.01):

        frac_rejected = []
        accs = []
        for dataset_name in dataset_names:
            dataset_idx = datasets_to_idx[dataset_name][0]

            # Fraction of reject
            if algo_name == "dst-pll-2024":
                non_rejected = base_data.query(
                    f"dataset == {dataset_idx} and reject > {thresh}")
                rejected = base_data.query(
                    f"dataset == {dataset_idx} and reject <= {thresh}")
            else:
                non_rejected = base_data.query(
                    f"dataset == {dataset_idx} and maxprob >= {thresh}")
                rejected = base_data.query(
                    f"dataset == {dataset_idx} and maxprob < {thresh}")

            if non_rejected.shape[0] <= 10:
                continue
            frac_rejected.append(
                rejected.shape[0] / (non_rejected.shape[0] + rejected.shape[0]))

            # Accuracy of non-rejected
            accs.append(non_rejected.query(
                "correct == 1").shape[0] / non_rejected.shape[0])

        if len(frac_rejected) != len(dataset_names):
            continue

        # Store results
        xs.append(thresh)
        frac_rejects.append(np.mean(frac_rejected))
        std_frac.append(np.std(frac_rejected))
        mean_acc_no_rejects.append(np.mean(accs))
        std_accs.append(np.std(accs))

    return xs, frac_rejects, std_frac, mean_acc_no_rejects, std_accs


# fontsize = 10 pt
fs = 9
linewidth = 5.5206248611  # inches

latex_preamble = r"""
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{microtype}
\usepackage{amsmath}
\renewcommand{\rmdefault}{ptm}
\renewcommand{\sfdefault}{phv}
"""

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": latex_preamble,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "font.size": fs,
    "xtick.labelsize": fs,
    "ytick.labelsize": fs,
    "legend.fontsize": fs,
    "savefig.bbox": "tight",
})

fmts1 = [
    "d", "o", "<", "^", ">", "v", "P", ".", "s", "X",
]
linestyle_fmts = [
    (0, ()), (0, (1, 1)), (5, (10, 3)), (0, (5, 5)), (0, (5, 1)),
    (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5)),
    (0, (5, 3)), (0, (3, 1, 3, 1, 1, 1)),
]
lgd_name = {
    "pl-knn-2005": "\\textsc{Pl-Knn}",
    "pl-svm-2008": "\\textsc{Pl-Svm}",
    "ipal-2015": "\\textsc{Ipal}",
    "pl-ecoc-2017": "\\textsc{Pl-Ecoc}",
    # Deep Learning methods
    "proden-2020": "\\textsc{Proden}",
    "rc-2020": "\\textsc{Rc}",
    "cc-2020": "\\textsc{Cc}",
    "valen-2021": "\\textsc{Valen}",
    "pop-2023": "\\textsc{Pop}",
    # Our method
    "dst-pll-2024": "Our method",
}


def compute_area(xs, ys):
    area = 0.0
    for x1, x2, y1, y2 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        area += 0.5 * (x2 - x1) * (y1 + y2)
    return area


def plot_tradeoff(dataset_name, dataset_kind, augtype, axis, target_num_pts=50):
    plt.style.use("tableau-colorblind10")
    for i, algo_name in enumerate(["dst-pll-2024"] + list(algos_to_idx)[:-1]):
        xs, frac_rejects, std_frac, mean_acc_no_rejects, std_accs = \
            get_reject_trajectory(runall_data, augtype,
                                  dataset_kind, dataset_name, algo_name)

        if algo_name == "dst-pll-2024":
            thresh_val = 0.0
        elif algo_name == "pl-knn-2005":
            thresh_val = 0.3
        else:
            thresh_val = 0.9
        idx_thresh = np.argmin(np.abs(np.array(xs) - thresh_val))

        axis.plot(
            frac_rejects, mean_acc_no_rejects,
            linestyle=linestyle_fmts[i], color=f"C{i}",
            label="_nolegend_",
            zorder=(2.3 if algo_name == "dst-pll-2024" else 2.2),
        )

        axis.plot(
            [frac_rejects[idx_thresh]], [mean_acc_no_rejects[idx_thresh]],
            fmts1[i], linestyle=linestyle_fmts[i], color=f"C{i}",
            label=lgd_name[algo_name], zorder=(
                2.5 if algo_name == "dst-pll-2024" else 2.4),
            markersize=5,
        )


fig, axis = plt.subplots(1, 3, figsize=(linewidth, 1.3))
plt.subplots_adjust(wspace=0.35)

dataset_name = ["ecoli"]
dataset_kind = "uci"
augtype = "instance-dependent"

axis[0].grid(alpha=1.0, color="#e7e7e7")
plot_tradeoff(dataset_name, dataset_kind, augtype, axis[0])
axis[0].set_xlim(0, 1)
axis[0].set_ylim(0.75, 1)
axis[0].set_ylabel("Non-Rejected Test Acc.")
axis[0].set_xlabel("Fraction of Rejects")

dataset_name = ["kmnist"]
dataset_kind = "mnistlike"
augtype = "instance-dependent"

axis[1].grid(alpha=1.0, color="#e7e7e7")
plot_tradeoff(dataset_name, dataset_kind, augtype, axis[1])
axis[1].set_xlim(0, 1)
axis[1].set_ylim(0.9, 1)
axis[1].set_xlabel("Fraction of Rejects")

dataset_name = ["msrc-v2"]
dataset_kind = "rl"
augtype = "rl"

axis[2].grid(alpha=1.0, color="#e7e7e7")
plot_tradeoff(dataset_name, dataset_kind, augtype, axis[2])
axis[2].set_xlim(0, 1)
axis[2].set_ylim(0.5, 1)
axis[2].set_xlabel("Fraction of Rejects")

axis[0].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
axis[1].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
axis[2].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])

axis[0].set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
axis[1].set_yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
axis[2].set_yticks([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])

lgd = plt.legend(bbox_to_anchor=(0.9, 1.56), ncol=5,
                 fontsize=fs, columnspacing=0.9, handletextpad=0.5)
lgd.get_frame().set_alpha(1)
lgd.get_frame().set_edgecolor("#e7e7e7")

plt.savefig("plots/reject.pdf")


group_res = runall_data.query("split == 1").groupby(
    ["augmenttype", "dataset", "datasetkind", "algo", "seed"], as_index=False)["correct"].mean()


def get_reject_trajectory(
    data: pd.DataFrame, augmenttype_name: str, datasetkind_name: str,
    dataset_names: List[str], algo_name: str, seed=None,
):
    algo_idx = algos_to_idx[algo_name]
    augmenttype_idx = aug_type_to_idx[augmenttype_name]
    datasetkind_idx = dataset_kind_to_idx[datasetkind_name]

    xs = []
    frac_rejects = []
    std_frac = []
    mean_acc_no_rejects = []
    std_accs = []

    if seed is None:
        base_data = data.query(
            f"algo == {algo_idx} and augmenttype == {augmenttype_idx} and datasetkind == {datasetkind_idx} and split == 1"
        )
    else:
        base_data = data.query(
            f"algo == {algo_idx} and augmenttype == {augmenttype_idx} and datasetkind == {datasetkind_idx}" +
            f" and split == 1 and seed == {seed}"
        )

    low = -1.0 if algo_name == "dst-pll-2024" else 0.0
    thresh_range = np.arange(low, 1.01, 0.01)
    for thresh in thresh_range:

        frac_rejected = []
        accs = []
        for dataset_name in dataset_names:
            dataset_idx = datasets_to_idx[dataset_name][0]

            # Fraction of reject
            if algo_name == "dst-pll-2024":
                non_rejected = base_data.query(
                    f"dataset == {dataset_idx} and reject > {thresh}")
                rejected = base_data.query(
                    f"dataset == {dataset_idx} and reject <= {thresh}")
            else:
                non_rejected = base_data.query(
                    f"dataset == {dataset_idx} and maxprob >= {thresh}")
                rejected = base_data.query(
                    f"dataset == {dataset_idx} and maxprob < {thresh}")

            if non_rejected.shape[0] <= 10:
                continue
            frac_rejected.append(
                rejected.shape[0] / (non_rejected.shape[0] + rejected.shape[0]))

            # Accuracy of non-rejected
            accs.append(non_rejected.query(
                "correct == 1").shape[0] / non_rejected.shape[0])

        if len(frac_rejected) != len(dataset_names):
            continue

        # Store results
        xs.append(thresh)
        frac_rejects.append(np.mean(frac_rejected))
        std_frac.append(np.std(frac_rejected))
        mean_acc_no_rejects.append(np.mean(accs))
        std_accs.append(np.std(accs))

    return xs, frac_rejects, std_frac, mean_acc_no_rejects, std_accs


def augment_group_df(group_res, all_data):

    res = []
    for _, augtype, dataset, datasetkind, algo, seed, _ in group_res.itertuples():

        augtype_name = idx_to_aug_type[augtype]
        dataset_name = idx_to_datasets[dataset]
        datasetkind_name = idx_to_dataset_kind[datasetkind]
        algo_name = idx_to_algos[algo]

        if algo_name == "dst-pll-2024":
            thresh_val = 0.0
        elif algo_name == "pl-knn-2005":
            thresh_val = 0.5
        else:
            thresh_val = 0.9

        xs, frac_rejects, std_frac, mean_acc_no_rejects, std_accs = \
            get_reject_trajectory(all_data, augtype_name, datasetkind_name, [
                                  dataset_name], algo_name, seed)

        idx_thresh = np.argmin(np.abs(np.array(xs) - thresh_val))

        frac, acc = frac_rejects[idx_thresh], mean_acc_no_rejects[idx_thresh]
        res.append((augtype, dataset, datasetkind, algo, seed, frac, acc))
    return pd.DataFrame(res, columns=["augtype", "dataset", "datasetkind", "algo", "seed", "frac", "acc"])


tradeoff_df = augment_group_df(group_res, runall_data)

tradeoff_risk_df = tradeoff_df.copy()
for lmbd in [0.0, 0.05, 0.1, 0.15, 0.2]:
    tradeoff_risk_df[f"risk_lmbd_{lmbd:.2f}"] = (
        1 - tradeoff_risk_df["acc"]) + lmbd * tradeoff_risk_df["frac"]

group_tradeoff_df = tradeoff_df.groupby(["algo"], as_index=False)[["frac", "acc"]].agg(
    frac_reject_mean=("frac", "mean"), frac_reject_std=("frac", "std"),
    acc_mean=("acc", "mean"), acc_std=("acc", "std"),
)
for lmbd in [0.0, 0.05, 0.1, 0.15, 0.2]:
    group_tradeoff_df[f"risk_mean_lmbd_{lmbd:.2f}"] = (
        1 - group_tradeoff_df["acc_mean"]) + lmbd * group_tradeoff_df["frac_reject_mean"]
    group_tradeoff_df[f"risk_std_lmbd_{lmbd:.2f}"] = np.sqrt(
        group_tradeoff_df["acc_std"] ** 2 + lmbd * group_tradeoff_df["frac_reject_std"] ** 2)

print("Risk trade-off table:")
for t in group_tradeoff_df.loc[:, "risk_mean_lmbd_0.00":].itertuples():
    print(" & ".join(
        map(lambda x: f"{x[0]:.2f} ($\\pm$ {x[1]:.2f})", zip(t[1::2], t[2::2]))))
print()
print()
