"""Summarize raw simulation estimates by sample size.

Reads results/raw/estimates.pk and writes results/summary.csv with, for each n,
the mean and SD of beta hat 2 and of the plug-in estimate, plus the true delta
and each estimator's bias. Only converged fits (status 0) are summarized; the
number of failed fits is reported in n_failed.
"""
import csv
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ESTIMATES_PATH = ROOT / "results" / "raw" / "estimates.pk"
SUMMARY_PATH = ROOT / "results" / "summary.csv"

COLUMNS = ["n", "true_delta", "beta2_mean", "beta2_sd", "beta2_bias",
           "plugin_mean", "plugin_sd", "plugin_bias", "n_fits", "n_failed"]

STATUS_COL = 7


def summarize(estimates):
    rows = []
    for n in np.unique(estimates[:, 0]):
        r_all = estimates[estimates[:, 0] == n]
        r = r_all[r_all[:, STATUS_COL] == 0]
        delta = r_all[0, 6]
        beta2, plug = r[:, 3], r[:, 5]
        rows.append([int(n), delta, beta2.mean(), beta2.std(ddof=1),
                     beta2.mean() - delta, plug.mean(), plug.std(ddof=1),
                     plug.mean() - delta, len(r_all), len(r_all) - len(r)])
    return rows


if __name__ == "__main__":
    with open(ESTIMATES_PATH, "rb") as f:
        rows = summarize(pickle.load(f))
    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print(f"Saved {SUMMARY_PATH}")
