#!/usr/bin/env python3
"""
Staff Evaluation Scores Visuals (matplotlib)
Generates: histogram, grouped boxplot by evaluation type, monthly trend,
and observer averages. Saves figures to ./figures_evals and summary tables
to ./outputs.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Defaults
CSV_PATH = "EVALUATION_SCORES.csv"
DATE_COL = "ObservationDateTime"
SCORE_COL = "ObservationScore"
SCHOOLYEAR_COL = "SchoolYear"
OBSERVER_COL = "ObserverLastFirstName"
OBSERVEE_COL = "ObserveeLastFirstName"  # kept for CLI symmetry (not used directly)
ETYPE_COL = "EvaluationType"

OUT_FIGS = Path("figures_evals")
OUT_DATA = Path("outputs")


def parse_args():
    ap = argparse.ArgumentParser(description="Staff evaluation score charts.")
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--school-year", default=None)
    ap.add_argument("--date-col", default=DATE_COL)
    ap.add_argument("--score-col", default=SCORE_COL)
    ap.add_argument("--schoolyear-col", default=SCHOOLYEAR_COL)
    ap.add_argument("--observer-col", default=OBSERVER_COL)
    ap.add_argument("--observee-col", default=OBSERVEE_COL)
    ap.add_argument("--etype-col", default=ETYPE_COL)
    return ap.parse_args()


def ensure_dirs():
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)


def load_clean(path, date_col, score_col, schoolyear_col, school_year_filter=None):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if date_col not in df.columns or score_col not in df.columns:
        raise ValueError(f"CSV must include {date_col!r} and {score_col!r}.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[date_col, score_col]).copy()
    if school_year_filter and schoolyear_col in df.columns:
        df = df[df[schoolyear_col].astype(str) == str(school_year_filter)].copy()
    df["Month"] = df[date_col].dt.to_period("M").astype(str)
    df["MonthStart"] = pd.to_datetime(df["Month"] + "-01")
    return df


def make_eval_group(df, etype_col):
    """Collapse EvaluationType to compact groups."""
    g = df.copy()
    g["EvalGroup"] = g[etype_col].astype(str).replace({
        r"^Teacher.*Announced.*$": "Teacher (Announced)",
        r"^Teacher.*Unannounced.*$": "Teacher (Unannounced)",
        r"^Teacher.*Domain 4.*$": "Teacher (Domain 4)",
        r"^Counselor.*Announced.*$": "Counselor (Announced)",
        r"^Counselor.*Unannounced.*$": "Counselor (Unannounced)",
        r"^Counselor.*Domain 4.*$": "Counselor (Domain 4)",
        r"^Nurse.*Announced.*$": "Nurse (Announced)",
        r"^Nurse.*Unannounced.*$": "Nurse (Unannounced)",
        r"^Nurse.*Domain 4.*$": "Nurse (Domain 4)",
        r"^Child Study Team.*$": "Child Study Team",
        r"^Related Services.*$": "Related Services",
        r"^Instructional Assistant.*$": "Instructional Assistant",
        r"^Administrators.*$": "Administrator",
        r"^Media.*$": "Media Specialist",
    }, regex=True)
    g["EvalGroup"] = g["EvalGroup"].fillna(g[etype_col].astype(str))
    return g


def plot_hist(df, score_col, tag=""):
    s = df[score_col].dropna()
    mean, median, std, n = s.mean(), s.median(), s.std(), len(s)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts, _, patches = ax.hist(s, bins=20, edgecolor="white", linewidth=0.5)
    cm = plt.cm.viridis
    for c, p in zip(counts, patches):
        plt.setp(p, "facecolor", cm(c / max(counts)))
    ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean:.2f}")
    ax.axvline(median, color="orange", linestyle=":", linewidth=2, label=f"Median = {median:.2f}")
    ax.set_title(f"Distribution of Evaluation Scores {tag}", fontsize=13, weight="bold")
    ax.set_xlabel("Score"); ax.set_ylabel("Count of Observations"); ax.grid(alpha=0.3)
    ax.text(0.98, 0.95, f"n = {n}\nMean = {mean:.2f}\nMedian = {median:.2f}\nSD = {std:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIGS / f"01_hist_scores{tag}.png", dpi=150)
    plt.close(fig)


def plot_box_by_type(df, score_col, etype_col, tag="", min_n=10):
    gdf = make_eval_group(df, etype_col)
    stats = (gdf.groupby("EvalGroup")[score_col]
                .agg(median="median", n="count")
                .reset_index())
    stats = stats[stats["n"] >= min_n].sort_values("median")
    order = stats["EvalGroup"].tolist()
    data = [gdf.loc[gdf["EvalGroup"] == label, score_col].dropna().values for label in order]

    fig = plt.figure(figsize=(12, max(6, int(len(order) * 0.45))))
    ax = plt.gca()
    plt.boxplot(data, vert=False, labels=order, showmeans=True)
    ax.set_title(f"Scores by Evaluation Type (Grouped) {tag}")
    ax.set_xlabel("Score"); ax.set_ylabel("Evaluation Type (Grouped)")
    ax.grid(axis="x", alpha=0.3)

    x_right = ax.get_xlim()[1]
    for y_idx, label in enumerate(order, start=1):
        n = int(stats.loc[stats["EvalGroup"] == label, "n"].iloc[0])
        ax.text(x_right, y_idx, f"  n={n}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_FIGS / f"02_box_by_type{tag}.png", dpi=150)
    plt.close(fig)

    (gdf.groupby("EvalGroup")[score_col]
        .agg(["count", "mean", "median", "std"])
        .loc[order]).to_csv(OUT_DATA / f"type_means_grouped{tag}.csv")


def plot_monthly_trend(df, date_col, score_col, tag=""):
    monthly = (df.groupby(["Month", "MonthStart"], as_index=False)[score_col]
                 .mean()
                 .sort_values("MonthStart"))
    fig = plt.figure(figsize=(11, 5))
    plt.plot(monthly["MonthStart"], monthly[score_col], marker="o", linewidth=2)
    plt.title(f"Average Score by Month {tag}")
    plt.xlabel("Date"); plt.ylabel("Average Score")
    fig.tight_layout()
    fig.savefig(OUT_FIGS / f"03_monthly_trend{tag}.png", dpi=150)
    plt.close(fig)


def plot_observer_bar(df, observer_col, score_col, tag="", min_n=5):
    stats = (df.groupby(observer_col)
               .agg(Mean=(score_col, "mean"),
                    Count=(score_col, "count"),
                    Std=(score_col, "std"))
               .reset_index())
    stats = stats[stats["Count"] >= min_n].sort_values("Mean").reset_index(drop=True)
    overall_mean = df[score_col].mean()
    colors = np.where(stats["Mean"] < overall_mean, "tab:blue", "tab:orange")

    fig = plt.figure(figsize=(12, max(5, int(len(stats) * 0.35))))
    x = np.arange(len(stats))
    plt.bar(x, stats["Mean"], color=colors)
    plt.axhline(overall_mean, color="gray", linestyle="--", linewidth=1.5,
                label=f"District Mean = {overall_mean:.2f}")
    plt.legend()
    plt.title(f"Average Score by Observer (n ≥ {min_n}) {tag}")
    plt.xlabel("Observer"); plt.ylabel("Average Score")
    plt.xticks(x, stats[observer_col], rotation=45, ha="right")
    for i, row in stats.iterrows():
        plt.text(i, row["Mean"], f"{row['Mean']:.2f}\n(n={int(row['Count'])})",
                 ha="center", va="bottom", fontsize=9)
    plt.ylim(min(3.0, stats["Mean"].min() - 0.1),
             min(4.05, max(4.0, stats["Mean"].max() + 0.1)))
    plt.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / f"05_observer_mean_bar{tag}.png", dpi=150)
    plt.close(fig)

    stats.to_csv(OUT_DATA / f"evaluator_stats{tag}.csv", index=False)


def main():
    args = parse_args()
    ensure_dirs()
    d = load_clean(args.csv, args.date_col, args.score_col,
                   args.schoolyear_col, args.school_year)
    tag = f"_{args.school_year.replace('/', '-').replace(' ', '_')}" if args.school_year else ""

    plot_hist(d, args.score_col, tag)
    plot_box_by_type(d, args.score_col, args.etype_col, tag, min_n=10)
    plot_monthly_trend(d, args.date_col, args.score_col, tag)
    plot_observer_bar(d, args.observer_col, args.score_col, tag, min_n=5)

    d.groupby([args.etype_col])[args.score_col].mean().reset_index() \
        .to_csv(OUT_DATA / f"type_means{tag}.csv", index=False)
    (d.groupby(["Month", "MonthStart"])[args.score_col].mean().reset_index()
       .sort_values("MonthStart")
       .drop(columns=["MonthStart"])
       .to_csv(OUT_DATA / f"monthly_means{tag}.csv", index=False))

    print(f"Saved figures in: {OUT_FIGS.resolve()}")
    print(f"Saved summaries in: {OUT_DATA.resolve()}")


if __name__ == "__main__":
    main()
