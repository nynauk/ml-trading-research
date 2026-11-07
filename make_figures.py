"""Draw the README figures from results/experiments.json and results/curves/ (run ml_trading.py first).

Each figure is written as results/figures/<name>_light.png and <name>_dark.png for GitHub's two themes, using
the same validated palette as the cpp-backtesting-engine figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
PX = 0.75  # one CSS pixel in points

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "neg": "#e34948",
              "mid": "#f0efec", "pos": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70", "#c98500"], "neg": "#e66767",
             "mid": "#383835", "pos": "#3987e5"},
}


def style(t: dict) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "axes.edgecolor": t["axis"], "axes.linewidth": PX,
        "axes.labelcolor": t["muted"], "axes.labelsize": 9, "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "xtick.major.width": PX, "ytick.major.width": PX,
        "xtick.major.size": 3, "ytick.major.size": 0, "text.color": t["ink"], "legend.frameon": False,
        "legend.fontsize": 8.5, "legend.labelcolor": t["ink2"],
    })


def frame(ax, t: dict, grid_axis: str = "y") -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=PX, linestyle="-")
    ax.set_axisbelow(True)


def titles(fig, t: dict, title: str, subtitle: str, x: float = 0.07) -> None:
    fig.text(x, 0.965, title, fontsize=12.5, fontweight="bold", color=t["ink"], va="top")
    fig.text(x, 0.9, subtitle, fontsize=9, color=t["ink2"], va="top", linespacing=1.4)


def luminance(hex_colour: str) -> float:
    def channel(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(c) for c in to_rgb(hex_colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def auc_figure(r: dict, t: dict, mode: str) -> None:
    rows = r["A_models"]
    sel = r["selection"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=200)
    fig.subplots_adjust(left=0.27, right=0.95, top=0.7, bottom=0.13)
    feature_sets = list(dict.fromkeys(row["feature_set"] for row in rows))
    ticks, tick_labels = [], []
    for g, fs in enumerate(feature_sets):  # one group per feature set, the two models side by side in it
        base = (len(feature_sets) - 1 - g) * 3.0
        ticks.append(base + 0.5)
        tick_labels.append(fs.replace(" + Asia range", "\n+ SwiftPip Asia range"))
        for k, model in enumerate(["logistic", "gradient boosting"]):
            row = next(x for x in rows if x["feature_set"] == fs and x["model"] == model)
            v, y, colour = row["validation"], base + 1 - k, t["series"][k]
            ax.plot(v["auc_95"], [y, y], color=colour, lw=2 * PX, solid_capstyle="round")
            ax.plot([v["auc"]], [y], "o", ms=8 * PX, color=colour, mec=t["surface"], mew=2 * PX, zorder=3)
            chosen = fs == sel["feature_set"] and model == sel["model"]
            ax.annotate(f"{v['auc']:.3f}" + ("  selected" if chosen else ""), (v["auc_95"][1], y), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=t["ink"],
                        fontweight="bold" if chosen else "normal")
    ax.axvline(0.5, color=t["axis"], lw=PX)
    ax.set_yticks(ticks, tick_labels, color=t["ink2"], fontsize=8.5)
    ax.set_ylim(-0.8, len(feature_sets) * 3 - 1.2)
    ax.set_xlim(0.495, max(row["validation"]["auc_95"][1] for row in rows) + 0.012)
    frame(ax, t, grid_axis="x")
    ax.set_xlabel("AUC on validation, 2015-2019 (0.5 = no skill), with 95% interval from resampled days")
    handles = [plt.Line2D([], [], color=t["series"][k], marker="o", lw=2 * PX, ms=6 * PX) for k in (0, 1)]
    fig.legend(handles, ["logistic regression", "gradient boosting"], loc="upper left", bbox_to_anchor=(0.265, 0.79),
               ncol=2)
    titles(fig, t, "A small, real signal in the next hour's direction",
           f"Each model was fitted on 2004-2014. Predicting a coin flip has log loss 0.6931; the selected model reaches "
           f"{sel['validation_log_loss']:.4f}.\nRule fixed in advance: lowest validation log loss.", x=0.02)
    fig.savefig(FIGURES / f"model_auc_{mode}.png")
    plt.close(fig)


def walk_forward_figure(r: dict, t: dict, mode: str) -> None:
    wf = pd.DataFrame(r["walk_forward"])
    fig, ax = plt.subplots(figsize=(8.6, 4.0), dpi=200)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.72, bottom=0.1)
    for k, model in enumerate(["logistic", "gradient boosting"]):
        d = wf[wf.model == model]
        ax.plot(d.year, d.auc, color=t["series"][k], lw=2 * PX, marker="o", ms=6 * PX, mec=t["surface"], mew=2 * PX,
                label="logistic regression" if model == "logistic" else "gradient boosting")
        ax.annotate(f"{d.auc.iloc[-1]:.3f}", (d.year.iloc[-1], d.auc.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=t["ink"])
    ax.axhline(0.5, color=t["axis"], lw=PX)
    ax.set_xticks(range(int(wf.year.min()), int(wf.year.max()) + 1, 2))
    frame(ax, t)
    ax.set_ylabel("AUC for the year")
    ax.margins(x=0.02)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.075, 0.81), ncol=2)
    above = (wf.groupby("year").auc.min() > 0.5).sum()
    titles(fig, t, "The signal held in every year, retrained annually",
           f"Each year is predicted by models refitted on all earlier years (expanding window).\nBoth models scored "
           f"above 0.5 in {above} of {wf.year.nunique()} years.", x=0.075)
    fig.savefig(FIGURES / f"walk_forward_auc_{mode}.png")
    plt.close(fig)


def threshold_figure(r: dict, t: dict, mode: str) -> None:
    grid = pd.DataFrame(r["D_validation_grid"])
    costs = list(dict.fromkeys(grid.cost))
    thresholds = sorted(grid.threshold.unique())
    sel = r["selection"]
    vals = grid.pivot(index="cost", columns="threshold", values="sharpe_pnl_zero_rf").reindex(costs)
    m = float(np.nanmax(np.abs(vals.to_numpy())))
    cmap = LinearSegmentedColormap.from_list("div", [t["neg"], t["mid"], t["pos"]])
    norm = TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)
    fig, ax = plt.subplots(figsize=(10.6, 3.6), dpi=200)
    fig.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=0.17)
    gap = 0.04
    for i, cost in enumerate(costs):
        for j, thr in enumerate(thresholds):
            v = vals.loc[cost, thr]
            y = len(costs) - 1 - i
            if np.isnan(v):
                ax.text(j + 0.5, y + 0.5, "flat", ha="center", va="center", fontsize=7, color=t["muted"])
                continue
            colour = matplotlib.colors.to_hex(cmap(norm(v)))
            ax.add_patch(Rectangle((j + gap / 2, y + gap / 2), 1 - gap, 1 - gap, color=colour, lw=0))
            text = f"{v:+.1f}" if round(v, 1) != 0 else "0.0"
            ax.text(j + 0.5, y + 0.5, text, ha="center", va="center", fontsize=7,
                    color="#ffffff" if luminance(colour) < 0.3 else "#0b0b0b")
            if cost == r["setup"]["baseline_cost"] and thr == sel["threshold"]:
                ax.add_patch(Rectangle((j + 0.03, y + 0.03), 0.94, 0.94, fill=False, ec=t["ink"], lw=2 * PX))
    ax.set_xlim(0, len(thresholds))
    ax.set_ylim(0, len(costs))
    ax.set_xticks(np.arange(len(thresholds)) + 0.5, [f"{x:.3f}" for x in thresholds], rotation=0, fontsize=7.5)
    ax.set_yticks(np.arange(len(costs)) + 0.5, costs[::-1], color=t["ink2"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Threshold: trade only when the predicted probability is this far from 0.5")
    cax = fig.add_axes([0.915, 0.17, 0.01, 0.63])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8)
    cb.set_label("Sharpe ratio", color=t["muted"], fontsize=8.5)
    best_gross = grid[grid.cost == "no costs"].sharpe_pnl_zero_rf.max()
    titles(fig, t, "Costs erase the edge at every threshold",
           f"Validation 2015-2019, selected model. Before costs the best Sharpe is {best_gross:+.1f}; outlined: the "
           f"threshold chosen at the baseline cost ({sel['threshold']:.3f}, Sharpe {sel['validation_sharpe']:+.2f}).", x=0.02)
    fig.savefig(FIGURES / f"threshold_cost_{mode}.png")
    plt.close(fig)


def test_figure(r: dict, t: dict, mode: str) -> None:
    names = ["test_selected", "test_momentum", "test_ma", "test_buyhold"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.72, bottom=0.09)
    ends = []
    for row, name, colour in zip(r["C_test"], names, t["series"]):
        c = pd.read_csv(RESULTS / "curves" / f"{name}.csv", parse_dates=["timestamp"])
        s = c.set_index("timestamp")["equity"] - 100_000
        label = "ML policy (selected on validation)" if name == "test_selected" else row["strategy"]
        ax.plot(s.index, s.values, color=colour, lw=2 * PX, solid_joinstyle="round", label=label)
        ends.append((s.iloc[-1], s.index[-1], colour))
    ax.axhline(0, color=t["axis"], lw=PX)
    frame(ax, t)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:+,.0f}K" if v else "0"))
    ax.set_ylabel("Cumulative P&L, $")
    ax.margins(x=0)
    lo, hi = ax.get_ylim()
    gap, placed = (hi - lo) * 0.055, []
    for value, x, colour in sorted(ends, key=lambda e: e[0]):
        y = value if not placed else max(value, placed[-1] + gap)
        placed.append(y)
        xn = mdates.date2num(x)
        ax.plot([xn], [value], "o", ms=8 * PX, color=colour, mec=t["surface"], mew=2 * PX, zorder=5, clip_on=False)
        if abs(y - value) > 1e-9:
            ax.plot([xn, ax.get_xlim()[1]], [value, y], color=t["axis"], lw=PX, clip_on=False)
        ax.annotate(f"{value / 1000:+,.1f}K", (ax.get_xlim()[1], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=t["ink"], annotation_clip=False)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.075, 0.8), ncol=2)
    sel = r["C_test"][0]
    titles(fig, t, "One run on the held-out test period, 2020-2024",
           f"0.5 pip + $3.50/side commission on every strategy. The ML policy made {sel['round_trips_per_year']:.0f} round "
           f"trips a year (Sharpe {sel['sharpe_pnl_zero_rf']:+.2f}).", x=0.075)
    fig.savefig(FIGURES / f"test_period_{mode}.png")
    plt.close(fig)


def volatility_figure(r: dict, t: dict, mode: str) -> None:
    wf = pd.DataFrame(r["walk_forward"])
    gb = wf[wf.model == "gradient boosting"].copy()
    gb["net"] = gb.policy.apply(lambda p: p["net_pnl"])
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.74, bottom=0.12)
    ax.axhline(0, color=t["axis"], lw=PX)
    ax.scatter(gb.average_hourly_move_pips, gb.net, s=(10 * PX) ** 2, color=t["series"][1], edgecolor=t["surface"],
               linewidth=2 * PX, zorder=3)
    quietest = gb.average_hourly_move_pips.idxmin()
    for i, row in gb.iterrows():  # label the volatile years and the quietest one; the cluster speaks for itself
        if row.average_hourly_move_pips > 10 or i == quietest:
            ax.annotate(str(int(row.year)), (row.average_hourly_move_pips, row.net), xytext=(6, 4),
                        textcoords="offset points", fontsize=8, color=t["ink2"])
    ax.annotate(f"{int((gb.average_hourly_move_pips <= 10).sum())} calmer years", (8.2, -2600), fontsize=8,
                color=t["muted"], ha="center", va="top")
    frame(ax, t)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:+,.0f}K" if v else "0"))
    ax.set_xlabel("Average one-hour move that year, pips")
    ax.set_ylabel("Net P&L that year, $")
    corr = np.corrcoef(gb.average_hourly_move_pips, gb.net)[0, 1]
    early = gb[gb.year <= 2010].net.sum()
    late = gb[gb.year > 2010].net.sum()
    titles(fig, t, "After costs, the edge only paid when markets moved a lot",
           f"Walk-forward gradient boosting, retrained each year, 0.5 pip + commission. Correlation with volatility "
           f"{corr:+.2f}.\n2008-2010 made {early / 1000:+,.1f}K; the {int(gb.year.max()) - 2010} years since made "
           f"{late / 1000:+,.1f}K in total.", x=0.02)
    fig.savefig(FIGURES / f"volatility_{mode}.png")
    plt.close(fig)


def main() -> None:
    r = json.loads((RESULTS / "experiments.json").read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        style(t)
        auc_figure(r, t, mode)
        walk_forward_figure(r, t, mode)
        threshold_figure(r, t, mode)
        test_figure(r, t, mode)
        volatility_figure(r, t, mode)
    print("wrote figures to", FIGURES)


if __name__ == "__main__":
    main()
