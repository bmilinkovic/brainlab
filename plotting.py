#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plotting.py
===========

High-level plotting helpers for TVB-AdEx simulations.

What this module gives you
--------------------------
1) `plot_tvb_results(...)`  — legacy-compatible multi-panel plots
   - Accepts the same inputs you previously used (`create_dicts` output).
   - Automatically puts E & I in the same panel (if both requested).
   - Regions may be: None (all), int, list[int], "mean", or "median".
   - Handles Raw / TemporalAverage / Bold monitors (dt inferred).

2) `plot_tvb_results_simple(...)` — convenience wrapper for a single
   (simconfig, seed) pair and a single result (no nested lists).

3) `plot_box(...)` — seaborn-based box plots with significance bars.

Aesthetic defaults
------------------
- Matplotlib style tuned for clear grid + readable labels.
- If plotting many regions, lines are semi-transparent and the mean
  is overlaid in bold.

Dependencies expected in your environment
-----------------------------------------
- numpy, matplotlib, seaborn (optional), pandas (for boxplots)
- Your existing helpers:
    from tvbsim.common import create_dicts, find_file_seed, pvalue_to_asterisks
    from tvbsim.printer import Printer
"""

from __future__ import annotations
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Optional deps for boxplots
try:
    import pandas as pd
    import seaborn as sns
except Exception:
    pd = None
    sns = None

from utils import create_dicts, find_file_seed, pvalue_to_asterisks
from utils import Printer


# ---------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------
def _nice_mpl():
    """Apply a clean, readable Matplotlib style."""
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "semibold",
        "axes.labelweight": "semibold",
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 150,
    })


def _infer_dt_ms(simconfig, monitor: str) -> float:
    """
    Infer the sampling step (ms) from the simconfig for a given monitor name.
    """
    pm = simconfig.general_parameters.parameter_monitor
    if monitor == "Bold":
        return float(pm["parameter_Bold"]["period"])
    elif monitor == "TemporalAverage":
        return float(pm["parameter_TemporalAverage"]["period"])
    # Default to integrator dt for Raw and others
    return float(simconfig.general_parameters.parameter_integrator["dt"])


def _coerce_regions(regions, n_regions: int):
    """
    Normalize 'regions' into an index list and a reducer (or None).
      - None            -> all indices
      - int             -> [that index]
      - list/tuple[int] -> those indices
      - "mean"/"median" -> all indices, with reducer np.mean / np.median
    """
    reducer = None
    if regions is None:
        idx = list(range(n_regions))
    elif isinstance(regions, (int, np.integer)):
        idx = [int(regions)]
    elif isinstance(regions, (list, tuple, np.ndarray)):
        idx = list(map(int, regions))
    elif isinstance(regions, str):
        if regions.lower() == "mean":
            idx = list(range(n_regions))
            reducer = np.mean
        elif regions.lower() == "median":
            idx = list(range(n_regions))
            reducer = np.median
        else:
            raise ValueError("Unsupported regions string. Use 'mean' or 'median'.")
    else:
        raise ValueError("Unsupported 'regions' argument.")

    # Keep indices in bounds
    idx = [i for i in idx if 0 <= i < n_regions]
    if len(idx) == 0:
        raise ValueError("No valid region indices selected.")
    return idx, reducer


def _combine_EI(var_select):
    """Return True if both 'E' and 'I' are in var_select."""
    s = set(var_select)
    return ("E" in s) and ("I" in s)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def plot_tvb_results(
    simconfigs,
    result,
    monitor,
    for_explan,
    var_select,
    seeds=[10],
    figsize=(9, 5),
    begin_time=None,
    save=False,
    color_regions=None,
    label_regions=None,
    priority_regions=None,
    end_time=None,
    with_legend_title=True,
    save_path=None,
    regions=None,
    with_title=True,
    extension="png",
):
    """
    Legacy-compatible plotter for TVB results.

    Parameters
    ----------
    simconfigs : list[SimConfig]
        Simulation configurations (one per row-group).
    result : list[list[any]]
        Nested results: result[i][j] corresponds to simconfigs[i], seeds[j].
        (This is exactly how you used it before with [[result]].)
    monitor : str
        One of 'Raw', 'TemporalAverage', 'Bold'.
    for_explan : tuple
        The second item returned by your legacy `get_result` (meta).
        Kept for signature compatibility; not used here.
    var_select : list[str]
        Variables to plot (e.g., ['E','I'] or ['E']).
    seeds : list[int]
        Seeds corresponding to second dimension in `result`.
    figsize : (float, float)
        Figure size for each row of subplots.
    begin_time : float or None
        Start time (ms). Defaults to simconfig.cut_transient.
    save : int
        0 show only, 1 save+show, 2 save only (no show).
    color_regions : list or None
        Optional colors per region index (used only when plotting many regions).
    label_regions : list or None
        Optional labels per region index for legend entries.
    priority_regions : list[int] or None
        Regions to draw last (on top).
    end_time : float or None
        End time (ms). Defaults to simconfig.run_sim or available data length.
    with_legend_title : bool
        If True, add axes labels and legends.
    save_path : str or None
        Directory to save images into (created if necessary).
    regions : None | int | list[int] | "mean" | "median"
        Region selection:
          - None: all
          - int: one region
          - list[int]: specific regions
          - "mean"/"median": aggregate across all regions (bold line)
    with_title : bool
        Whether to put the simconfig title on axes.
    extension : str
        File extension for saved figures.

    Notes
    -----
    - This function expects each `result[i][j]` to be the *return* of your
      legacy `get_result`'s "result" element, and uses your `create_dicts`
      to turn it into a dict: {'E': (T, N), 'I': (T, N), ...}.
    """
    _nice_mpl()
    priority_regions = priority_regions or []

    # Grid dimensions
    n_rows = len(simconfigs) * max(1, len(seeds))
    n_cols = len(var_select) - 1 if _combine_EI(var_select) else len(var_select)
    n_cols = max(1, n_cols)

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows), squeeze=False)

    # Loop over simconfigs and seeds to populate each row
    row_idx = 0
    for i, cfg in enumerate(simconfigs):
        dt_ms = _infer_dt_ms(cfg, monitor)
        for j, seed in enumerate(seeds):
            # Build variables dictionary from legacy structure
            res_ij = result[i][j]
            var_dict = create_dicts(cfg, res_ij, monitor, for_explan, var_select, seed=seed)
            # Sanity: expected shape (T, N) for each var
            any_var = next(iter(var_dict.values()))
            T, N = any_var.shape

            # Time vector in ms (then shown in seconds)
            t_ms = np.arange(T, dtype=float) * dt_ms
            _begin = float(cfg.cut_transient if begin_time is None else begin_time)
            _end   = float(cfg.run_sim      if end_time   is None else end_time)
            _end   = min(_end, t_ms[-1] if T > 0 else _end)

            # Clip to time window
            mask = (t_ms >= _begin) & (t_ms <= _end)
            if not np.any(mask):
                Printer.print(f"No samples in window for cfg row {i}, seed {seed}.", level=1)
                row_idx += 1
                continue
            t = (t_ms[mask] / 1000.0)  # seconds

            # Region selection / aggregation
            idx, reducer = _coerce_regions(regions, N)  # may be all
            # Panel bookkeeping
            if _combine_EI(var_select):
                # Put E & I together in the first column
                ax_ei = axes[row_idx, 0]
                _plot_var_lines(ax_ei, t, var_dict, "E", idx, reducer, color_regions, label_regions,
                                title=cfg.get_plot_title() if with_title else None,
                                ylabel="Firing rate (Hz)" if with_legend_title else None)
                _plot_var_lines(ax_ei, t, var_dict, "I", idx, reducer, color_regions, label_regions,
                                title=None, ylabel=None)
                if with_legend_title:
                    ax_ei.set_xlabel("Time (s)")
                    ax_ei.legend(loc="upper right", fontsize=8)
            else:
                # Separate columns for each var
                for c, var in enumerate(var_select):
                    ax = axes[row_idx, c]
                    _plot_var_lines(ax, t, var_dict, var, idx, reducer, color_regions, label_regions,
                                    title=cfg.get_plot_title() if (with_title and c == 0) else None,
                                    ylabel="Firing rate (Hz)" if (with_legend_title and c == 0) else None)
                    if with_legend_title:
                        ax.set_xlabel("Time (s)")
                        if c == 0:
                            ax.legend(loc="upper right", fontsize=8)

            row_idx += 1

    plt.tight_layout()

    # Save/show
    if save:
        basename = f"{monitor}_{'_'.join(var_select)}"
        out_name = f"{basename}.{extension}"
        out_path = os.path.join(save_path or ".", out_name)
        plt.savefig(out_path, bbox_inches="tight")
        if save == 1:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()


def _plot_var_lines(
    ax,
    t_s: np.ndarray,
    var_dict: dict[str, np.ndarray],
    varname: str,
    region_idx: list[int],
    reducer,
    color_regions,
    label_regions,
    title=None,
    ylabel=None,
):
    """
    Draw one variable's timeseries on `ax`.

    var_dict[varname] shape is (T, N). `region_idx` selects columns to plot.
    If `reducer` is not None (e.g. np.mean), draw a single bold aggregate line.
    Otherwise draw one line per region (with alpha) + overlay their mean in bold.
    """
    if varname not in var_dict:
        return

    Y = var_dict[varname]  # (T, N)
    Y = np.asarray(Y)
    Y = Y[:, region_idx]

    if reducer is not None:
        y = reducer(Y, axis=1)
        ax.plot(t_s, y, lw=2.2, label=f"{varname} ({reducer.__name__})")
    else:
        # Many lines, semi-transparent
        n_r = Y.shape[1]
        for k in range(n_r):
            color = None if (color_regions is None) else color_regions[region_idx[k] % len(color_regions)]
            label = None
            if label_regions is not None:
                label = label_regions[region_idx[k]]
            ax.plot(t_s, Y[:, k], lw=0.9, alpha=0.35, label=label, color=color)
        # Overlay mean
        ax.plot(t_s, np.mean(Y, axis=1), lw=2.2, label=f"{varname} (mean)")

    if title:
        ax.set_title(title, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Sensible y-lims for rates (Hz)
    lo = np.nanpercentile(Y, 1)
    hi = np.nanpercentile(Y, 99)
    pad = max(1.0, 0.08 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)


# ---------------------------------------------------------------------
# Convenience: simple (single) plotter
# ---------------------------------------------------------------------
def plot_tvb_results_simple(
    simconfig,
    result_single,
    for_explan,
    vars_to_plot=("E", "I"),
    monitor="Raw",
    regions=0,
    begin_time=None,
    end_time=None,
    figsize=(8, 3.5),
    title_prefix=None,
    save=0,
    save_path=None,
    extension="png",
):
    """
    Convenience wrapper for a single (simconfig, result) pair.

    Parameters are analogous to `plot_tvb_results`, but `result_single`
    is a single legacy result (not nested) for one seed/config.
    """
    _nice_mpl()
    dt_ms = _infer_dt_ms(simconfig, monitor)
    var_dict = create_dicts(simconfig, result_single, monitor, for_explan, list(vars_to_plot), seed=None)

    # Infer shapes and time window
    any_var = next(iter(var_dict.values()))
    T, N = any_var.shape
    t_ms = np.arange(T, dtype=float) * dt_ms
    _begin = float(simconfig.cut_transient if begin_time is None else begin_time)
    _end   = float(simconfig.run_sim      if end_time   is None else end_time)
    _end   = min(_end, t_ms[-1] if T > 0 else _end)
    mask = (t_ms >= _begin) & (t_ms <= _end)
    t = (t_ms[mask] / 1000.0)

    idx, reducer = _coerce_regions(regions, N)

    n_cols = len(vars_to_plot) - 1 if _combine_EI(vars_to_plot) else len(vars_to_plot)
    n_cols = max(1, n_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(figsize[0] * n_cols, figsize[1]))
    axes = np.atleast_2d(axes)

    if _combine_EI(vars_to_plot):
        ax = axes[0, 0]
        _plot_var_lines(ax, t, {k: v[mask] for k, v in var_dict.items()}, "E", idx, reducer, None, None,
                        title=(title_prefix or simconfig.get_plot_title()), ylabel="Firing rate (Hz)")
        _plot_var_lines(ax, t, {k: v[mask] for k, v in var_dict.items()}, "I", idx, reducer, None, None)
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", fontsize=8)
    else:
        for c, var in enumerate(vars_to_plot):
            ax = axes[0, c]
            _plot_var_lines(ax, t, {k: v[mask] for k, v in var_dict.items()}, var, idx, reducer, None, None,
                            title=(title_prefix if c == 0 else None), ylabel=("Firing rate (Hz)" if c == 0 else None))
            ax.set_xlabel("Time (s)")
            if c == 0:
                ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if save:
        os.makedirs(save_path or ".", exist_ok=True)
        base = title_prefix or simconfig.get_plot_title()
        out = os.path.join(save_path or ".", f"{base}_{monitor}_{'-'.join(vars_to_plot)}.{extension}")
        plt.savefig(out, bbox_inches="tight")
        if save == 1:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------
# Box plot with significance bars
# ---------------------------------------------------------------------
def plot_box(
    simconfigs,
    file_prefix,
    n_seeds=None,
    data=None,
    save=0,
    path=None,
    file_name=None,
    x=None,
    save_path=None,
    boxplot_params=None,
    set_params=None,
    box_colors=None,
    figsize=(8, 5.5),
):
    """
    Seaborn boxplots with group comparisons and significance bars.

    Parameters
    ----------
    simconfigs : list[SimConfig]
        Which configurations to collect/label (if `data` is None).
    file_prefix : str
        Prefix used by your pipeline when saving per-seed metrics
        (used together with `find_file_seed`). Ignored if `data` provided.
    n_seeds : int or None
        Number of seeds to collect per simconfig (if `data` is None).
    data : np.ndarray or None
        If provided, must be shape (n_groups, n_seeds). Skips file loading.
    save : int
        0 show, 1 save+show, 2 save only.
    path : str or None
        Base folder for file discovery (if `data` is None).
    file_name : str or None
        Output filename when saving.
    x : list[str] or None
        Custom x-axis labels (defaults to simconfig.get_plot_title()).
    save_path : str or None
        Folder to save plot into.
    boxplot_params : dict
        Extra kwargs passed to seaborn.boxplot.
    set_params : dict
        Axes `.set(**set_params)` is applied (e.g. labels, title).
    box_colors : list or None
        Palette argument for seaborn.
    figsize : (float, float)
        Figure size.

    Returns
    -------
    None
    """
    if pd is None or sns is None:
        raise RuntimeError("plot_box requires pandas and seaborn to be installed.")

    boxplot_params = boxplot_params or {}
    set_params = set_params or {}

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Collect/prepare data
    if data is None:
        if n_seeds is None:
            raise ValueError("When data=None you must pass n_seeds.")
        data = np.zeros((len(simconfigs), n_seeds))
        for i, simconfig in enumerate(simconfigs):
            file_path = find_file_seed(simconfig, path, file_prefix, n_minimal_seeds=n_seeds)
            data[i] = np.load(file_path)[:n_seeds]
    else:
        n_seeds = data.shape[1]

    labels = x if x is not None else [cfg.get_plot_title() for cfg in simconfigs]

    # Long-form DataFrame
    rows = []
    for i, label in enumerate(labels):
        for v in data[i, :]:
            rows.append({"Group": label, "Value": float(v)})
    df = pd.DataFrame(rows)

    _nice_mpl()
    fig, ax = plt.subplots(figsize=figsize)
    palette = box_colors if box_colors else sns.color_palette()

    sns.boxplot(data=df, x="Group", y="Value", palette=palette, ax=ax, **boxplot_params)
    sns.stripplot(data=df, x="Group", y="Value", ax=ax, color="k", size=3, alpha=0.35, jitter=0.15)

    # Mean line per group
    means = df.groupby("Group")["Value"].mean()
    for pos, mean_val in enumerate(means):
        ax.hlines(mean_val, pos - 0.35, pos + 0.35, colors="red", linewidth=2, zorder=10)

    # Significance bars (pairwise t-tests)
    unique_groups = list(df["Group"].unique())
    grouped_values = [df[df["Group"] == g]["Value"].values for g in unique_groups]

    y_min, y_max = float(df["Value"].min()), float(df["Value"].max())
    h = max(1e-6, (y_max - y_min) * 0.06)
    cur_y = y_max + h

    for i, j in itertools.combinations(range(len(unique_groups)), 2):
        from scipy.stats import ttest_ind
        stat, pval = ttest_ind(grouped_values[i], grouped_values[j], equal_var=False)
        stars = pvalue_to_asterisks(pval)
        x1, x2 = i, j
        ax.plot([x1, x1, x2, x2], [cur_y, cur_y + h, cur_y + h, cur_y], lw=1.5, c="k")
        ax.text((x1 + x2) * 0.5, cur_y + h, stars, ha="center", va="bottom", fontsize=11)
        cur_y += h * 1.6

    ax.set(**set_params)
    ax.set_xlabel(ax.get_xlabel(), fontweight="semibold")
    ax.set_ylabel(ax.get_ylabel(), fontweight="semibold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save:
        fname = file_name or "boxplot.png"
        out_path = os.path.join(save_path or ".", fname)
        plt.savefig(out_path, bbox_inches="tight")
        if save == 1:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()
