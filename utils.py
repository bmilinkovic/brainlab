#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py
========
Lightweight utility functions and classes for printing and configuration comparison.

This module provides:
    - A configurable Printer for controlling console verbosity.
    - A print_dict_differences() function for comparing parameter dictionaries.
"""

import os
import re
import ast
from pathlib import Path
import numpy as np

from scipy.stats import zscore


from parameters import Parameters  # optional, only for typing/clarity



# ---------------------------------------------------------------------
# Global print verbosity level
# ---------------------------------------------------------------------
PRINT_LEVEL = 0
"""
Global print level for controlling verbosity across the codebase.

Levels
------
0 : print everything (default)
1 : only warnings and exceptions
2 : only exceptions
3 : silent mode
"""


# ---------------------------------------------------------------------
# Printer class
# ---------------------------------------------------------------------
class Printer:
    """
    Unified print handler with adjustable verbosity.

    Use this class instead of the built-in print() to control
    message output according to global PRINT_LEVEL.

    Example
    -------
    >>> from utils import Printer
    >>> Printer.print("Running simulation...", level=0)
    >>> Printer.print("Warning: high variance", level=1)
    """

    @staticmethod
    def print(*args, level=0):
        """Print message if its priority >= PRINT_LEVEL."""
        if level >= PRINT_LEVEL:
            print(*args)


# ---------------------------------------------------------------------
# Dictionary comparison helper
# ---------------------------------------------------------------------
def print_dict_differences(dict1, dict2):
    """
    Compare two dictionaries and print any differences between them.

    Parameters
    ----------
    dict1 : dict
        New or modified dictionary.
    dict2 : dict
        Reference (preexisting) dictionary.

    Returns
    -------
    bool
        True if no differences were found, False otherwise.

    Example
    -------
    >>> from utils import print_dict_differences
    >>> a = {"a": 1, "b": 2}
    >>> b = {"a": 1, "b": 3, "c": 4}
    >>> print_dict_differences(a, b)
    Difference in key 'b':
       New dictionary value: 2
       Preexisting dictionary value: 3
    Key 'c' not found in new dictionary
    """
    no_diff = True
    for key in dict1:
        if key in dict2:
            if dict1[key] != dict2[key]:
                print(f"Difference in key '{key}':")
                print(f"   New dictionary value: {dict1[key]}")
                print(f"   Preexisting dictionary value: {dict2[key]}")
                no_diff = False
        else:
            print(f"Key '{key}' not found in preexisting dictionary")
            no_diff = False

    for key in dict2:
        if key not in dict1:
            print(f"Key '{key}' not found in new dictionary")
            no_diff = False

    return no_diff


# ---------------------------------------------------------------------
# Result loader
# ---------------------------------------------------------------------

def get_result(simconfig, additional_path_folder='', time_begin=None, time_end=None,
               seed=10, vars_int=('E','I','W_e'), simu_path=None):
    """
    Load simulation results between given times and return legacy-shaped arrays.

    This version is fully backward-compatible with your old pipeline and also
    supports the new consolidated artifact `result_raw.npz` when present.

    Parameters
    ----------
    simconfig : SimConfig
        Simulation configuration used to derive the output folder and metadata.
    additional_path_folder : str
        Optional subfolder inside the run directory (kept for compatibility).
    time_begin : float or None
        Start time in ms (defaults to simconfig.cut_transient).
    time_end : float or None
        End time in ms (defaults to simconfig.run_sim).
    seed : int
        Seed for locating the run folder name.
    vars_int : iterable[str]
        Variables to extract. Supported canonical names:
          'E','I','C_ee','C_ei','C_ii','W_e','W_i','noise'
        (If a requested var is not present, it is returned as NaNs.)
    simu_path : str or None
        Override for the root folder (useful when runs were moved).

    Returns
    -------
    result : list[np.ndarray]
        One entry per active monitor in the run, each shaped (len(vars_int), T, n_nodes).
    for_explan : tuple
        Metadata: (parameter_monitor, list(vars_int), shape) where shape == (T, n_nodes)
        for the first monitor.

    Notes
    -----
    - Prefers `result_raw.npz` (single-file) if found. Falls back to legacy `step_*.npy`.
    - E/I (and 'noise') are returned in Hz; others are unit-preserving (as saved).
    - Disconnected regions are NOT dropped here; that filtering is typically handled
      downstream by `create_dicts`.
    """
    # ---- defaults for slicing window ----
    if time_begin is None:
        time_begin = simconfig.cut_transient
    if time_end is None:
        time_end = simconfig.run_sim

    # ---- resolve path ----
    folder_root, sim_name = simconfig.get_sim_name(seed)
    if simu_path is not None:
        folder_root = simu_path
    run_dir = os.path.join(folder_root, sim_name, additional_path_folder)
    Printer.print(f"Loading: {run_dir}")

    # ---- metadata from config ----
    p_sim = simconfig.general_parameters.parameter_simulation
    p_mon = simconfig.general_parameters.parameter_monitor
    dt = float(simconfig.general_parameters.parameter_integrator["dt"])
    save_time = float(p_sim.get("save_time", 1000.0))

    # canonical index map for Raw/TA ordering (as in your setup)
    dict_var_int = {'E':0, 'I':1, 'C_ee':2, 'C_ei':3, 'C_ii':4, 'W_e':5, 'W_i':6, 'noise':7}
    vars_int = list(vars_int)  # ensure list for stable ordering

    # ---- helper: build an (len(vars_int), T, N) block from E/I arrays (and NaN for others) ----
    def _pack_vars_from_arrays(E, I, other_source=None):
        """
        E, I: (T, N) in Hz
        other_source: dict name->(T,N) if available (e.g., from legacy)
        """
        T, N = (E.shape[0], E.shape[1]) if (isinstance(E, np.ndarray) and E.size) else (I.shape[0], I.shape[1])
        out = np.full((len(vars_int), T, N), np.nan, dtype=float)
        for idx, name in enumerate(vars_int):
            if name == 'E' and isinstance(E, np.ndarray) and E.size:
                out[idx] = E
            elif name == 'I' and isinstance(I, np.ndarray) and I.size:
                out[idx] = I
            elif other_source is not None and name in other_source and isinstance(other_source[name], np.ndarray):
                # already correct units
                arr = other_source[name]
                # ensure (T,N)
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    arr = arr[:, None]
                out[idx] = arr
            # else remains NaN
        return out

    # ---- prefer new single-file artifact if present ----
    npz_path = os.path.join(run_dir, "result_raw.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        t_ms = np.asarray(data.get("t_ms", np.array([]))).reshape(-1)
        E = np.asarray(data.get("E", np.array([])))
        I = np.asarray(data.get("I", np.array([])))
        # TA optional; we expose it as a second monitor if available
        TA_E = np.asarray(data.get("TA_E", np.array([])))
        TA_I = np.asarray(data.get("TA_I", np.array([])))

        # slice by time window
        if t_ms.size == 0:
            # nothing recorded
            mask = slice(0, 0)
        else:
            mask = (t_ms >= time_begin) & (t_ms <= time_end)

        def _safe_slice(X):
            X = np.asarray(X)
            if X.size == 0:
                return X
            return X[mask, :]

        t_sel = t_ms[mask] if t_ms.size else np.array([])
        E_sel = _safe_slice(E)
        I_sel = _safe_slice(I)
        # Result list will mirror the “active monitors” idea:
        result_blocks = []

        # 1) Raw-like monitor (always first if Raw was enabled; else still provide it if E/I exist)
        if p_mon.get("Raw", True) or (E_sel.size and I_sel.size):
            # E_sel and I_sel are already in Hz in the NPZ
            raw_block = _pack_vars_from_arrays(E_sel, I_sel, other_source=None)
            result_blocks.append(raw_block)

        # 2) TemporalAverage-like (if present in NPZ and enabled)
        if p_mon.get("TemporalAverage", False) and TA_E.size and TA_I.size:
            # TemporalAverage NPZ is already in Hz
            TA_E_sel = _safe_slice(TA_E)
            TA_I_sel = _safe_slice(TA_I)
            ta_block = _pack_vars_from_arrays(TA_E_sel, TA_I_sel, other_source=None)
            result_blocks.append(ta_block)

        # 3) Bold / Ca not assembled from NPZ here; keep compatibility by returning NaNs if requested
        #    (Your plotting functions typically use 'Raw' or 'TemporalAverage' for E/I.)
        if p_mon.get("Bold", False):
            # construct an empty placeholder with correct shape if someone requests BOLD variables
            T = t_sel.shape[0]
            N = E_sel.shape[1] if (isinstance(E_sel, np.ndarray) and E_sel.ndim == 2) else (I_sel.shape[1] if (isinstance(I_sel, np.ndarray) and I_sel.ndim == 2) else 0)
            bold_block = np.full((len(vars_int), T, N), np.nan, dtype=float)
            result_blocks.append(bold_block)
        if p_mon.get("Ca", False) or p_mon.get("Afferent_coupling", False):
            # same placeholder strategy
            T = t_sel.shape[0]
            N = E_sel.shape[1] if (isinstance(E_sel, np.ndarray) and E_sel.ndim == 2) else (I_sel.shape[1] if (isinstance(I_sel, np.ndarray) and I_sel.ndim == 2) else 0)
            extra_block = np.full((len(vars_int), T, N), np.nan, dtype=float)
            result_blocks.append(extra_block)

        # for_explan shape mirrors legacy: shape of first variable (T, N)
        first_shape = (result_blocks[0][0].shape[0], result_blocks[0][0].shape[1]) if result_blocks else (0, 0)
        return result_blocks, (p_mon, vars_int, first_shape)

    # ---- otherwise, fall back to legacy step_*.npy files ----
    # compute which step files to read
    count_begin = int(time_begin / save_time)
    # -1 to ignore step_init
    n_steps = len([n for n in os.listdir(run_dir)
                   if os.path.isfile(os.path.join(run_dir, n)) and n.startswith('step_')]) - 1
    count_end = min(int(time_end / save_time) + 1, n_steps)

    # Count monitors the legacy way
    nb_monitor = p_mon.get('Raw', False) + p_mon.get('TemporalAverage', False) + p_mon.get('Bold', False) + p_mon.get('Ca', False)
    if p_mon.get('Afferent_coupling', False):
        nb_monitor += 1

    # Accumulators per monitor
    # Each entry will be a list of time blocks & data blocks we concatenate after the loop
    times_list = [[] for _ in range(nb_monitor)]
    data_list  = [[] for _ in range(nb_monitor)]  # each element stores (n_t, n_vars, n_nodes)

    for count in range(count_begin, max(count_end, count_begin)):
        fpath = os.path.join(run_dir, f'step_{count}.npy')
        if not os.path.exists(fpath):
            continue
        step_arr = np.load(fpath, allow_pickle=True)
        # step_arr shape: (nb_monitor,) object; each contains a list of (t, y) blocks
        for i in range(min(nb_monitor, step_arr.shape[0])):
            chunks = step_arr[i]
            if chunks is None or len(chunks) == 0:
                continue
            # chunks is a list of tuples (t_block, y_block)
            block_times = []
            block_datas = []
            for (t_blk, y_blk) in chunks:
                t_blk = np.asarray(t_blk).reshape(-1)
                y_blk = np.asarray(y_blk)
                # ensure y shape is (n_t, n_vars, n_nodes)
                if y_blk.ndim == 2:
                    y_blk = y_blk[:, :, None]
                # select times within [time_begin, time_end]
                if t_blk.size == 0:
                    continue
                sel = (t_blk >= time_begin) & (t_blk <= time_end)
                if not np.any(sel):
                    continue
                block_times.append(t_blk[sel])
                block_datas.append(y_blk[sel, :, :])
            if block_times:
                times_list[i].append(np.concatenate(block_times, axis=0))
                data_list[i].append(np.concatenate(block_datas, axis=0))

    # Build the legacy-shaped outputs per monitor: (len(vars_int), T, N)
    result = []
    for i in range(nb_monitor):
        if not times_list[i]:
            # empty monitor → 0-length block
            result.append(np.zeros((len(vars_int), 0, 0), dtype=float))
            continue

        t_cat = np.concatenate(times_list[i], axis=0)
        y_cat = np.concatenate(data_list[i], axis=0)  # (T, n_vars_all, N)

        T, n_vars_all, N = y_cat.shape
        out = np.zeros((len(vars_int), T, N), dtype=float)
        out[:] = np.nan

        # For requested vars, copy from the correct columns
        # E/I/noise are stored in kHz in legacy -> convert to Hz
        for j, name in enumerate(vars_int):
            col = dict_var_int.get(name, None)
            if col is None or col >= n_vars_all:
                continue
            vec = y_cat[:, col, :]  # (T, N)
            if name in ('E', 'I', 'noise'):
                out[j] = 1e3 * vec
            else:
                out[j] = vec

        result.append(out)

    first_shape = (result[0][0].shape[0], result[0][0].shape[1]) if result and result[0].shape[1] else (0, 0)
    return result, (p_mon, vars_int, first_shape)


def pvalue_to_asterisks(pvalue):
    if pvalue < 0.001:
        return '***'
    if pvalue < 0.01:
        return '**'
    if pvalue < 0.05:
        return '*'
    return 'ns'

def find_file_seed(simconfig, path, file_prefix, seeds=None, n_minimal_seeds=None):
    """
    Locate a saved file corresponding to a specific simulation configuration.

    The function supports two search modes:
    --------------------------------------
      1. Exact seed list match  -> filenames like:  myprefix_simname_[1,2,3].npy
      2. Minimal seed count     -> filenames like:  myprefix_simname_10.npy

    Parameters
    ----------
    simconfig : SimConfig
        The simulation configuration object (used to derive folder name).
    path : str
        Directory path to search in.
    file_prefix : str
        Prefix used when saving the files (e.g., 'LZc', 'PCI', etc.).
    seeds : list[int] or None
        Exact list of seeds. If provided, will look for a file ending with
        `_[1,2,3].npy` matching this list.
    n_minimal_seeds : int or None
        If provided (and seeds is None), will look for files with a numeric
        suffix `_<num>.npy` where num >= n_minimal_seeds.

    Returns
    -------
    str or bool
        Full path to the matching file, or False if not found.
    """
    folder_root, sim_name = simconfig.get_sim_name(0)
    if path is not None:
        folder_root = path

    if not os.path.exists(folder_root):
        Printer.print(f"Path not found: {folder_root}", level=2)
        return False

    for file in os.listdir(folder_root):
        filename = os.fsdecode(file)

        # 1. Match filenames like ..._[1,2,3].npy
        match_list = re.match(r'^(.+)_\[(.+)\]\.npy$', filename)
        if match_list and seeds is not None:
            sim_name_found = match_list.group(1)
            seed_str = match_list.group(2)
            try:
                seed_list = ast.literal_eval(f'[{seed_str}]')
            except (SyntaxError, ValueError):
                continue

            expected_prefix = file_prefix + '_' + sim_name[:-2]
            if sim_name_found == expected_prefix and seed_list == seeds:
                Printer.print(f"{filename} found (exact seed list match).")
                _, simconfig_name = simconfig.get_sim_name(len(seeds))
                return os.path.join(folder_root, f"{file_prefix}_{simconfig_name}.npy")

        # 2. Match filenames like ..._10.npy (integer suffix)
        match_num = re.match(r'^(.+)_([0-9]+)\.npy$', filename)
        if match_num and n_minimal_seeds is not None and seeds is None:
            sim_name_found = match_num.group(1)
            n_seeds = int(match_num.group(2))
            expected_prefix = file_prefix + '_' + sim_name[:-2]
            if sim_name_found == expected_prefix and n_seeds >= n_minimal_seeds:
                Printer.print(f"{filename} found (minimal seed count match).")
                _, simconfig_name = simconfig.get_sim_name(n_seeds)
                return os.path.join(folder_root, f"{file_prefix}_{simconfig_name}.npy")

    Printer.print("No matching file found.", level=1)
    return False


def create_dicts(simconfig, result, monitor, for_explan, var_select,
                 seed=10, additional_path_folder='', return_TR=False):
    """
    Convert legacy TVB simulation result arrays into a dict of variables.

    This is used by plotting functions to extract time-series arrays for
    variables like 'E', 'I', 'W_e', etc., from the legacy result structure.

    Parameters
    ----------
    simconfig : SimConfig
        The simulation configuration object.
    result : list[np.ndarray]
        Output from `get_result()` corresponding to a specific monitor.
    monitor : str
        Which monitor to extract ('Raw', 'TemporalAverage', 'Bold', etc.).
    for_explan : tuple
        Metadata returned by `get_result` — (parameter_monitor, vars_int, shape).
    var_select : list[str]
        Variables to extract (e.g., ['E', 'I']).
    seed : int
        Simulation seed (used only for file naming/debugging).
    additional_path_folder : str
        Optional subfolder (not used here, kept for compatibility).
    return_TR : bool
        If True, returns (result_dict, TR) where TR is the BOLD period.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping each selected variable name -> array (T, n_regions).
    float (optional)
        If return_TR=True and monitor=='Bold', also returns the TR period.
    """
    parameters = simconfig.general_parameters
    parameter_monitor = parameters.parameter_monitor

    # Determine index of the requested monitor
    list_monitors = {}
    c = 0
    for key, enabled in parameter_monitor.items():
        if enabled is True:
            list_monitors[key] = c
            c += 1

    if monitor not in list_monitors:
        raise ValueError(f"Monitor '{monitor}' not active in this simulation.")

    result = result[list_monitors[monitor]]  # extract the correct monitor array

    # Map variable names to indices
    _, vars_int, _ = for_explan
    list_vars = {var: k for k, var in enumerate(vars_int)}

    result_fin = {}
    n_regions = result[0].shape[1] if isinstance(result[0], np.ndarray) else result[0][0].shape[1]

    disconnect_regions = simconfig.custom_parameters.get('disconnect_regions', [])
    include_regions = [i for i in range(n_regions) if i not in disconnect_regions]

    for var in var_select:
        if var not in list_vars:
            Printer.print(f"Variable '{var}' not found in result.", level=2)
            continue

        idx = list_vars[var]
        data = result[idx][:, include_regions]
        if monitor.lower() == 'bold' and var in ('E', 'I'):
            data = zscore(data, axis=0)
        result_fin[var] = data

    if return_TR and "parameter_Bold" in parameter_monitor:
        TR = parameter_monitor["parameter_Bold"]["period"]
        return result_fin, TR

    return result_fin


