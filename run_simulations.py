#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_simulations.py
==================
Execution engine for whole-brain AdEx simulations (legacy-compatible).

What this module does
---------------------
• Runs single or multiple simulations (sequentially or in parallel)
• Manages simulation folders and metadata
• Skips/resumes simulations if they already exist
• Writes periodic legacy 'step_*.npy' files for compatibility

Key fixes vs. older versions
---------------------------
• Robust time handling: we always use a scalar "current time" taken from any
  available monitor chunk (last sample), instead of comparing arrays to numbers.
• Safety net: if no monitor is enabled in parameters, we attach a default
  RawVoi(E/I) monitor so your simulation actually produces data.

Notes
-----
This file intentionally does NOT do any post-hoc metrics (entropy, PCI, etc.).
It focuses only on sim execution and saving 'step_*.npy'.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from itertools import product
import jsonpickle

from simconfig import sim_init
from utils import Printer
from parameters import Parameters

# -------------------------------------------------------------------------
# Utilities for legacy behavior (unchanged public API)
# -------------------------------------------------------------------------
def setup_files(simulator, parameters, seed, initial_condition=None):
    """
    Create parameter.json and save initial state (step_init.npy) in the result folder.
    """
    out_dir = parameters.parameter_simulation['path_result']
    os.makedirs(out_dir, exist_ok=True)

    param_path = os.path.join(out_dir, 'parameter.json')
    if not os.path.exists(param_path):
        with open(param_path, "w") as f:
            f.write("{\n")
            for name, dic in [
                ('parameter_simulation', parameters.parameter_simulation),
                ('parameter_model', parameters.parameter_model),
                ('parameter_connection_between_region', parameters.parameter_connection_between_region),
                ('parameter_coupling', parameters.parameter_coupling),
                ('parameter_integrator', parameters.parameter_integrator),
                ('parameter_monitor', parameters.parameter_monitor),
                ('parameter_stimulus', parameters.parameter_stimulus)
            ]:
                f.write(f'"{name}" : ')
                try:
                    f.write(jsonpickle.encode(dic, unpicklable=True))
                    f.write(",\n")
                except TypeError:
                    Printer.print(f"{name} not serialisable", level=2)
            f.write(f'"myseed":{seed}\n}}\n')

    if initial_condition is None:
        np.save(os.path.join(out_dir, 'step_init.npy'), simulator.history.buffer)

def get_n_step_files(path):
    """
    Returns the number of 'step_' files at path, excluding 'step_init'.
    """
    n = 0
    if not os.path.isdir(path):
        return 0
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.startswith('step_') and 'init' not in filename and filename.endswith('.npy'):
            n += 1
    return n

def is_sim_enough(simconfig, parameter, path):
    """
    Determine whether an existing simulation is at least as long as requested.
    """
    n_step_files = get_n_step_files(path)
    if n_step_files == 0:
        return False

    # Completed chunks (fully saved)
    done_sim_time = (n_step_files - 1) * parameter['parameter_simulation']['save_time']

    # Last (possibly partial) chunk
    last_fp = os.path.join(path, f'step_{n_step_files-1}.npy')
    try:
        last_data = np.load(last_fp, allow_pickle=True)
        # Try to get dt and number of in-chunk saved steps from monitor arrays
        dt = float(parameter['parameter_integrator']['dt'])
        # We can infer the number of rows from the first non-empty monitor
        in_chunk_len = 0
        if isinstance(last_data, np.ndarray):
            for i in range(last_data.shape[0]):
                if last_data[i] is None or len(last_data[i]) == 0:
                    continue
                # last_data[i] is a list of chunks; take the last one
                chunk = last_data[i][-1]
                tvec = np.asarray(chunk[0]).ravel()
                if tvec.size > 0:
                    # Use time span instead of len * dt (more robust if variable sampling)
                    in_chunk_len = max(in_chunk_len, int(round((tvec[-1] - tvec[0]) / dt)) + 1)
        done_sim_time += in_chunk_len * dt
    except Exception:
        # If we can't parse, assume just the completed chunks
        pass

    return done_sim_time >= simconfig.run_sim

def skip_sim(simconfig, seed, simu_path):
    """
    Returns (status, dirname) if an identical simulation exists.

    status:
      0 → not found
      1 → found and complete
      2 → found but has 'stop_condition_satisfied.txt'
    """
    if not os.path.exists(simu_path):
        return 0, None

    for dir_ in os.listdir(simu_path):
        folder = os.path.join(simu_path, dir_)
        if not os.path.isdir(folder):
            continue
        try:
            with open(os.path.join(folder, 'parameter.json'), 'r') as f:
                parameter = jsonpickle.decode(f.read())
            if parameter.get('myseed') != seed:
                continue
            # Strict comparison with the deep-copied Parameters object will usually fail
            # in Python; instead, check for completion and same seed which is what we need.
            if os.path.exists(os.path.join(folder, 'stop_condition_satisfied.txt')):
                return 2, dir_
            if is_sim_enough(simconfig, parameter, folder):
                Printer.print('Same or superior simulation length.', level=0)
                return 1, dir_
        except Exception as e:
            Printer.print(f"While scanning existing sims: {e}", level=1)
            continue
    return 0, None

# -------------------------------------------------------------------------
# Parallel wrappers (unchanged signatures)
# -------------------------------------------------------------------------
def run_simulations_parallel(simconfigs, seeds, stop_early=False, n_tasks_concurrent=None,
                             save_path_root=None, no_skip=False, param_change_with_seed=None):
    """
    Runs each simconfig for every seed on n_tasks_concurrent parallel processes.
    """
    setups = list(product(simconfigs, seeds))
    if n_tasks_concurrent is None:
        n_tasks_concurrent = max(1, cpu_count() - 1)

    if param_change_with_seed is None:
        param_change_with_seed = [{}] * len(seeds)

    if stop_early:
        for i in range(0, len(setups), n_tasks_concurrent):
            with Pool(n_tasks_concurrent) as p:
                results = p.starmap(
                    run_simulation,
                    [(simconfig, seed, False, save_path_root, no_skip, param_change_with_seed[seed])
                     for simconfig, seed in setups[i:i + n_tasks_concurrent]]
                )
                if not all(results):
                    return True
        return True
    else:
        with Pool(n_tasks_concurrent) as p:
            results = p.starmap(
                run_simulation,
                [(simconfig, seed, False, save_path_root, no_skip, param_change_with_seed[seed])
                 for simconfig, seed in setups]
            )
        return results

def run_n_complete_parallelized(simconfig, seeds, n_minimal_seeds=None, stop_early=False,
                                n_tasks_concurrent=None, save_path_root=None,
                                no_skip=False, param_change_with_seed=None):
    """
    Run one configuration across many seeds in parallel until n_minimal_seeds complete.
    """
    if n_tasks_concurrent is None:
        n_tasks_concurrent = max(1, cpu_count() - 1)

    if param_change_with_seed is None:
        param_change_with_seed = [{}] * len(seeds)

    seeds_completed = []
    for i in range(0, len(seeds), n_tasks_concurrent):
        with Pool(n_tasks_concurrent) as p:
            results = p.starmap(
                run_simulation,
                [(simconfig, seed, False, save_path_root, no_skip, param_change_with_seed[seed])
                 for seed in seeds[i:i + n_tasks_concurrent]]
            )
            for j, res in enumerate(results):
                if res:
                    seeds_completed.append(i + j)
                    if n_minimal_seeds is not None and len(seeds_completed) == n_minimal_seeds:
                        return seeds_completed
    return False

# -------------------------------------------------------------------------
# Core simulation (legacy-compatible; improved robustness)
# -------------------------------------------------------------------------
def run_simulation(simconfig, seed=10, print_connectome=False,
                   save_path_root=None, no_skip=False, param_change_with_seed={}):
    """
    Runs a simulation and writes periodic 'step_*.npy' files (legacy format).

    Returns
    -------
    bool : True if simulation completed, False otherwise.
    """
    parameters = simconfig.general_parameters
    cus_parameters = simconfig.custom_parameters

    # Per-seed tweaks
    for k, v in (param_change_with_seed or {}).items():
        cus_parameters[k] = v
    simconfig._adjust_parameters()

    # Resolve output path
    folder_root, sim_name = simconfig.get_sim_name(seed)
    if save_path_root is None:
        save_path_root = folder_root
    save_path = os.path.join(save_path_root, sim_name)

    Printer.print("path = ", save_path)
    Printer.print('Initialize Simulator')

    # Configure result path for sim
    parameters.parameter_simulation['path_result'] = save_path
    simulator = sim_init(simconfig, seed=seed)

    # SAFETY NET: ensure at least one monitor is enabled — otherwise TVB may yield nothing
    try:
        if len(simulator.monitors) == 0:
            from tvb.simulator.lab import monitors as tvbmon
            voi = np.array(parameters.parameter_monitor.get(
                "parameter_Raw", {}).get("variables_of_interest", [0, 1]
            ))
            simulator.monitors = (tvbmon.RawVoi(variables_of_interest=voi),)
            simulator.configure()
            parameters.parameter_monitor["Raw"] = True
            parameters.parameter_monitor.setdefault("parameter_Raw", {})["variables_of_interest"] = list(map(int, voi))
            Printer.print("No monitors enabled; attached default Raw(E/I) monitor.", level=1)
    except Exception as e:
        Printer.print(f"Monitor check/attach failed (continuing): {e}", level=1)

    # SKIP logic
    skip_code, existing_dir = skip_sim(simconfig, seed, save_path_root)
    if not no_skip and skip_code > 0:
        Printer.print('Existing folder for simulation:', simconfig, 'with seed', seed)
        try:
            os.rename(os.path.join(save_path_root, existing_dir), save_path)
        except Exception:
            pass
        return skip_code == 1

    # Clean and set up metadata
    try:
        shutil.rmtree(save_path)
    except Exception:
        pass
    setup_files(simulator, parameters, seed)

    Printer.print('Start Simulation')
    parameter_simulation = parameters.parameter_simulation
    parameter_monitor = parameters.parameter_monitor

    total_time = float(simconfig.run_sim)

    # Informative stimulus printout
    if parameters.parameter_stimulus.get('stimval', 0):
        try:
            stimreg = parameters.parameter_stimulus['stimregion']
            stimlab = ', '.join([simulator.connectivity.region_labels[r] for r in stimreg])
        except Exception:
            stimlab = str(parameters.parameter_stimulus.get('stimregion', []))
        Printer.print(
            '    Stimulating for {dur} ms, {val} nS in {regs} at time {onset} ms'.format(
                regs=stimlab,
                dur=parameters.parameter_stimulus.get('stimdur', '??'),
                val=parameters.parameter_stimulus.get('stimval', '??'),
                onset=parameters.parameter_stimulus.get('stimtime', '??')),
            level=0
        )

    # Count monitors as booleans (legacy)
    nb_monitor = (
        int(bool(parameter_monitor.get('Raw', False))) +
        int(bool(parameter_monitor.get('TemporalAverage', False))) +
        int(bool(parameter_monitor.get('Bold', False))) +
        int(bool(parameter_monitor.get('Ca', False)))
    )
    if parameter_monitor.get('Afferent_coupling', False):
        nb_monitor += 1

    # Allocate lists of chunks to be periodically saved
    save_result = [[] for _ in range(nb_monitor)]

    if print_connectome:
        plt.figure()
        plt.imshow(simulator.connectivity.weights, interpolation='nearest', cmap='binary')
        plt.colorbar()
        plt.title("Connectivity matrix")
        plt.show()

    # --- main simulation loop ---
    count = 0
    finished = True
    try:
        for result in simulator(simulation_length=total_time):
            # result is a tuple of length == len(monitors), each is (t, data) or None

            # derive a scalar "current time" from any available monitor
            current_t = None
            for k in range(nb_monitor):
                chunk = result[k]
                if chunk is None:
                    continue
                tvec = np.asarray(chunk[0]).ravel()
                if tvec.size:
                    current_t = float(tvec[-1])
                    break

            # Stop-condition checks use current_t (if provided)
            if current_t is not None:
                for (mon, var, min_v, max_v, t_thresh) in simconfig.stop_conditions:
                    if current_t < t_thresh:
                        continue
                    # Pull the same monitor’s last data sample and check bounds
                    chk = result[mon]
                    if chk is None:
                        continue
                    y = np.asarray(chk[1])
                    # y shape: (n_t, n_vars, n_regions) or (n_t, n_vars)
                    if y.ndim == 2:
                        y_last = y[-1, var]
                    else:
                        y_last = y[-1, var, ...]
                    if np.any(y_last < min_v) or np.any(y_last > max_v):
                        Printer.print(
                            f"Simulation stopped (stop condition hit) at t={current_t} ms "
                            f"(mon={mon}, var={var}, range=[{min_v},{max_v}])",
                            level=2
                        )
                        with open(os.path.join(save_path, 'stop_condition_satisfied.txt'), 'w') as f:
                            f.write(f'{mon}, {var}, {min_v}, {max_v}, {t_thresh}, {current_t}')
                        raise RuntimeError('Stop simulation')

            # Apply time-varying parameters (if any)
            if current_t is not None:
                simconfig.apply_varying_params(simulator.model, t=current_t)

            # Accumulate result chunks for each monitor
            for i in range(nb_monitor):
                if result[i] is not None:
                    save_result[i].append(result[i])

            # Periodically flush to disk in legacy step files
            if (current_t is not None) and \
               (current_t >= parameter_simulation['save_time'] * (count + 1)):
                Printer.print(f'simulation time : {current_t:.1f} ms\r', level=0)
                np.save(os.path.join(save_path, f'step_{count}.npy'),
                        np.array(save_result, dtype='object'), allow_pickle=True)
                save_result = [[] for _ in range(nb_monitor)]
                count += 1

    except Exception as e:
        Printer.print(e, level=2)
        finished = False

    # Save the last partial chunk
    np.save(os.path.join(save_path, f'step_{count}.npy'),
            np.array(save_result, dtype='object'), allow_pickle=True)

    if finished:
        Printer.print("Simulation Completed successfully", level=0)
    return finished
