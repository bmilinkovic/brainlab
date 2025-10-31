#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_simulations.py
==================
Execution engine for whole-brain AdEx simulations.

This module handles:
  • Running single or multiple simulations (sequentially or in parallel)
  • Managing simulation folders and metadata
  • Skipping or resuming simulations if they already exist
  • Handling stop conditions for unstable simulations

It does *not* compute post-hoc measures or analysis — it focuses purely
on running and storing simulation data.

Typical usage
-------------
>>> from simconfig import SimConfig
>>> from parameters import Parameters
>>> from run_simulations import run_simulation, run_simulations_parallel
>>> 
>>> params = Parameters()
>>> cfg = SimConfig(params, run_sim=10000, cut_transient=1000, parameters={"b_e": 5.0})
>>> run_simulation(cfg, seed=42)

For large batches:
>>> results = run_simulations_parallel([cfg], seeds=[1,2,3,4])
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
# 1. PARALLEL EXECUTION WRAPPERS
# -------------------------------------------------------------------------
def run_simulations_parallel(simconfigs, seeds, stop_early=False, n_tasks_concurrent=None,
                             save_path_root=None, no_skip=False, param_change_with_seed=None):
    """
    Run multiple simulations in parallel across different configs and seeds.

    Parameters
    ----------
    simconfigs : list[SimConfig]
        Simulation configurations to run.
    seeds : list[int]
        Random seeds for each run.
    stop_early : bool, optional
        Stop after the first batch if any simulation fails.
    n_tasks_concurrent : int, optional
        Number of parallel processes. Defaults to (CPU cores - 1).
    save_path_root : str, optional
        Root folder for simulation output.
    no_skip : bool, optional
        If True, force re-run even if identical simulation exists.
    param_change_with_seed : list[dict], optional
        Optional per-seed parameter modifications.

    Returns
    -------
    list[bool]
        True/False per simulation indicating success.
    """
    setups = list(product(simconfigs, seeds))
    n_tasks_concurrent = n_tasks_concurrent or (cpu_count() - 1)
    param_change_with_seed = param_change_with_seed or [{}] * len(seeds)

    if stop_early:
        for i in range(0, len(setups), n_tasks_concurrent):
            with Pool(n_tasks_concurrent) as p:
                results = p.starmap(
                    run_simulation,
                    [(cfg, seed, False, save_path_root, no_skip, param_change_with_seed[seed])
                     for cfg, seed in setups[i:i + n_tasks_concurrent]]
                )
                if not all(results):
                    return True
    else:
        with Pool(n_tasks_concurrent) as p:
            results = p.starmap(
                run_simulation,
                [(cfg, seed, False, save_path_root, no_skip, param_change_with_seed[seed])
                 for cfg, seed in setups]
            )
    return results


def run_n_complete_parallelized(simconfig, seeds, n_minimal_seeds=None, stop_early=False,
                                n_tasks_concurrent=None, save_path_root=None,
                                no_skip=False, param_change_with_seed=None):
    """
    Run one simulation configuration on multiple seeds in parallel
    until a minimum number complete successfully.

    Useful when some runs fail due to instability (e.g. “explosions”).

    Parameters
    ----------
    simconfig : SimConfig
        Configuration defining the simulation setup.
    seeds : list[int]
        Random seeds to iterate through.
    n_minimal_seeds : int, optional
        Minimum number of completed simulations required.
    stop_early : bool
        Stop immediately when one fails.
    n_tasks_concurrent : int, optional
        Number of parallel processes. Defaults to CPU count - 1.
    save_path_root : str, optional
        Base directory for saving results.
    no_skip : bool
        Force re-run even if identical simulation exists.
    param_change_with_seed : list[dict], optional
        Per-seed parameter overrides.

    Returns
    -------
    list[int] or bool
        List of completed seeds, or False if insufficient.
    """
    n_tasks_concurrent = n_tasks_concurrent or (cpu_count() - 1)
    param_change_with_seed = param_change_with_seed or [{}] * len(seeds)
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
                    if len(seeds_completed) == n_minimal_seeds:
                        return seeds_completed
    return False


def run_n_complete(simconfig, seeds, n_minimal_seeds, simu_path):
    """
    Run simulations sequentially until a target number complete.

    Parameters
    ----------
    simconfig : SimConfig
        Simulation configuration.
    seeds : list[int]
        Seeds to iterate through.
    n_minimal_seeds : int
        Target number of successful runs.
    simu_path : str
        Directory to store simulation results.

    Returns
    -------
    list[int] or bool
        Seeds that completed successfully, or False if insufficient.
    """
    seeds_completed = []
    for s, seed in enumerate(seeds):
        finished = run_simulation(simconfig=simconfig, seed=seed, save_path_root=simu_path)
        if finished:
            seeds_completed.append(seed)
            if len(seeds_completed) == n_minimal_seeds:
                break
    if len(seeds_completed) != n_minimal_seeds:
        Printer.print(
            f"Not enough seeds given ({len(seeds)}) to reach {n_minimal_seeds} completed runs "
            f"({len(seeds_completed)} successful)."
        )
        return False
    return seeds_completed


# -------------------------------------------------------------------------
# 2. SIMULATION MANAGEMENT HELPERS
# -------------------------------------------------------------------------
def setup_files(simulator, parameters, seed, initial_condition=None):
    """
    Create parameter.json and save initial simulator state.

    This metadata file documents the parameters used for reproducibility.

    Parameters
    ----------
    simulator : tvb.simulator.Simulator
        Configured simulator instance.
    parameters : Parameters
        Parameter object used for this run.
    seed : int
        Random seed.
    initial_condition : optional
        Initial condition to save, if provided.
    """
    out_dir = parameters.parameter_simulation["path_result"]
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "parameter.json")
    if not os.path.exists(json_path):
        with open(json_path, "w") as f:
            f.write(jsonpickle.encode({
                "parameter_simulation": parameters.parameter_simulation,
                "parameter_model": parameters.parameter_model,
                "parameter_connection_between_region": parameters.parameter_connection_between_region,
                "parameter_coupling": parameters.parameter_coupling,
                "parameter_integrator": parameters.parameter_integrator,
                "parameter_monitor": parameters.parameter_monitor,
                "parameter_stimulus": parameters.parameter_stimulus,
                "myseed": seed
            }, unpicklable=True))

    if initial_condition is None:
        np.save(
            os.path.join(out_dir, "step_init.npy"),
            simulator.history.buffer
        )


def get_n_step_files(path):
    """Count how many simulation step files exist in a directory (excluding 'step_init')."""
    return sum(1 for f in os.listdir(path)
               if f.startswith("step_") and "init" not in f)


def is_sim_enough(simconfig, parameter, path):
    """
    Check if an existing simulation is complete.

    Parameters
    ----------
    simconfig : SimConfig
        Simulation configuration.
    parameter : dict
        Decoded parameter.json contents.
    path : str
        Folder path containing simulation output.

    Returns
    -------
    bool
        True if the simulated time >= requested run time.
    """
    n_step = get_n_step_files(path)
    if n_step == 0:
        return False
    done_sim_time = (n_step - 1) * parameter["parameter_simulation"]["save_time"]
    last_data = np.load(os.path.join(path, f"step_{n_step-1}.npy"), allow_pickle=True)
    done_sim_time += last_data.shape[1] * parameter["parameter_integrator"]["dt"]
    return done_sim_time >= simconfig.run_sim


def skip_sim(simconfig, seed, simu_path):
    """
    Check if an identical simulation (same parameters + seed) already exists.

    Returns
    -------
    tuple[int, str or None]
        (status_code, folder_name)
        status_code:
            0 → not found
            1 → found and complete
            2 → found but stopped early
    """
    if not os.path.exists(simu_path):
        return 0, None

    for dir_ in os.listdir(simu_path):
        folder = os.path.join(simu_path, dir_)
        if not os.path.isdir(folder):
            continue
        try:
            with open(os.path.join(folder, "parameter.json"), "r") as f:
                param = jsonpickle.decode(f.read())
            if param.get("myseed") != seed:
                continue
            if param == simconfig.general_parameters:
                Printer.print("Existing simulation found.")
                if os.path.exists(os.path.join(folder, "stop_condition_satisfied.txt")):
                    return 2, dir_
                if is_sim_enough(simconfig, param, folder):
                    return 1, dir_
        except Exception as e:
            print("Error reading parameter.json:", e)
            continue
    return 0, None


# -------------------------------------------------------------------------
# 3. CORE SIMULATION LOOP
# -------------------------------------------------------------------------
def run_simulation(simconfig, seed=10, print_connectome=False,
                   save_path_root=None, no_skip=False, param_change_with_seed=None):
    """
    Run a single simulation using a given configuration.

    Parameters
    ----------
    simconfig : SimConfig
        Full configuration (model, connectivity, coupling, etc.).
    seed : int
        Random seed.
    print_connectome : bool
        If True, plots the connectivity matrix before running.
    save_path_root : str
        Root directory for saving results.
    no_skip : bool
        Force re-run even if identical simulation exists.
    param_change_with_seed : dict, optional
        Optional per-seed parameter modifications.

    Returns
    -------
    bool
        True if the simulation finished successfully, False otherwise.
    """
    parameters = simconfig.general_parameters
    custom_params = simconfig.custom_parameters

    for k, v in (param_change_with_seed or {}).items():
        custom_params[k] = v
    simconfig._adjust_parameters()

    folder_root, sim_name = simconfig.get_sim_name(seed)
    save_path_root = save_path_root or folder_root
    save_path = os.path.join(save_path_root, sim_name)

    Printer.print("Initializing simulator...")
    parameters.parameter_simulation["path_result"] = save_path
    simulator = sim_init(simconfig, seed=seed)

    skip_code, existing_dir = skip_sim(simconfig, seed, save_path_root)
    if not no_skip and skip_code > 0:
        os.rename(os.path.join(save_path_root, existing_dir), save_path)
        return skip_code == 1

    shutil.rmtree(save_path, ignore_errors=True)
    setup_files(simulator, parameters, seed)
    Printer.print("Starting simulation...")

    nb_monitor = sum([
        parameters.parameter_monitor.get(k, False)
        for k in ["Raw", "TemporalAverage", "Bold", "Ca"]
    ])
    if parameters.parameter_monitor.get("Afferent_coupling", False):
        nb_monitor += 1

    if print_connectome:
        plt.imshow(simulator.connectivity.weights, cmap="binary")
        plt.colorbar()
        plt.title("Connectivity matrix")
        plt.show()

    save_result = [[] for _ in range(nb_monitor)]
    count, finished = 0, True

    try:
        for result in simulator(simulation_length=simconfig.run_sim):
            # Apply time-varying parameters
            simconfig.apply_varying_params(simulator.model, t=result[0][0])

            for i in range(nb_monitor):
                if result[i] is not None:
                    save_result[i].append(result[i])

            # Save periodically
            if result[0][0] >= parameters.parameter_simulation["save_time"] * (count + 1):
                np.save(f"{save_path}/step_{count}.npy", np.array(save_result, dtype="object"))
                save_result = [[] for _ in range(nb_monitor)]
                count += 1
    except Exception as e:
        Printer.print(e, 2)
        finished = False

    np.save(f"{save_path}/step_{count}.npy", np.array(save_result, dtype="object"))
    if finished:
        Printer.print("Simulation completed successfully.")
    return finished
