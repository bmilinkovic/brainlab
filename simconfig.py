#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simconfig.py
============

Core configuration and simulator factory for whole-brain AdEx simulations.

This version is organized by sections and supports SUBJECT-SPECIFIC connectivity
by default (one participant per run), with an OPTIONAL mode to average SC/TL
across multiple subjects.

Data layout (CSV):
    data/organised/dataframes/SC_CNT.csv   # edge list per subject
    data/organised/dataframes/TL_CNT.csv   # edge list per subject

Helper API (from tvb_adex.io.connectome):
    - subjects_in_edge_csv(sc_csv) -> list of subject IDs (e.g., ['sub-01', ...])
    - load_connectivity_from_edge_csvs(sc_csv, tl_csv, subject, symmetric=True) -> TVB Connectivity

Sections:
    1) Imports & depends
    2) SimConfig (parameter management, reporting)
    3) Connectivity utilities (subject list / averaging)
    4) Simulator factory (sim_init)
"""

# ---------------------------------------------------------------------------
# 1) Imports & depends
# ---------------------------------------------------------------------------
import os
import copy
import collections.abc
import numpy as np
import numpy.random as rgn
import tvb.simulator.lab as lab

# project utils
from utils import Printer, print_dict_differences
from parameters import Parameter, ListParameter, Parameters


# connectivity helpers
from connectome import subjects_in_edge_csv, load_connectivity_from_edge_csvs

# ---------------------------------------------------------------------------
# 2) SimConfig: parameter management & reporting
# ---------------------------------------------------------------------------
class SimConfig:
    """
    SimConfig
    =========
    Holds a deep copy of default Parameters and applies user overrides.
    Also carries naming/reporting preferences and time-varying parameters.

    Parameters
    ----------
    general_parameters : Parameters
        Default parameter object.
    run_sim : float
        Simulation duration (ms).
    cut_transient : float
        Time (ms) to ignore at the beginning of results (downstream usage).
    parameters : dict | None
        Flat dict of parameter overrides. Keys can belong to any of the
        'parameter_*' dicts inside `Parameters`.
    params_to_report : list[str] | None
        Keys to include in file names / plot titles.
    stop_conditions : list[tuple] | None
        Stop conditions: (monitor_id, var_id, min, max, t). (Not used here.)
    auto_report : bool
        If True, automatically add modified keys to params_to_report.
    """

    def __init__(self, general_parameters, run_sim, cut_transient,
                 parameters=None, params_to_report=None,
                 stop_conditions=None, auto_report=False):

        self.run_sim = run_sim
        self.cut_transient = cut_transient
        self.general_parameters = copy.deepcopy(general_parameters)
        self.custom_parameters = parameters or {}
        self.stop_conditions = stop_conditions or []

        # All parameter dicts from Parameters
        parameter_dicts = [
            v for k, v in vars(self.general_parameters).items()
            if k.startswith("parameter")
        ]

        # Validate & prepare custom overrides
        for name, value in self.custom_parameters.items():
            if not any(name in d for d in parameter_dicts):
                raise KeyError(f"Parameter '{name}' not found in default Parameters.")
            # Time-varying lists -> ListParameter
            if not isinstance(value, str) and isinstance(value, collections.abc.Iterable):
                if not isinstance(value, dict):
                    self.custom_parameters[name] = ListParameter(name, value, "list")
                    continue
            # Numpy scalars -> plain float
            if isinstance(value, np.floating):
                Printer.print(
                    f"Parameter {name} has NumPy type {type(value)}, converting to float."
                )
                self.custom_parameters[name] = float(value)

        # Identify time-varying (Parameter-based) custom parameters
        self.special_parameters = [
            n for n, v in self.custom_parameters.items()
            if isinstance(v, Parameter)
        ]

        # Reporting
        self.params_to_report = params_to_report or []
        if auto_report:
            self.params_to_report.extend(self.custom_parameters.keys())
            self.params_to_report = list(dict.fromkeys(self.params_to_report))

        # Ensure reported params have values (inject defaults if not overridden)
        for name in self.params_to_report:
            if name not in self.custom_parameters:
                for d in parameter_dicts:
                    if name in d:
                        self.custom_parameters[name] = d[name]
                        break

        # Apply overrides into deep-copied general_parameters
        self._adjust_parameters()


    def _adjust_parameters(self):
        """
        Write custom parameters into the deep-copied defaults, converting any
        time-varying Parameter objects to their initial numeric value at t=0.
        """
        if not self.custom_parameters:
            return

        # All parameter dicts from the Parameters object
        param_dicts = [v for k, v in vars(self.general_parameters).items() if k.startswith("parameter")]
        # Include nested dicts (e.g., coupling_parameter, noise_parameter, ...)
        nested = []
        for d in param_dicts:
            for v in d.values():
                if isinstance(v, dict):
                    nested.append(v)
        param_dicts += nested

        for name, value in self.custom_parameters.items():
            for d in param_dicts:
                if name in d:
                    # If it's a time-varying wrapper, take initial value at t=0
                    if isinstance(value, Parameter):
                        try:
                            d[name] = value.get(t=0)
                        except Exception:
                            d[name] = getattr(value, "value", value)
                    else:
                        d[name] = value
                    break  # done with this name

    # ----- Reporting helpers -----
    def get_sim_name(self, seed=""):
        folder_root = "./result/synch"
        sim_name = ""
        if self.custom_parameters.get("stimval", 0):
            folder_root = "./result/evoked"
            sim_name += f"stim_{self.custom_parameters['stimval']}_"
        if self.custom_parameters.get("disconnect_regions", []):
            dr = self.custom_parameters["disconnect_regions"]
            sim_name += f"dr_{len(dr)}_r_{dr[0]}_"
        sim_name += "_".join([f"{n}_{self.custom_parameters[n]}" for n in self.params_to_report])
        sim_name += f"_{seed}"
        return folder_root, sim_name

    def get_plot_title(self):
        return ", ".join([
            f"{n}={round(self.custom_parameters[n],2) if isinstance(self.custom_parameters[n], float) else self.custom_parameters[n]}"
            for n in self.params_to_report
        ])

    # ----- Time-varying params (applied during simulation loop) -----
    def reset_varying_params(self):
        for name in self.special_parameters:
            self.custom_parameters[name].reset()

    def apply_varying_params(self, model, t):
        for name in self.special_parameters:
            setattr(model, name, np.array(self.custom_parameters[name].get(t=t)))

    def __str__(self):
        return (f"run_sim={self.run_sim}, cut_transient={self.cut_transient}\n"
                f"custom={self.custom_parameters}\n"
                f"special={self.special_parameters}")


# ---------------------------------------------------------------------------
# 3) Connectivity utilities: subjects & averaging
# ---------------------------------------------------------------------------
SC_CSV_DEFAULT = "data/organised/dataframes/SC_CNT.csv"
TL_CSV_DEFAULT = "data/organised/dataframes/TL_CNT.csv"


def list_available_subjects(sc_csv: str = SC_CSV_DEFAULT):
    """Return sorted list of subject IDs present in the SC CSV."""
    return sorted(subjects_in_edge_csv(sc_csv))


def _average_connectivity_over_subjects(subjects, sc_csv=SC_CSV_DEFAULT, tl_csv=TL_CSV_DEFAULT, symmetric=True):
    """
    Average (elementwise mean) SC and TL over a set of subjects.

    Returns
    -------
    connection : lab.connectivity.Connectivity
        TVB Connectivity with averaged weights and tract_lengths.
    """
    assert len(subjects) > 0, "No subjects provided for averaging."

    conns = []
    for s in subjects:
        conn = load_connectivity_from_edge_csvs(sc_csv, tl_csv, subject=s, symmetric=symmetric)
        conns.append(conn)

    # Use the first as shape reference
    W = np.stack([c.weights for c in conns], axis=0)           # (S, N, N)
    TL = np.stack([c.tract_lengths for c in conns], axis=0)    # (S, N, N)
    Wm = np.nanmean(W, axis=0)
    TLm = np.nanmean(TL, axis=0)

    # Build averaged Connectivity
    avg_conn = lab.connectivity.Connectivity(
        weights=Wm,
        tract_lengths=TLm,
        region_labels=conns[0].region_labels,
        centres=getattr(conns[0], "centres", None),
        cortical=getattr(conns[0], "cortical", None),
    )
    return avg_conn


# ---------------------------------------------------------------------------
# 4) Simulator factory: subject-specific by default, optional averaging
# ---------------------------------------------------------------------------
def sim_init(
    simconfig: SimConfig,
    initial_condition=None,
    seed: int = 10,
    subject: str = "sub-01",
    symmetric: bool = True,
    average_subjects: bool = False,
    subjects_to_average: list | None = None,
):
    """
    Build and return a configured TVB Simulator.

    Connectivity behavior
    ---------------------
    - By default, loads SC/TL for a SINGLE `subject` from CSVs (subject-specific runs).
    - If `average_subjects=True`, averages SC/TL across all available subjects
      (or across `subjects_to_average` if provided).

    Parameters
    ----------
    simconfig : SimConfig
        Configuration holder with Parameters and overrides applied.
    initial_condition : dict | None
        Optional initial state dictionary.
    seed : int
        Random seed.
    subject : str
        Subject ID for subject-specific connectivity (e.g., 'sub-01').
    symmetric : bool
        If True, symmetrize SC (average with transpose + zero diagonal).
    average_subjects : bool
        If True, ignore `subject` and average SC/TL across subjects.
    subjects_to_average : list[str] | None
        Optional list of subject IDs to include in the average.

    Returns
    -------
    simulator : tvb.simulator.Simulator
        A ready-to-run simulator.
    """
    # ----- Extract parameter groups -----
    p = simconfig.general_parameters
    p_sim = p.parameter_simulation
    p_model = p.parameter_model
    p_conn = p.parameter_connection_between_region
    p_coupling = p.parameter_coupling
    p_integrator = p.parameter_integrator
    p_monitor = p.parameter_monitor
    p_stim = p.parameter_stimulus

    # ----- RNG seed -----
    p_sim["seed"] = seed
    Printer.print(f"Setting random seed: {seed}")
    rgn.seed(seed)

    # ----- Model selection (uses your merged zerlaut.py) -----
    from zerlaut import (
        Base_Zerlaut_adaptation_first_order,
        Base_Zerlaut_adaptation_second_order,
        GK_gNa_Zerlaut_adaptation_first_order,
        GK_gNa_Zerlaut_adaptation_second_order,
        Matteo_Zerlaut_adaptation_first_order,
        Matteo_Zerlaut_adaptation_second_order,
        Matteo_gK_gNa_Zerlaut_adaptation_first_order,
        Matteo_gK_gNa_Zerlaut_adaptation_second_order_gK_gNa,
    )

    if p_model["order"] == 1:
        if p_model["matteo"]:
            model_cls = Matteo_gK_gNa_Zerlaut_adaptation_first_order if p_model["gK_gNa"] \
                        else Matteo_Zerlaut_adaptation_first_order
        else:
            model_cls = GK_gNa_Zerlaut_adaptation_first_order if p_model["gK_gNa"] \
                        else Base_Zerlaut_adaptation_first_order
        voi = "E I W_e W_i noise".split()
    elif p_model["order"] == 2:
        if p_model["matteo"]:
            model_cls = Matteo_gK_gNa_Zerlaut_adaptation_second_order_gK_gNa if p_model["gK_gNa"] \
                        else Matteo_Zerlaut_adaptation_second_order
        else:
            model_cls = GK_gNa_Zerlaut_adaptation_second_order if p_model["gK_gNa"] \
                        else Base_Zerlaut_adaptation_second_order
        voi = "E I C_ee C_ei C_ii W_e W_i noise".split()
    else:
        raise ValueError("Model 'order' must be 1 or 2.")

    model = model_cls(variables_of_interest=voi)

    # Assign static model parameters (skip flags/initial_condition)
    for key, value in p_model.items():
        if key not in ("initial_condition", "matteo", "order", "gK_gNa"):
            try:
                setattr(model, key, np.array(value))
            except Exception:
                pass

    # Initial conditions
    for key, val in p_model["initial_condition"].items():
        model.state_variable_range[key] = val

    # ----- Connectivity: SUBJECT-SPECIFIC by default; averaging optional -----
    if average_subjects:
        if subjects_to_average is None:
            subjects_to_average = list_available_subjects(SC_CSV_DEFAULT)
        Printer.print(f"Averaging connectivity across {len(subjects_to_average)} subjects.")
        connection = _average_connectivity_over_subjects(
            subjects=subjects_to_average,
            sc_csv=SC_CSV_DEFAULT,
            tl_csv=TL_CSV_DEFAULT,
            symmetric=symmetric,
        )
    else:
        # Single participant (subject-specific)
        connection = load_connectivity_from_edge_csvs(
            sc_csv=SC_CSV_DEFAULT,
            tl_csv=TL_CSV_DEFAULT,
            subject=subject,
            symmetric=symmetric,
        )
        Printer.print(f"Loaded subject-specific connectivity for {subject}.")

    # Optional post-processing (consistent with your previous code)
    if p_conn.get("nullify_diagonals", True):
        np.fill_diagonal(connection.weights, 0.0)
    if p_conn.get("normalised", True):
        connection.weights /= (np.sum(connection.weights, axis=0) + 1e-12)
    connection.speed = np.array(p_conn["speed"])

    if p_conn.get("disconnect_regions"):
        dr = p_conn["disconnect_regions"]
        connection.weights[dr, :] = 0.0
        connection.weights[:, dr] = 0.0

    # ----- Stimulus -----
    if p_stim["stimval"] == 0.0:
        stimulation = None
    else:
        eqn_t = lab.equations.PulseTrain()
        eqn_t.parameters["onset"] = np.array(p_stim["stimtime"])
        eqn_t.parameters["tau"] = np.array(p_stim["stimdur"])
        eqn_t.parameters["T"] = np.array(p_stim["stimperiod"])
        weights = np.zeros(len(connection.weights))
        weights[list(p_stim["stimregion"])] = p_stim["stimval"]
        stimulation = lab.patterns.StimuliRegion(
            temporal=eqn_t, connectivity=connection, weight=weights
        )
        model.stvar = p_stim["stimvariables"]

    # ----- Coupling -----
    coupling = getattr(lab.coupling, p_coupling["type"])(
        **{k: np.array(v) for k, v in p_coupling["coupling_parameter"].items()}
    )

    # ----- Integrator & noise -----
    if not p_integrator["stochastic"]:
        integrator_cls = getattr(lab.integrators, f"{p_integrator['type']}Deterministic")
        integrator = integrator_cls(dt=np.array(p_integrator["dt"]))
    else:
        # Derive required noise shape: (n_vars, n_regions)
        n_vars = len(voi)
        try:
            n_regions = getattr(connection, "number_of_regions", None) or connection.weights.shape[0]
        except Exception:
            n_regions = connection.weights.shape[0]

        def _coerce_nsig(nsig_raw):
            a = np.asarray(nsig_raw, dtype=float)
            # Helper: map along variable names when nsig lists follow canonical 8-var order
            canonical8 = ['E','I','C_ee','C_ei','C_ii','W_e','W_i','noise']
            if a.ndim == 1 and a.shape[0] == 8 and n_vars != 8:
                # Reduce/select by variable names in 'voi'
                idxmap = {name: i for i, name in enumerate(canonical8)}
                try:
                    sel = np.array([a[idxmap[name]] for name in voi], dtype=float)
                    a = sel  # length n_vars
                except Exception:
                    pass
            if a.ndim == 2 and a.shape[0] == 8 and n_vars != 8:
                # Reduce/select rows, keep regions columns
                idxmap = {name: i for i, name in enumerate(canonical8)}
                try:
                    sel = np.stack([a[idxmap[name], :] for name in voi], axis=0)
                    a = sel  # (n_vars, maybe n_regions or 1)
                except Exception:
                    pass
            if a.ndim == 0:
                return np.full((n_vars, n_regions), float(a))
            if a.ndim == 1:
                if a.shape[0] == n_vars:
                    return np.broadcast_to(a.reshape(n_vars, 1), (n_vars, n_regions))
                if a.shape[0] == n_regions:
                    return np.broadcast_to(a.reshape(1, n_regions), (n_vars, n_regions))
                # Heuristic: if still length 8 (order-2) but model needs fewer, try mapping again
                if a.shape[0] == 8 and n_vars in (5,):
                    canonical8 = ['E','I','C_ee','C_ei','C_ii','W_e','W_i','noise']
                    idxmap = {name: i for i, name in enumerate(canonical8)}
                    sel = np.array([a[idxmap[name]] for name in voi], dtype=float)
                    return np.broadcast_to(sel.reshape(n_vars, 1), (n_vars, n_regions))
            if a.ndim == 2:
                if a.shape == (n_vars, n_regions):
                    return a
                if a.shape == (n_vars, 1):
                    return np.broadcast_to(a, (n_vars, n_regions))
                if a.shape == (1, n_regions):
                    return np.broadcast_to(a, (n_vars, n_regions))
                if a.shape[0] == 8 and a.shape[1] in (1, n_regions):
                    canonical8 = ['E','I','C_ee','C_ei','C_ii','W_e','W_i','noise']
                    idxmap = {name: i for i, name in enumerate(canonical8)}
                    sel = np.stack([a[idxmap[name], :] for name in voi], axis=0)
                    if sel.shape[1] == 1:
                        return np.broadcast_to(sel, (n_vars, n_regions))
                    if sel.shape[1] == n_regions:
                        return sel
            raise ValueError(
                f"Noise nsig has shape {a.shape}, but requires broadcastable to (n_vars={n_vars}, n_regions={n_regions})."
            )

        nsig_broadcast = _coerce_nsig(p_integrator["noise_parameter"]["nsig"]) 

        noise = lab.noise.Additive(
            nsig=nsig_broadcast,
            ntau=p_integrator["noise_parameter"]["ntau"],
        )
        noise.random_stream.seed(seed)
        integrator_cls = getattr(lab.integrators, f"{p_integrator['type']}Stochastic")
        integrator = integrator_cls(noise=noise, dt=p_integrator["dt"])

    # ----- Monitors -----
    monitors = []
    if p_monitor["Raw"]:
        monitors.append(lab.monitors.RawVoi(
            variables_of_interest=np.array(p_monitor["parameter_Raw"]["variables_of_interest"])
        ))
    if p_monitor["TemporalAverage"]:
        monitors.append(lab.monitors.TemporalAverage(
            variables_of_interest=np.array(p_monitor["parameter_TemporalAverage"]["variables_of_interest"]),
            period=p_monitor["parameter_TemporalAverage"]["period"]
        ))
    if p_monitor["Bold"]:
        monitors.append(lab.monitors.Bold(
            variables_of_interest=np.array(p_monitor["parameter_Bold"]["variables_of_interest"]),
            period=p_monitor["parameter_Bold"]["period"]
        ))

    # ----- Simulator -----
    simulator = lab.simulator.Simulator(
        model=model,
        connectivity=connection,
        coupling=coupling,
        integrator=integrator,
        monitors=monitors,
        stimulus=stimulation,
        initial_conditions=initial_condition,
    )
    simulator.configure()
    return simulator
