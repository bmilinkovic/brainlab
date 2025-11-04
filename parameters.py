#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parameters.py
=============
Default configuration for whole-brain simulations using AdEx mean-field models.

This module now contains two main parts:

SECTION 1:  Parameter class hierarchy
    - Static and time-varying parameter classes (Parameter, ListParameter,
      VaryingParameter, GaussianParameter, OrnsteinUhlenbeckParameter, etc.)
    - These provide dynamic behaviour for any model parameter via a unified .get(t) API.

SECTION 2:  Default Parameters class (AdEx mean-field configuration)
    - Stores all simulation, model, coupling, and monitor defaults.

It can be imported and used directly:

    from parameters import Parameters, GaussianParameter
    p = Parameters()
    p.parameter_model["external_input_ex_ex"] = GaussianParameter("ext", 1000, 0.0003, 1e-4)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import numpy as np
from abc import ABC
import collections.abc


# ---------------------------------------------------------------------------
# SECTION 1: PARAMETER CLASSES (STATIC + TIME-VARYING)
# ---------------------------------------------------------------------------
class Parameter(ABC):
    """
    Abstract base for all parameters (static or varying).

    Attributes
    ----------
    name : str
        Parameter key (must correspond to a model attribute).
    value : any
        Current value (scalar, list, or nested Parameter).

    Notes
    -----
    Provides a unified `.get(t)` interface so SimConfig can update models
    without caring whether a parameter is static or time-dependent.
    """
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def get(self, t):
        """Return parameter value at time t (static → just value)."""
        return self.value

    def __str__(self):
        return str(self.value) if self.value is not None else ""

    def __eq__(self, other):
        return isinstance(other, Parameter) and self.name == other.name and self.value == other.value


class ListParameter(Parameter):
    """
    A parameter represented by a list (or list of Parameters).

    Used when:
        - assigning per-region values, or
        - combining static and time-varying parameters in a sequence.

    Example
    -------
    >>> ListParameter("g", [0.2, 0.25, 0.22])
    """
    def __init__(self, name, value, custom_value_display="list"):
        assert isinstance(value, list), \
            "ListParameter requires a Python list (not numpy array)."
        super().__init__(name, value)
        self.custom_value_display = custom_value_display
        self.varying_parameters_idxs = [i for i, v in enumerate(self.value)
                                        if issubclass(type(v), Parameter)]

    def __str__(self):
        return self.custom_value_display

    def __round__(self):
        return self.custom_value_display

    def reset(self):
        for i in self.varying_parameters_idxs:
            self.value[i].reset()

    def get(self, t):
        v = self.value[:]  # shallow copy
        for i in self.varying_parameters_idxs:
            v[i] = v[i].get(t=t)
        return np.array(v)[:, np.newaxis]

    def __getitem__(self, idx):
        return self.value[idx]

    def __list__(self):
        return self.value


class VaryingParameter(Parameter, ABC):
    """
    Abstract class for parameters that vary through time.
    """
    def __init__(self, name, value=None, log_values=False):
        super().__init__(name, value)
        self.log_values = log_values
        if log_values:
            self.prev_values = []

    def get(self, t):
        v = self._compute_value(t)
        if self.log_values:
            self.prev_values.append(v)
        return v

    def _compute_value(self, t):
        raise NotImplementedError

    def reset(self):
        pass

    def __str__(self):
        return super().__str__()


class PureFunctionParameter(VaryingParameter):
    """
    Calls a user-supplied function f(self, t) each time .get(t) is called.
    """
    def __init__(self, name, log_values, func):
        super().__init__(name, log_values=log_values)
        self._compute_value = func


class PeriodicallyVaryingParameter(VaryingParameter, ABC):
    """
    Base class for parameters updated every `period` ms.
    Holds previous value between updates for efficiency.
    """
    def __init__(self, name, period, log_values=False):
        super().__init__(name, log_values=log_values)
        self.period = period
        self.count = 0

    @staticmethod
    def periodic(func):
        def wrapper(self, *args, **kwargs):
            if kwargs["t"] >= (self.count + 1) * self.period or not hasattr(self, "cur_value"):
                self.count += 1
                self.cur_value = func(self, *args, **kwargs)
                if self.log_values:
                    self.prev_values.append(self.cur_value)
            return self.cur_value
        return wrapper

    def __str__(self):
        return super().__str__() + f"period_{self.period}_"

    def reset(self):
        self.count = 0


class GaussianParameter(PeriodicallyVaryingParameter):
    """Sample from N(mean, std) every `period` ms."""
    def __init__(self, name, period, mean, std, force_non_negative=False, log_values=False):
        super().__init__(name, period, log_values=log_values)
        self.mean = mean
        self.std = std
        self.force_non_negative = force_non_negative

    @PeriodicallyVaryingParameter.periodic
    def get(self, t):
        v = np.random.normal(loc=self.mean, scale=self.std)
        return np.abs(v) if self.force_non_negative else v

    def __str__(self):
        return f"{self.name}_gaussian_{self.mean}_{self.std}"

    def __eq__(self, other):
        return (isinstance(other, GaussianParameter) and
                (self.mean, self.std, self.period) ==
                (other.mean, other.std, other.period))


class UniformParameter(PeriodicallyVaryingParameter):
    """Sample from Uniform(min, max) every `period` ms."""
    def __init__(self, name, period, min_, max_, log_values=False):
        super().__init__(name, period, log_values=log_values)
        self.min = min_
        self.max = max_

    @PeriodicallyVaryingParameter.periodic
    def get(self, t):
        return np.random.uniform(self.min, self.max)

    def __str__(self):
        return f"{self.name}_uniform_{self.min}_{self.max}"

    def __eq__(self, other):
        return (isinstance(other, UniformParameter) and
                (self.period, self.min, self.max) ==
                (other.period, other.min, other.max))


class OrnsteinUhlenbeckParameter(VaryingParameter):
    """
    Parameter following an Ornstein–Uhlenbeck (OU) stochastic process:

        x_{t+dt} = x_t + speed*(mean - x_t)*dt + volatility*sqrt(dt)*N(0,1)
    """
    def __init__(self, name, speed, mean, volatility, dt, x0=None, force_non_negative=False):
        super().__init__(name)
        self.speed = speed
        self.mean = mean
        self.volatility = volatility
        self.dt = dt
        self.x0 = mean if x0 is None else x0
        self.x = self.x0
        self.force_non_negative = force_non_negative

    def get(self, t):
        eps = np.random.normal(0, 1)
        new_x = self.x + self.speed * (self.mean - self.x) * self.dt + \
                self.volatility * np.sqrt(self.dt) * eps
        self.x = max(new_x, 0.0) if self.force_non_negative else new_x
        return self.x

    def reset(self):
        self.x = self.x0

    def __str__(self):
        return (f"{self.name}_OU_speed{self.speed}_mean{self.mean}_"
                f"vol{self.volatility}_x0{self.x0}")

    def __eq__(self, other):
        return (isinstance(other, OrnsteinUhlenbeckParameter) and
                (self.speed, self.mean, self.volatility, self.x0, self.dt) ==
                (other.speed, other.mean, other.volatility, other.x0, other.dt))


# ---------------------------------------------------------------------------
# SECTION 2: DEFAULT PARAMETERS FOR WHOLE-BRAIN ADEX SIMULATIONS
# ---------------------------------------------------------------------------

class Parameters:
    """
    Parameters for Whole-Brain Mean-Field Simulations Based on the AdEx Model
    =========================================================================

    This class defines all numerical and structural parameters used to simulate
    large-scale cortical dynamics using a mean-field reduction of Adaptive
    Exponential Integrate-and-Fire (AdEx) neuronal populations.

    Each brain region is represented as an interacting population of excitatory
    and inhibitory neurons, described by their mean firing rates, conductances,
    and adaptation states. Regions are coupled through anatomical connectivity.

    --------------------------------------------------------------------------
    Parameter groups
    --------------------------------------------------------------------------

    **parameter_simulation**
        General simulation control.
        - `path_result`: output directory for simulation results.
        - `seed`: random generator seed for reproducibility.
        - `save_time`: duration (ms) of simulation data per saved file.

    **parameter_model**
        Biophysical parameters of the AdEx mean-field model.

        Flags:
        - `matteo`: enable Matteo di Volo variant.
        - `gK_gNa`: include Na⁺/K⁺ conductance channels.
        - `order`: model expansion order (1 or 2).

        Cellular parameters:
        - `E_L_e`, `E_L_i`: resting potentials (mV).
        - `C_m`: membrane capacitance (pF).
        - `g_L`: leak conductance (nS).
        - `E_Na_e`, `E_Na_i`, `E_K_e`, `E_K_i`: sodium/potassium reversal potentials (mV).
        - `g_Na_e`, `g_Na_i`, `g_K_e`, `g_K_i`: Na⁺/K⁺ channel conductances (nS).

        Adaptation parameters:
        - `a_e`, `a_i`: subthreshold adaptation (nS).
        - `b_e`, `b_i`: spike-triggered adaptation increment (pA).
        - `tau_w_e`, `tau_w_i`: adaptation time constants (ms).

        Synaptic and connectivity parameters:
        - `E_e`, `E_i`: excitatory/inhibitory reversal potentials (mV).
        - `Q_e`, `Q_i`: quantal synaptic conductances (nS).
        - `tau_e_e`, `tau_e_i`, `tau_i`: synaptic decay constants (ms).
        - `N_tot`: number of neurons per population.
        - `p_connect_e`, `p_connect_i`: connection probabilities.
        - `g`: ratio of inhibitory to excitatory neurons.

        Threshold and nonlinearity:
        - `T`: scaling constant in the firing threshold polynomial.
        - `P_e`, `P_i`: polynomial coefficients for excitatory and inhibitory threshold fits.

        Noise and external drive:
        - `external_input_ex_ex`, etc.: mean background rates (kHz).
        - `tau_OU`: Ornstein–Uhlenbeck noise correlation time (ms).
        - `weight_noise`: noise amplitude.
        - `K_ext_e`, `K_ext_i`: number of external synapses to E/I neurons.

        Initial conditions:
        - `initial_condition`: starting values for E, I, covariance terms, adaptation, and noise.

    **parameter_connection_between_region**
        Structural connectivity between brain regions.
        - `from_file`: whether to load connectivity from disk.
        - `path`: directory containing connectivity data.
        - `conn_name`: filename (e.g. “connectivity_68.zip”).
        - `tract_lengths`, `weights`: connection matrices.
        - `speed`: conduction velocity (m/s).
        - `normalised`: normalise weights to unit sum.
        - `nullify_diagonals`: remove self-connections.
        - `disconnect_regions`: indices of regions to silence.

    **parameter_coupling**
        Inter-regional coupling model.
        - `type`: coupling function (‘Linear’, ‘Sigmoidal’, etc.)
        - `coupling_parameter`: dict of function-specific parameters.

    **parameter_integrator**
        Numerical integration settings.
        - `type`: integrator (‘Heun’ or ‘Euler’).
        - `stochastic`: enable stochastic integration.
        - `noise_type`: additive or multiplicative noise.
        - `noise_parameter`: noise properties (nsig, ntau, dt).
        - `dt`: integration step (ms).

    **parameter_monitor**
        Output variables and temporal downsampling.
        - `Raw`: record unfiltered variables.
        - `TemporalAverage`: record averaged dynamics.
        - `Bold`: compute synthetic BOLD signal.
        - `Ca`: record calcium-like slow variable.
        Each includes its own `parameter_*` dict with options.

    **parameter_stimulus**
        External perturbations or stimuli.
        - `stimtime`: onset (ms).
        - `stimdur`: duration (ms).
        - `stimperiod`: repetition interval (ms).
        - `stimregion`: region index (None = all).
        - `stimval`: amplitude.
        - `stimvariables`: variable indices to apply stimulation to.
    """

    def __init__(
        self,
        parameter_simulation=None,
        parameter_model=None,
        parameter_connection_between_region=None,
        parameter_coupling=None,
        parameter_integrator=None,
        parameter_monitor=None,
        parameter_stimulus=None,
        **kwargs,
    ):
        # Shortcut: user-supplied parameter dicts
        if parameter_simulation is not None:
            self.parameter_simulation = parameter_simulation
            self.parameter_model = parameter_model
            self.parameter_connection_between_region = parameter_connection_between_region
            self.parameter_coupling = parameter_coupling
            self.parameter_integrator = parameter_integrator
            self.parameter_monitor = parameter_monitor
            self.parameter_stimulus = parameter_stimulus
            return

        # ------------------------------------------------------------------
        # Defaults
        # ------------------------------------------------------------------
        here = os.path.dirname(os.path.abspath(__file__))

        self.parameter_simulation = {
            "path_result": "./result/synch/",
            "seed": 10,
            "save_time": 1000.0,
        }

        self.parameter_model = {
            "matteo": False,
            "gK_gNa": False,
            "order": 1,
            "inh_factor": 1.0,
            "E_Na_e": 50.0,
            "E_Na_i": 50.0,
            "E_K_e": -90.0,
            "E_K_i": -90.0,
            "g_L": 10.0,
            "g_K_e": 8.214285714285714,
            "g_Na_e": 1.7857142857142865,
            "g_K_i": 8.214285714285714,
            "g_Na_i": 1.7857142857142865,
            "E_L_e": -64.0,
            "E_L_i": -65.0,
            "C_m": 200.0,
            "b_e": 5.0,
            "a_e": 0.0,
            "b_i": 0.0,
            "a_i": 0.0,
            "tau_w_e": 500.0,
            "tau_w_i": 1.0,
            "E_e": 0.0,
            "E_i": -80.0,
            "Q_e": 1.5,
            "Q_i": 5.0,
            "tau_e_e": 5.0,
            "tau_e_i": 5.0,
            "tau_i": 5.0,
            "N_tot": 10000,
            "p_connect_e": 0.05,
            "p_connect_i": 0.05,
            "g": 0.2,
            "T": 20.0,
            "P_e": [
                -0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
                 0.00341614, -0.01156433, 0.00194753, 0.00274079, -0.01066769
            ],
            "P_i": [
                -0.05184978, 0.0061593, -0.01403522, 0.00166511, -0.0020559,
                 0.00318432, -0.03112775, 0.00656668, 0.00171829, -0.04516385
            ],
            "external_input_ex_ex": 0.315 * 1e-3,
            "external_input_ex_in": 0.0,
            "external_input_in_ex": 0.315 * 1e-3,
            "external_input_in_in": 0.0,
            "tau_OU": 5.0,
            "weight_noise": 1e-4,
            "K_ext_e": 400,
            "K_ext_i": 0,
            "initial_condition": {
                "E": [0.0, 0.0],
                "I": [0.0, 0.0],
                "C_ee": [0.0, 0.0],
                "C_ei": [0.0, 0.0],
                "C_ii": [0.0, 0.0],
                "W_e": [100.0, 100.0],
                "W_i": [0.0, 0.0],
                "noise": [0.0, 0.0],
            },
        }

        self.parameter_connection_between_region = {
            "default": False,
            "from_file": True,
            "from_h5": False,
            "path": os.path.normpath(os.path.join(here, "..", "..", "data", "connectivity")),
            "conn_name": "connectivity_68.zip",
            "number_of_regions": 0,
            "tract_lengths": [],
            "weights": [],
            "speed": 4.0,
            "normalised": True,
            "nullify_diagonals": True,
            "disconnect_regions": [],
        }

        self.parameter_coupling = {
            "type": "Linear",
            "coupling_parameter": {"a": 0.3, "b": 0.0},
        }

        self.parameter_integrator = {
            "type": "Heun",
            "stochastic": True,
            "noise_type": "Additive",
            "noise_parameter": {
                "nsig": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "ntau": 0.0,
                "dt": 0.1,
            },
            "dt": 0.1,
        }

        dt = self.parameter_integrator["dt"]

        self.parameter_monitor = {
            "Raw": True,
            "parameter_Raw": {"variables_of_interest": [0, 1]},
            "TemporalAverage": False,
            "parameter_TemporalAverage": {
                "variables_of_interest": [0, 1, 2, 3, 4, 5, 6, 7],
                "period": dt * 10.0,
            },
            "Bold": False,
            "parameter_Bold": {
                "variables_of_interest": [0],
                "period": dt * 20000.0,
            },
            "Ca": False,
            "parameter_Ca": {
                "variables_of_interest": [0, 1, 2],
                "tau_rise": 0.01,
                "tau_decay": 0.1,
            },
        }

        self.parameter_stimulus = {
            "stimtime": 99.0,
            "stimdur": 9.0,
            "stimperiod": 1e9,
            "stimregion": None,
            "stimval": 0.0,
            "stimvariables": [0],
        }

    def __eq__(self, other):
        if isinstance(other, Parameters):
            return (
                self.parameter_model == other.parameter_model
                and self.parameter_connection_between_region == other.parameter_connection_between_region
                and self.parameter_coupling == other.parameter_coupling
                and self.parameter_integrator == other.parameter_integrator
                and self.parameter_monitor == other.parameter_monitor
                and self.parameter_stimulus == other.parameter_stimulus
            )
        if isinstance(other, dict):
            return (
                dict_inclusion_except(
                    self.parameter_simulation,
                    other["parameter_simulation"],
                    ["path_result"],
                )
                and self.parameter_model == other["parameter_model"]
                and self.parameter_connection_between_region == other["parameter_connection_between_region"]
                and self.parameter_coupling == other["parameter_coupling"]
                and self.parameter_integrator == other["parameter_integrator"]
                and self.parameter_monitor == other["parameter_monitor"]
                and self.parameter_stimulus == other["parameter_stimulus"]
            )
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def dict_inclusion_except(d1, d2, exceptions):
    """Checks that all keys of d1 are in d2, ignoring keys listed in `exceptions`."""
    for k in d1:
        if k in exceptions:
            continue
        if k not in d2:
            return False
    return True
