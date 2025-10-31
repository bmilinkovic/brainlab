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
