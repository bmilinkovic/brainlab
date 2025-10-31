# =========================
# tvb_adex/io/connectome.py
# =========================

from typing import Tuple, Optional, List
import numpy as np
import pandas as pd

try:
    from tvb.datatypes.connectivity import Connectivity
except Exception as e:
    Connectivity = None  # type: ignore
    _CONNECTIVITY_IMPORT_ERROR = e
else:
    _CONNECTIVITY_IMPORT_ERROR = None


def load_connectivity(h5_path: str) -> 'Connectivity':
    """Load a TVB Connectivity from a .zip/.h5 produced by TVB.

    Parameters
    ----------
    h5_path : str
        Path to TVB connectivity file (e.g., 'sub-01_connectivity.h5' or TVB .zip).

    Returns
    -------
    Connectivity
        TVB Connectivity with weights (coupling), tract lengths (mm), and region_labels.
    """
    if Connectivity is None:
        raise ImportError(
            "tvb-library not found. Install with `pip install tvb-library` or fix your env.\n"
            f"Original import error: {_CONNECTIVITY_IMPORT_ERROR}"
        )
    conn = Connectivity.from_file(h5_path)
    conn.configure()
    return conn


def build_connectivity_from_mats(weights: np.ndarray,
                                 tract_lengths_mm: np.ndarray,
                                 region_labels: Optional[np.ndarray] = None) -> 'Connectivity':
    """Construct a TVB Connectivity object from raw matrices.

    Shapes must be (n_regions, n_regions) for both matrices.
    """
    if Connectivity is None:
        raise ImportError(
            "tvb-library not found; cannot build Connectivity. Install tvb-library."
        )
    if weights.shape != tract_lengths_mm.shape:
        raise ValueError("weights and tract_lengths_mm must have the same shape")
    n = weights.shape[0]
    if region_labels is None:
        region_labels = np.array([f"R{i:02d}" for i in range(n)], dtype=object)

    conn = Connectivity()
    conn.weights = weights.astype(float)
    conn.tract_lengths = tract_lengths_mm.astype(float)
    # TVB requires fixed-width unicode dtype for labels (not object)
    conn.region_labels = np.asarray(region_labels, dtype="U128")
    # TVB requires region centres (x,y,z). Provide placeholders at origin.
    n_regions = weights.shape[0]
    conn.centres = np.zeros((n_regions, 3), dtype=float)
    conn.configure()
    return conn

def subjects_in_edge_csv(csv_path: str, subject_col: str = 'subject') -> List[str]:
    """Return sorted list of subject IDs present in an edge CSV (long format)."""
    df = pd.read_csv(csv_path, usecols=[subject_col])
    return sorted(df[subject_col].unique().tolist())


def _edge_csv_to_matrix(df: pd.DataFrame, subject: str, value_col: str,
                        r1_col: str = 'region1', r2_col: str = 'region2') -> np.ndarray:
    sub = df[df['subject'] == subject]
    if sub.empty:
        raise ValueError(f"Subject '{subject}' not found in dataframe.")
    n = int(max(sub[r1_col].max(), sub[r2_col].max())) + 1
    M = np.zeros((n, n), dtype=float)
    # Fast fill: pivot
    M[sub[r1_col].to_numpy(), sub[r2_col].to_numpy()] = sub[value_col].to_numpy(dtype=float)
    return M


def load_connectivity_from_edge_csvs(sc_csv: str,
                                     tl_csv: str,
                                     subject: str,
                                     sc_value_col: Optional[str] = None,
                                     tl_value_col: Optional[str] = None,
                                     symmetric: bool = False,
                                     region_labels: Optional[np.ndarray] = None) -> 'Connectivity':
    """Build a TVB Connectivity from *edge‑list* CSVs like your SC_CNT/TL_CNT files.

    Expected columns: ['subject','region1','region2', VALUE_COL].
    VALUE_COL is auto‑detected if not provided (the last non‑index column).
    """
    if Connectivity is None:
        raise ImportError(
            "tvb-library not found; cannot build Connectivity. Install tvb-library."
        )

    sc_df = pd.read_csv(sc_csv)
    tl_df = pd.read_csv(tl_csv)

    if sc_value_col is None:
        sc_value_col = [c for c in sc_df.columns if c not in ('subject','region1','region2')][0]
    if tl_value_col is None:
        tl_value_col = [c for c in tl_df.columns if c not in ('subject','region1','region2')][0]

    W = _edge_csv_to_matrix(sc_df, subject, sc_value_col)
    L = _edge_csv_to_matrix(tl_df, subject, tl_value_col)

    if symmetric:
        # Symmetrize and zero diagonal by convention
        W = 0.5 * (W + W.T)
        L = 0.5 * (L + L.T)
        np.fill_diagonal(W, 0.0)
        np.fill_diagonal(L, 0.0)

    if region_labels is None:
        n = W.shape[0]
        region_labels = np.array([f"R{i:02d}" for i in range(n)], dtype=object)

    return build_connectivity_from_mats(W, L, region_labels)