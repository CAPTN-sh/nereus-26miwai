from movingpandas import Trajectory
from typing import Iterable
from scipy.signal import savgol_filter
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve
import numpy as np


def smooth_col(
    trajectories: Iterable[Trajectory], col_name, savgol_len, whittaker_lambda
) -> Iterable[Trajectory]:
    """
    Smooths column using the average of
    Savitzky-Golay Filters and Whittaker-Henderson smoother
    """
    for traj in trajectories:
        savgol = savgol_filter(traj.df[col_name], window_length=savgol_len, polyorder=2)
        whittaker = _whittaker_smooth(traj.df[col_name], whittaker_lambda)
        traj.df[col_name] = (savgol + whittaker) / 2
    return trajectories


def _whittaker_smooth(values, y=50):
    """
    Smooths values using Whittaker–Henderson smoother
    https://pubs.acs.org/doi/10.1021/acsmeasuresciau.1c00054

    Args:
        values: values to smooth
        y: lambda for smoothing
    """
    m = len(values)
    E = identity(m)
    D = diags([1, -2, 1], [0, 1, 2], shape=(m - 2, m))
    P = y * (D.T @ D)

    # Ensure A is in CSR or CSC format
    A = (E + P).tocsc()

    z = spsolve(A, values)
    return z


def merge_cols(trajectories: Iterable[Trajectory], trigger_col, merge_cols):
    for traj in trajectories:
        trigger = traj.df[trigger_col]
        for col, values in merge_cols.items():
            col_true = traj.df[values[0]] if isinstance(values[0], str) else values[0]
            col_false = traj.df[values[1]] if isinstance(values[1], str) else values[1]
            traj.df[col] = np.where(trigger, col_true, col_false)
    return trajectories
