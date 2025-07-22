from movingpandas import Trajectory
from typing import Iterable
from preprocessing.utils.pipeline.function_inport import import_from_string
from typing import List


def detect_outliers(
    trajectories: Iterable[Trajectory],
    in_col_name: str,
    value: float,
    operator: str,
    measures: List[str] = [],
    out_col_name: str = None,
) -> Iterable[Trajectory]:
    """
    Detects outlier by applying measures to the in_col_name
    and comparing to the given value throught the operator.
    The result is saved in the out_col_name, if non is specified the outliers are droped.

    Args:
        trajectories (Iterable[Trajectory]): Traj to apply the outlier detection on.
        in_col_name (str): Column to apply the outlier detection on.
        value (float): Threshold for outlier detection.
        operator (str): measure like operator.gt to compare with value (threshold).
        measures (List[str]): List of measures to apply on the culumn before the operator.
        out_col_name (str): Column to store if the row is an outlier.
                If no column is given the outliers are droped!

    Return:
        Iterable[Trajectory]: Trajectories with new column or droped outliers.
    """
    compare = import_from_string(f"operator.{operator}")

    for traj in trajectories:
        values = traj.df[in_col_name].copy()
        for measure in measures:
            values = import_from_string(measure)(values)
        drop = compare(values, value)
        if out_col_name is None:
            traj.df = traj.df[~drop].copy()
        else:
            traj.df[out_col_name] = drop
    return trajectories
