from torch.utils.data import DataLoader
from desire.data_loader.trajectories import TrajectoryDataset, seq_collate


def data_loader(
    nodes_path,
    edges_path,
    normalizer,
    obs_len=8,
    pred_len=12,
    max_vessels=10,
    batch_size=20,
    num_workers=8,
    min_date=None,
    max_date=None,
):
    dset = TrajectoryDataset(
        nodes_path,
        edges_path,
        normalizer,
        obs_len=obs_len,
        pred_len=pred_len,
        max_vessels=max_vessels,
        min_date=min_date,
        max_date=max_date,
        num_workers=num_workers,
    )

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=seq_collate,
    )
    return dset, loader
