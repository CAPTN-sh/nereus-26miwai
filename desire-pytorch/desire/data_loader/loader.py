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
    loader_num_workers=8,
):
    dset = TrajectoryDataset(
        nodes_path,
        edges_path,
        normalizer,
        obs_len=obs_len,
        pred_len=pred_len,
        max_vessels=max_vessels,
    )

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_num_workers,
        collate_fn=seq_collate,
    )
    return dset, loader
