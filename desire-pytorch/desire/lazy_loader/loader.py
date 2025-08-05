from torch.utils.data.distributed import DistributedSampler
from desire.lazy_loader.trajectories import TrajectoryDataset, seq_collate


def lazy_loader(dset, rank, batch_size=20, num_workers=8):
    """
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
    """

    loader = DistributedSampler(
        dset=dset,
        rank=rank,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=seq_collate,
    )
    return dset, loader
