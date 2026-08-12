"""One-off script: are there real (ground-truth) hull collisions in the AIS dataset?

Reuses the shape-aware CPA/hull-overlap logic from eval/metrics/cpa.py, but applied to
the actual observed future trajectories (data.y_pos / data.y_all) instead of model
predictions -- i.e. checks the raw data for physically overlapping vessel hulls, not
model prediction quality. Not part of the eval pipeline.
"""
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from data.graph.build_dataloader import graph_loader
from eval.metrics.cpa import shape_aware_cpa_and_min_dist
from utils.config import DATA_FOLDER_PATH

device = torch.device("cuda:0")
torch.cuda.set_device(device)

REGION = "kiel"
SOURCE = "fh"
data_folder = DATA_FOLDER_PATH / f"ais/4_features/{SOURCE}_10/{REGION}"


def batch_collisions_with_ids(data, pred_abs):
    """Like compute_batch_collision_risk, but also returns which ego target_ids collided.

    Mirrors src/eval/metrics/cpa.py:compute_batch_collision_risk's pairing loop; g (the
    loop index over ego_indices) lines up 1:1 with data.target_id's batch order, since
    both follow the same per-graph batch ordering.
    """
    device = pred_abs.device
    batch_ids = data.batch
    ego_indices = data.is_ego.nonzero(as_tuple=True)[0]

    pair_ego, pair_other, pair_ego_static, pair_other_static = [], [], [], []
    pair_ego_course, pair_other_course, pair_graph_ids = [], [], []

    for g, ego_node in enumerate(ego_indices):
        graph_nodes = (batch_ids == batch_ids[ego_node]).nonzero(as_tuple=True)[0]
        neighbors = graph_nodes[graph_nodes != ego_node]
        if neighbors.numel() == 0:
            continue
        other_future = data.y_all[neighbors, 0]
        ego_future = pred_abs[g].unsqueeze(0).expand_as(other_future)
        pair_ego.append(ego_future)
        pair_other.append(other_future)
        pair_ego_static.append(data.static[ego_node].unsqueeze(0).expand(len(neighbors), -1))
        pair_other_static.append(data.static[neighbors])
        pair_ego_course.append(data.x_raw[ego_node, -1, 1].repeat(len(neighbors)))
        pair_other_course.append(data.x_raw[neighbors, -1, 1])
        pair_graph_ids.append(torch.full((len(neighbors),), g, device=device))

    if len(pair_ego) == 0:
        return [], [], torch.tensor([], device=device)

    risk, tcpa, dcpa, min_hull_dist = shape_aware_cpa_and_min_dist(
        torch.cat(pair_ego), torch.cat(pair_other),
        torch.cat(pair_ego_static), torch.cat(pair_other_static),
        torch.cat(pair_ego_course), torch.cat(pair_other_course),
    )
    graph_ids = torch.cat(pair_graph_ids)

    num_graphs = len(ego_indices)
    graph_min_dist = torch.full((num_graphs,), float("inf"), device=device)
    graph_min_dist = graph_min_dist.scatter_reduce(0, graph_ids, min_hull_dist, reduce="amin")
    valid_graphs = torch.unique(graph_ids)
    graph_min_dist = graph_min_dist[valid_graphs]  # aligned with valid_graphs (== g values)

    collided_g = valid_graphs[graph_min_dist <= 0]
    collided_target_ids = data.target_id[collided_g].tolist()
    # x_raw node_feat_cols = [speed, course, acc, angular_difference]; last observed step.
    collided_ego_speed = data.x_raw[ego_indices[collided_g], -1, 0].tolist()
    return collided_target_ids, collided_ego_speed, graph_min_dist


def check_split(flag, batch_size=512):
    loader, dset = graph_loader(
        data_folder=data_folder,
        flag=flag,
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        batch_size=batch_size,
        pin_memory=True,
        pred_len=30,
        obs_len=60,
        max_edge_dist=500,
        shuffle=False,
        ship_group="all",
    )

    import os
    max_batches = int(os.environ.get("MAX_BATCHES", "0")) or None

    total_graphs = 0
    total_collisions = 0
    total_min_dist = 0.0
    moving_collisions = 0  # ego speed > 0.5 kn at the observed moment (i.e. not moored/anchored)
    examples = []

    for bi, data in enumerate(tqdm(loader, desc=f"{flag}: scanning for real hull collisions", total=max_batches or None)):
        if max_batches and bi >= max_batches:
            break
        data = data.to(device, non_blocking=True)
        ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
        # Ground-truth ego future positions, in place of model predictions.
        gt_abs_pos = data.y_pos

        collided_target_ids, collided_ego_speed, graph_min_dist = batch_collisions_with_ids(data, gt_abs_pos)
        n_graphs = graph_min_dist.numel()
        collision_count = len(collided_target_ids)
        total_graphs += n_graphs
        total_collisions += collision_count
        total_min_dist += float(graph_min_dist[torch.isfinite(graph_min_dist)].sum()) if n_graphs else 0.0
        moving_collisions += sum(1 for s in collided_ego_speed if s > 0.5)

        if collided_target_ids and len(examples) < 20:
            examples.extend(list(zip(collided_target_ids[:5], collided_ego_speed[:5])))

    return {
        "split": flag,
        "n_graphs": total_graphs,
        "n_collisions": total_collisions,
        "n_moving_collisions_ego_speed_gt_0.5kn": moving_collisions,
        "collision_ratio_pct": 100 * total_collisions / max(1, total_graphs),
        "mean_min_hull_dist_m": total_min_dist / max(1, total_graphs),
        "example_target_ids": examples,
    }


if __name__ == "__main__":
    splits = sys.argv[1:] or ["test"]
    results = [check_split(s) for s in splits]
    for r in results:
        print(r)
