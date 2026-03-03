import torch

def denormalize_static(static):
    static = static.clone()
    static[:, 0] *= 100.0
    static[:, 1] *= 100.0
    static[:, 2] *= 60.0
    static[:, 3] *= 60.0
    return static


def build_rectangle(static):
    A = static[:, 0]
    B = static[:, 1]
    C = static[:, 2]
    D = static[:, 3]

    return torch.stack([
        torch.stack([-B, -D], dim=-1),
        torch.stack([-B,  C], dim=-1),
        torch.stack([ A,  C], dim=-1),
        torch.stack([ A, -D], dim=-1),
    ], dim=1)


def rotate_rectangle(rect, course_deg):
    theta = torch.deg2rad((90.0 - course_deg) % 360.0)
    c = torch.cos(theta)
    s = torch.sin(theta)

    R = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s,  c], dim=-1),
    ], dim=-2)

    return torch.einsum("pij,pkj->pki", R, rect)


def hull_distance_vertices(hull1, hull2):
    """
    hull1, hull2: (..., 4, 2)
    returns: (...)
    """
    diff = hull1.unsqueeze(-3) - hull2.unsqueeze(-4)
    return torch.norm(diff, dim=-1).amin(dim=-1).amin(dim=-1)

def hull_min_distance(hull1, hull2):
    """
    hull1, hull2: (P,4,2)
    Returns:
        dcpa (P,)
        Correctly returns 0 if hulls overlap.
    """

    # --------------------------------------------------
    # 1) Compute edge normals (SAT axes)
    # --------------------------------------------------
    def get_axes(hull):
        edges = hull[:, 1:] - hull[:, :-1]
        edges = torch.cat([edges, hull[:, :1] - hull[:, -1:]], dim=1)
        normals = torch.stack([-edges[..., 1], edges[..., 0]], dim=-1)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
        return normals

    axes1 = get_axes(hull1)
    axes2 = get_axes(hull2)
    axes = torch.cat([axes1, axes2], dim=1)  # (P,8,2)

    # --------------------------------------------------
    # 2) Project onto axes
    # --------------------------------------------------
    proj1 = torch.einsum("pak,pvk->pav", axes, hull1)
    proj2 = torch.einsum("pak,pvk->pav", axes, hull2)

    min1, max1 = proj1.min(dim=-1).values, proj1.max(dim=-1).values
    min2, max2 = proj2.min(dim=-1).values, proj2.max(dim=-1).values

    overlap = (max1 >= min2) & (max2 >= min1)
    collision_mask = overlap.all(dim=1)

    # --------------------------------------------------
    # 3) Vertex–vertex distance
    # --------------------------------------------------
    diff = hull1.unsqueeze(2) - hull2.unsqueeze(1)
    dist = torch.norm(diff, dim=-1).amin(dim=-1).amin(dim=-1)

    # --------------------------------------------------
    # 4) If overlapping → distance = 0
    # --------------------------------------------------
    dist = torch.where(collision_mask, torch.zeros_like(dist), dist)

    return dist


def shape_aware_cpa_and_min_dist(
    ego_pos,
    other_pos,
    ego_static,
    other_static,
    ego_course,
    other_course,
    dt=10.0,
    w_tcpa=600.0,
    w_dcpa=500.0,
):
    """
    Returns:
        risk (P,)
        dcpa_hull_at_tcpa (P,)
        min_hull_dist_over_horizon (P,)
    """

    # --------------------------------------------------
    # 1) Continuous center-based TCPA
    # --------------------------------------------------
    r0 = other_pos[:, 0] - ego_pos[:, 0]

    ego_vel = (ego_pos[:, 1] - ego_pos[:, 0]) / dt
    other_vel = (other_pos[:, 1] - other_pos[:, 0]) / dt

    v = other_vel - ego_vel
    v_norm_sq = (v ** 2).sum(dim=-1) + 1e-6

    tcpa = -(r0 * v).sum(dim=-1) / v_norm_sq
    horizon = dt * (ego_pos.shape[1] - 1)
    tcpa = torch.clamp(tcpa, 0.0, horizon)

    # --------------------------------------------------
    # 2) Move centers to CPA
    # --------------------------------------------------
    ego_cpa = ego_pos[:, 0] + ego_vel * tcpa.unsqueeze(-1)
    other_cpa = other_pos[:, 0] + other_vel * tcpa.unsqueeze(-1)

    # --------------------------------------------------
    # 3) Build hull geometry
    # --------------------------------------------------
    ego_static = denormalize_static(ego_static)
    other_static = denormalize_static(other_static)

    ego_rect = rotate_rectangle(
        build_rectangle(ego_static),
        ego_course
    )
    other_rect = rotate_rectangle(
        build_rectangle(other_static),
        other_course
    )

    # --------------------------------------------------
    # 4) Hull DCPA (at TCPA)
    # --------------------------------------------------
    ego_hull_cpa = ego_rect + ego_cpa.unsqueeze(1)
    other_hull_cpa = other_rect + other_cpa.unsqueeze(1)

    dcpa_hull = hull_min_distance(
        ego_hull_cpa,
        other_hull_cpa
    )

    # --------------------------------------------------
    # 5) True minimum hull distance over trajectory (SAT-correct)
    # --------------------------------------------------
    T = ego_pos.shape[1]

    ego_hull_all = ego_rect.unsqueeze(1) + ego_pos.unsqueeze(2)      # (P,T,4,2)
    other_hull_all = other_rect.unsqueeze(1) + other_pos.unsqueeze(2)

    P = ego_hull_all.shape[0]

    # Flatten pair-time dimension so we can reuse hull_min_distance
    ego_flat = ego_hull_all.reshape(P * T, 4, 2)
    other_flat = other_hull_all.reshape(P * T, 4, 2)

    # Compute SAT-based hull distance per timestep
    hull_dist_flat = hull_min_distance(ego_flat, other_flat)

    # Reshape back to (P,T)
    hull_dist_all = hull_dist_flat.view(P, T)

    # True minimum distance across trajectory
    min_hull_dist = hull_dist_all.amin(dim=1)

    # --------------------------------------------------
    # 6) Risk (uses DCPA at TCPA only)
    # --------------------------------------------------
    risk = (
        (1 - tcpa / w_tcpa).clamp(0, 1)
        * (1 - dcpa_hull / w_dcpa).clamp(0, 1)
    )

    return risk, min_hull_dist


# ============================================================
# BATCH COLLISION AGGREGATION
# ============================================================

def compute_batch_collision_risk(data, pred_abs):

    device = pred_abs.device
    batch_ids = data.batch
    ego_indices = data.is_ego.nonzero(as_tuple=True)[0]

    if ego_indices.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    pair_ego = []
    pair_other = []
    pair_ego_static = []
    pair_other_static = []
    pair_ego_course = []
    pair_other_course = []
    pair_graph_ids = []

    # ---------------------------------------------------------
    # Build ego–neighbor pairs
    # ---------------------------------------------------------
    for g, ego_node in enumerate(ego_indices):

        graph_nodes = (batch_ids == batch_ids[ego_node]).nonzero(as_tuple=True)[0]
        neighbors = graph_nodes[graph_nodes != ego_node]

        if neighbors.numel() == 0:
            continue

        other_future = data.y_all[neighbors, 0]
        ego_future = pred_abs[g].unsqueeze(0).expand_as(other_future)

        pair_ego.append(ego_future)
        pair_other.append(other_future)

        pair_ego_static.append(
            data.static[ego_node].unsqueeze(0).expand(len(neighbors), -1)
        )
        pair_other_static.append(data.static[neighbors])

        pair_ego_course.append(
            data.x_raw[ego_node, -1, 1].repeat(len(neighbors))
        )
        pair_other_course.append(
            data.x_raw[neighbors, -1, 1]
        )

        pair_graph_ids.append(
            torch.full((len(neighbors),), g, device=device)
        )

    if len(pair_ego) == 0:
        return 0.0, 0.0, 0.0, 0

    # ---------------------------------------------------------
    # Concatenate pairs
    # ---------------------------------------------------------
    ego_tensor = torch.cat(pair_ego, dim=0)
    other_tensor = torch.cat(pair_other, dim=0)
    ego_static = torch.cat(pair_ego_static, dim=0)
    other_static = torch.cat(pair_other_static, dim=0)
    ego_course = torch.cat(pair_ego_course, dim=0)
    other_course = torch.cat(pair_other_course, dim=0)
    graph_ids = torch.cat(pair_graph_ids, dim=0)

    # ---------------------------------------------------------
    # Compute CPA-based risk + hull distances
    # ---------------------------------------------------------
    risk, min_hull_dist = shape_aware_cpa_and_min_dist(
        ego_tensor,
        other_tensor,
        ego_static,
        other_static,
        ego_course,
        other_course,
    )

    # ---------------------------------------------------------
    # Aggregate per graph
    # ---------------------------------------------------------
    num_graphs = len(ego_indices)

    graph_risk = torch.zeros(num_graphs, device=device)
    graph_risk = graph_risk.scatter_reduce(
        0, graph_ids, risk, reduce="amax"
    )

    graph_min_dist = torch.full((num_graphs,), float("inf"), device=device)
    graph_min_dist = graph_min_dist.scatter_reduce(
        0, graph_ids, min_hull_dist, reduce="amin"
    )

    valid_graphs = torch.unique(graph_ids)

    graph_max_risk = graph_risk[valid_graphs]
    graph_min_dist = graph_min_dist[valid_graphs]
    collision_count = (graph_min_dist <= 0)

    return (
        graph_max_risk.sum(),
        graph_min_dist.sum(),
        collision_count.sum(),
        len(valid_graphs)
    )