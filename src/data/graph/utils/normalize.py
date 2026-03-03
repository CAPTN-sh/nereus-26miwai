import numpy as np

def norm_deg(df, col, cols):
    """Encode angular feature using sine/cosine and update feature list."""

    rad = np.deg2rad(df[col])
    df[col + "_sin"] = np.sin(rad)
    df[col + "_cos"] = np.cos(rad)
    cols.append(col + "_sin")
    cols.append(col + "_cos")
    cols.remove(col)
    return df, cols

def normalize_ship_db(ship_db, static_feat_cols):
    """One-hot encode ship type and scale vessel dimensions."""

    ship_db["sailing"] = ship_db["ship_group"] == "sailing"
    ship_db["cargo"] = ship_db["ship_group"] == "cargo"
    ship_db["passenger"] = ship_db["ship_group"] == "passenger"
    ship_db["other"] = ship_db["ship_group"] == "other"
    ship_db['to_bow'] /= 100
    ship_db['to_stern'] /= 100
    ship_db['to_port'] /= 60
    ship_db['to_starboard'] /= 60

    if "ship_group" in static_feat_cols:
        static_feat_cols += ["sailing", "cargo", "passenger", "other"]
        static_feat_cols.remove("ship_group")
    return ship_db, static_feat_cols


def normalize_nodes(nodes, node_feat_cols):
    """Scale dynamic node features and encode angular quantities."""

    nodes["rel_x"] /= 100.0
    nodes["rel_y"] /= 100.0
    nodes["speed"] /= 40.0
    nodes["acc"] /= 4.0

    nodes, node_feat_cols = norm_deg(nodes, 'course', node_feat_cols)
    nodes, node_feat_cols = norm_deg(nodes, 'angular_difference', node_feat_cols)

    return nodes, node_feat_cols


def normalize_edges(edges, edge_feat_cols, static_feat_cols, max_edge_dist):
    """Normalize interaction features and encode angular quantities."""

    edges["dist"] = np.log1p(edges["dist"]) / np.log1p(max_edge_dist)
    edges["rel_speed"] = edges["rel_speed"] / 40
    edges["tcpa"] = np.log1p(edges["tcpa"]) / np.log1p(3600)
    edges["dcpa"] = np.log1p(edges["dcpa"]) / np.log1p(2000)

    edges, edge_feat_cols = norm_deg(edges, 'course_diff', edge_feat_cols)
    edges, edge_feat_cols = norm_deg(edges, 'rel_bearing', edge_feat_cols)
    
    edge_feat_cols += static_feat_cols + [f"{c}_other" for c in static_feat_cols]

    return edges, edge_feat_cols