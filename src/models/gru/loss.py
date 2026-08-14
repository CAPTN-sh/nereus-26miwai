def mse_loss(pred_rel, data, config = None):
    """Mean squared error - loss function
    """
    fut_rel = data.y_rel_pos
    fut_mask = data.y_mask

    err = (pred_rel - fut_rel).pow(2).sum(dim=-1)
    err = err * fut_mask

    mse = err.sum() / fut_mask.sum().clamp_min(1)

    return mse, {"mse": mse}
