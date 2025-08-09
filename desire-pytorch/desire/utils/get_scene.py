import torch
from desire.utils.normalizer import TorchCoordsNormalizer


def get_scene(scene, ypred, scene_size, normalizer: TorchCoordsNormalizer):
    """get_scene
    input
    =====
    scene: (x, W/2, H/2, x)
    ypred: (x, y) where x, y are floats
    output:
    """

    width = scene_size[0]
    height = scene_size[1]
    shrinkage = scene_size[2]

    # TODO range in setting/config
    lat_min, lat_max = 54.31, 54.46
    lon_min, lon_max = 10.13, 10.32

    lat, lon = normalizer.denormalize_coords(ypred).T

    x_px = (((lon - lon_min) / (lon_max - lon_min)) * width).long()
    y_px = (((lat_max - lat) / (lat_max - lat_min)) * height).long()

    x_shrunk = torch.clamp(x_px // shrinkage, 0, scene.shape[1] - 1)
    y_shrunk = torch.clamp(y_px // shrinkage, 0, scene.shape[2] - 1)
    return scene[:, y_shrunk, x_shrunk].transpose(0, 1)
