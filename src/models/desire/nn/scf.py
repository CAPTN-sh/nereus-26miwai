import numpy as np
import torch
import torch.nn as nn

from src.lazy_loader.normalizer import TorchCoordsNormalizer
from src.models.desire.nn import SocialPool
from src.models.desire.utils import (
    SCFParams,
    SocialPoolingParams,
    get_fc_act,
    get_scene,
)


class SCF(nn.Module):
    def __init__(self, index, params: SCFParams, normalizer: TorchCoordsNormalizer):
        super(SCF, self).__init__()
        self.params = params
        self.index = index
        self.velocity_fc = get_fc_act(params.velocity_fc)
        self.sp_nn = SocialPool(index, SocialPoolingParams())
        self.normalizer = normalizer

    def forward(
        self, hidden, y_pred, y_pred_rel, velocity, scene, x_start, seq_start_end=None
    ):

        vel_out = self.velocity_fc(y_pred_rel)
        # print(y_pred.device,
        #       x_start.device,
        #       hidden.device,
        #       vel_out.device)

        scene_out = get_scene(scene, y_pred, self.params.scene_size, self.normalizer)

        # print("scene out device", scene_out.device)
        sp_out = self.sp_nn(y_pred, x_start, hidden, seq_start_end)
        # print("Shapes 1", y_pred.shape, x_start.shape, scene.shape)
        # print ("Shapes",
        #        sp_out.shape,
        #        vel_out.shape,
        #        scene_out.shape)
        return torch.cat((sp_out, vel_out, scene_out), 1)


if __name__ == "__main__":
    idx = 0
    num_agents = 4
    dimensions = 2
    length = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scf = SCF(idx, SCFParams()).to(device)
    print(scf)
    y = torch.randn(num_agents, dimensions)
    x_start = torch.randn(num_agents, dimensions).to(device)
    v = torch.Tensor(np.gradient(y, axis=0)).to(device)
    y = y.to(device)
    hidden = torch.randn(num_agents, 48).to(device)
    scene = torch.randn(32, 720 // 2, 576 // 2).to(device)

    m = scf(hidden, y, v, scene, x_start)
