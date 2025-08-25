import torch
import torch.nn as nn
import torch.nn.functional as F

from models.desire.utils.params import DESIREParams


class CVAEEncoder(nn.Module):
    def __init__(self, params: DESIREParams):
        super().__init__()
        _h = params.hidden_size * 2
        _l = params.latent_size

        self.fc         = nn.Linear(_h, _l)
        self.fc_means   = nn.Linear(_l, _l)
        self.fc_log_var = nn.Linear(_l, _l)

    def forward(self, x, label):
        x = torch.cat([x, label], dim=-1)
        h = F.relu(self.fc(x))
        mean = self.fc_means(h)
        log_var = self.fc_log_var(h)

        return mean, log_var