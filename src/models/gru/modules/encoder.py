from torch import nn


class GRUEncoder(nn.Module):
    """Single layer GRU encoder.
    """

    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.gru_cell = nn.GRUCell(
            input_size=input_size,
            hidden_size=self.hidden_size,
        )

    def forward(self, x, mask):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden_size)

        for t in range(T):
            h_new = self.gru_cell(x[:, t], h)
            m = mask[:, t].unsqueeze(-1).float()
            h = m * h_new + (1.0 - m) * h
        return h
