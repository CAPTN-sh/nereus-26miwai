class MetricAccumulator:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, values):
        self.total += values.sum().item()
        self.count += values.numel()

    def compute(self):
        return self.total / self.count