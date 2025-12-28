class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.bad_epochs = 0
        self.stop = False

    def step(self, value: float) -> bool:
        """Return True if should stop."""
        value = float(value)
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        self.stop = self.bad_epochs >= self.patience
        return self.stop