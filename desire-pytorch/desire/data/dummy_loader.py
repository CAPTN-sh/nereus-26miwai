from torch.utils.data import Dataset, DataLoader
import time
import os

class DummyDataset(Dataset):
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        time.sleep(0.1)
        print(f"PID {os.getpid()} loading index {idx}")
        return idx

dl = DataLoader(DummyDataset(), batch_size=16, num_workers=8)

for batch in dl:
    pass