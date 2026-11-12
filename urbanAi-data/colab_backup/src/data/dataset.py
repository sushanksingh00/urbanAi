import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrafficDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Args:
            x: input sequence array (samples, history_steps, nodes, features)
            y: target sequence array (samples, horizon_steps, nodes, features)
        """
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

def get_dataloaders(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    """
    train_dataset = TrafficDataset(x_train, y_train)
    val_dataset = TrafficDataset(x_val, y_val)
    test_dataset = TrafficDataset(x_test, y_test)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
