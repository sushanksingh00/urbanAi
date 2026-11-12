import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Any
from src.evaluation.metrics import masked_mae

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    num_epochs: int,
    device: torch.device,
    null_val: float = 0.0
) -> dict[str, list[float]]:
    """
    Standard training loop for traffic forecasting models.
    """
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item() * batch_x.size(0)
            
        train_loss_avg = train_loss_total / len(train_loader.dataset)
        history["train_loss"].append(train_loss_avg)
        
        model.eval()
        val_loss_total = 0.0
        val_mae_total = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                preds = model(batch_x)
                loss = loss_fn(preds, batch_y)
                mae = masked_mae(preds, batch_y, null_val=null_val)
                
                val_loss_total += loss.item() * batch_x.size(0)
                val_mae_total += mae.item() * batch_x.size(0)
                
        val_loss_avg = val_loss_total / len(val_loader.dataset)
        val_mae_avg = val_mae_total / len(val_loader.dataset)
        history["val_loss"].append(val_loss_avg)
        history["val_mae"].append(val_mae_avg)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss_avg:.4f} - Val Loss: {val_loss_avg:.4f} - Val MAE: {val_mae_avg:.4f}")
        
    return history
