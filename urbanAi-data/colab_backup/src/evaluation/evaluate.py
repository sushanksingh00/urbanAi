import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable
from .metrics import masked_mae, masked_rmse, masked_mape, masked_wape, masked_r2

def evaluate_model_comprehensive(model: nn.Module, loader: DataLoader, device: torch.device, null_val: float = 0.0, stgcn_cheb: torch.Tensor = None) -> Dict[str, Any]:
    """
    Evaluates a model across all metrics and computes errors per prediction horizon.
    If the model requires cheb_polynomials (STGCN), pass it to `stgcn_cheb`.
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if stgcn_cheb is not None:
                preds = model(batch_x, stgcn_cheb)
            else:
                preds = model(batch_x)
            
            all_preds.append(preds.cpu())
            all_labels.append(batch_y.cpu())
            
    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0) # (total_samples, horizon, nodes, features)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate overall metrics
    overall_mae = masked_mae(all_preds, all_labels, null_val).item()
    overall_rmse = masked_rmse(all_preds, all_labels, null_val).item()
    overall_mape = masked_mape(all_preds, all_labels, null_val).item()
    overall_wape = masked_wape(all_preds, all_labels, null_val).item()
    overall_r2 = masked_r2(all_preds, all_labels, null_val).item()
    
    # Calculate metrics per horizon step
    horizon_steps = all_labels.size(1)
    horizon_metrics = {"MAE": [], "RMSE": [], "MAPE": [], "WAPE": [], "R2": []}
    
    for t in range(horizon_steps):
        preds_t = all_preds[:, t, :, :]
        labels_t = all_labels[:, t, :, :]
        
        horizon_metrics["MAE"].append(masked_mae(preds_t, labels_t, null_val).item())
        horizon_metrics["RMSE"].append(masked_rmse(preds_t, labels_t, null_val).item())
        horizon_metrics["MAPE"].append(masked_mape(preds_t, labels_t, null_val).item())
        horizon_metrics["WAPE"].append(masked_wape(preds_t, labels_t, null_val).item())
        horizon_metrics["R2"].append(masked_r2(preds_t, labels_t, null_val).item())
        
    # Calculate per node MAE
    nodes = all_labels.size(2)
    node_mae = []
    for n in range(nodes):
        preds_n = all_preds[:, :, n, :]
        labels_n = all_labels[:, :, n, :]
        node_mae.append(masked_mae(preds_n, labels_n, null_val).item())
        
    return {
        "overall": {
            "MAE": overall_mae,
            "RMSE": overall_rmse,
            "MAPE": overall_mape,
            "WAPE": overall_wape,
            "R2": overall_r2
        },
        "horizon": horizon_metrics,
        "node_mae": node_mae,
        "raw_preds": all_preds,
        "raw_labels": all_labels
    }
