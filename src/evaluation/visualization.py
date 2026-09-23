import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path

def plot_actual_vs_predicted(actual: np.ndarray, predicted: np.ndarray, node_idx: int = 0, feature_idx: int = 0, sample_idx: int = 0, save_path: str = None):
    """
    Plot actual vs predicted traffic for a specific node and sample across the prediction horizon.
    actual, predicted shape: (batch_size, horizon, nodes, features)
    """
    # If torch tensors, convert to numpy
    if isinstance(actual, torch.Tensor):
        actual = actual.numpy()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.numpy()
        
    act = actual[sample_idx, :, node_idx, feature_idx]
    pred = predicted[sample_idx, :, node_idx, feature_idx]
    
    plt.figure(figsize=(10, 5))
    plt.plot(act, label='Actual', marker='o')
    plt.plot(pred, label='Predicted', marker='x')
    plt.title(f'Actual vs Predicted Traffic (Node {node_idx})')
    plt.xlabel('Prediction Horizon Step')
    plt.ylabel('Traffic Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_error_by_horizon(horizon_metrics: dict, metric_name: str = 'MAE', save_path: str = None):
    """
    Plot how the error changes as the prediction horizon increases.
    horizon_metrics: dict from evaluate_model_comprehensive["horizon"]
    """
    errors = horizon_metrics.get(metric_name)
    if not errors:
        raise ValueError(f"Metric {metric_name} not found in horizon_metrics")
        
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-')
    plt.title(f'{metric_name} by Prediction Horizon')
    plt.xlabel('Horizon Step')
    plt.ylabel(metric_name)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_node_performance(node_mae: list, save_path: str = None):
    """
    Plot the distribution of MAE across all nodes.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(node_mae, bins=30, kde=True)
    plt.title('Distribution of MAE Across Sensors (Nodes)')
    plt.xlabel('MAE')
    plt.ylabel('Number of Sensors')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_history(history: dict, save_path: str = None):
    """
    Plot training and validation loss curves.
    """
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    plt.figure(figsize=(10, 5))
    if train_loss:
        plt.plot(train_loss, label='Train Loss')
    if val_loss:
        plt.plot(val_loss, label='Validation Loss')
        
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
