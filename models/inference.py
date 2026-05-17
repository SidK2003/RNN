"""
MC Dropout inference for the GRU+Attention model.

In simple words: This is how we actually *use* the model to make predictions.
Normally, AI is a "black box" that gives you a number and says "Trust me".
By using Monte Carlo (MC) Dropout, we ask the AI the same question 50 times.
Because we randomly turn off parts of its brain (Dropout) each time, it gives slightly different answers.
If the answers are 0.80, 0.81, 0.79... it is highly confident.
If the answers are 0.20, 0.90, 0.50... it is guessing (low confidence).
We pass this confidence to the Stage 2 RL Agent so it knows when to step back from the market.
"""

import torch
import numpy as np
from typing import Dict, Tuple

from models.gru_attention import GRUAttentionModel


def mc_dropout_predict(
    model: GRUAttentionModel,
    x: torch.Tensor,
    num_passes: int = 50,
) -> Dict[str, torch.Tensor]:
    """
    Run MC Dropout inference with T stochastic (random) forward passes.
    """
    # CRITICAL: model.train() normally means "we are training".
    # But here, we use it to force the Dropout layers to stay ACTIVE during prediction.
    # Without this, Dropout turns off during prediction, and we'd get the exact same answer 50 times.
    model.train()

    all_predictions = []

    # torch.no_grad() tells the GPU "Don't track memory for training, we are just predicting". 
    # This makes it run much faster and use less VRAM.
    with torch.no_grad():
        for _ in range(num_passes):
            pred = model(x)  # Ask the network for a prediction
            all_predictions.append(pred)

    # Stack the 50 predictions into a single grid of numbers
    all_predictions = torch.stack(all_predictions, dim=0)

    # Calculate the mathematical average (mean) of all 50 passes. This is our final UP probability.
    p_up = all_predictions.mean(dim=0)          
    
    # Calculate how spread out the answers were (Standard Deviation). 
    std = all_predictions.std(dim=0)            
    
    # Confidence is 1 minus the spread. Low spread = High confidence.
    confidence = 1.0 - std                       
    
    # Binary decision: If probability > 50%, call it UP (1). Else DOWN (0).
    direction = (p_up > 0.5).float()            

    return {
        "p_up": p_up,
        "confidence": confidence,
        "direction": direction,
        "std": std,
        "all_predictions": all_predictions,
    }


def mc_dropout_predict_with_attention(
    model: GRUAttentionModel,
    x: torch.Tensor,
    num_passes: int = 50,
) -> Dict[str, torch.Tensor]:
    """
    Exactly the same as above, but it also extracts the attention weights.
    We average the attention weights across all 50 passes so we can draw pretty charts
    showing exactly which days the AI was looking at.
    """
    model.train()

    all_predictions = []
    all_mha_weights = []
    all_pool_weights = []

    with torch.no_grad():
        for _ in range(num_passes):
            pred, mha_w, pool_w = model.get_attention_weights(x)
            all_predictions.append(pred)
            all_mha_weights.append(mha_w)
            all_pool_weights.append(pool_w)

    # Stack predictions: (num_passes, batch_size)
    all_predictions = torch.stack(all_predictions, dim=0)

    # Average attention weights across passes for stable visualization charts
    avg_mha_weights = torch.stack(all_mha_weights, dim=0).mean(dim=0)
    avg_pool_weights = torch.stack(all_pool_weights, dim=0).mean(dim=0)

    p_up = all_predictions.mean(dim=0)
    std = all_predictions.std(dim=0)
    confidence = 1.0 - std
    direction = (p_up > 0.5).float()

    return {
        "p_up": p_up,
        "confidence": confidence,
        "direction": direction,
        "std": std,
        "all_predictions": all_predictions,
        "mha_weights": avg_mha_weights,
        "pool_weights": avg_pool_weights,
    }


def load_trained_model(
    checkpoint_path: str,
    num_features: int,
    config: dict,
    device: torch.device,
) -> GRUAttentionModel:
    """
    Utility function to load a saved AI brain (weights) from a file on the hard drive.
    """
    model_cfg = config["model"]

    # Recreate the empty brain structure
    model = GRUAttentionModel(
        num_features=num_features,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
    ).to(device)

    # Load the "memories/weights" from the file and put them into the empty brain
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"  Loaded model from {checkpoint_path}")
    print(f"  Trained epoch: {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.6f}")

    return model
