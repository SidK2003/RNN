"""
PyTorch Dataset for sequential stock data.

Think of this file as the "Data Feeder" for our Neural Network.
Neural Networks need data in very specific shapes (tensors). 
This file takes our long, flat Excel-like data and chops it into overlapping "windows" 
or "sequences" that the GRU (which processes time-series) can understand.

Neutral-label filtering:
  Sequences whose target is -1.0 (neutral zone) are excluded from training.
  However, the underlying feature data stays contiguous — we don't delete rows.
  We only skip certain indices when sampling. This preserves the time-series
  continuity that the GRU needs (no gaps in the lookback window).
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


class StockSequenceDataset(Dataset):
    """
    Creates sequences of (features, target) pairs from processed stock data.

    In simple words: 
    If our seq_length is 20 (approx 1 month of trading days), this class acts like a sliding window.
    It looks at days 0 to 19, and sets the "target" as what happened on day 19's forward return.
    Then it slides one day forward: looks at days 1 to 20, and targets day 20's forward return.
    
    Sequences where the target is -1.0 (neutral / sideways move) are automatically
    excluded from training. The model only learns from clear UP or DOWN signals.

    Args:
        features: NumPy array of shape (num_days, num_features). The raw numbers.
        targets: NumPy array of shape (num_days,). Values: 1.0 (UP), 0.0 (DOWN), -1.0 (NEUTRAL).
        seq_length: Number of consecutive days per sequence (our "lookback" window).
        filter_neutrals: If True (default), exclude sequences with target == -1.0.
                         Set to False for inference where you want predictions for all dates.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: int,
                 filter_neutrals: bool = True):
        # We convert standard Numpy arrays into PyTorch Tensors.
        # Tensors are just arrays that can be processed by a GPU very quickly.
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_length = seq_length

        # Calculate how many full sliding windows we can make from the data.
        total_sequences = len(features) - seq_length + 1

        # Safety check: ensure we have enough data to form at least one sequence.
        if total_sequences <= 0:
            raise ValueError(
                f"Not enough data for sequences: {len(features)} rows, "
                f"seq_length={seq_length}. Need at least {seq_length + 1} rows."
            )

        # Build the list of valid sequence indices.
        # A sequence at index 'i' uses features[i : i+seq_length] and targets[i+seq_length-1].
        # We EXCLUDE sequences whose target is -1.0 (neutral zone) during training.
        # The feature data is NOT modified — all rows stay in place to keep the
        # sliding window contiguous (the GRU sees an unbroken time series).
        if filter_neutrals:
            self.valid_indices = [
                i for i in range(total_sequences)
                if targets[i + seq_length - 1] != -1.0
            ]
        else:
            # For inference: include all sequences (even neutrals)
            self.valid_indices = list(range(total_sequences))

        if len(self.valid_indices) == 0:
            raise ValueError(
                f"No valid sequences after filtering neutrals! "
                f"Total sequences: {total_sequences}, all were neutral."
            )

    def __len__(self) -> int:
        # PyTorch needs to know exactly how many items are in this dataset.
        # This is now the count of non-neutral sequences.
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is the magic function PyTorch calls when it says "Give me the 5th sample".
        It returns the X (the past N days of features) and the Y (the target prediction).
        """
        # Map the requested index to the actual position in the contiguous data.
        # For example, if valid_indices = [0, 1, 3, 5], then idx=2 maps to position 3.
        actual_idx = self.valid_indices[idx]

        # Slice the features array to get exactly 'seq_length' days starting from 'actual_idx'.
        x = self.features[actual_idx : actual_idx + self.seq_length]

        # The target corresponds to the LAST day in our input window.
        y = self.targets[actual_idx + self.seq_length - 1]

        return x, y
