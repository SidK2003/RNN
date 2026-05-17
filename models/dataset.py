"""
PyTorch Dataset for sequential stock data.

Think of this file as the "Data Feeder" for our Neural Network.
Neural Networks need data in very specific shapes (tensors). 
This file takes our long, flat Excel-like data and chops it into overlapping "windows" 
or "sequences" that the GRU (which processes time-series) can understand.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


class StockSequenceDataset(Dataset):
    """
    Creates sequences of (features, target) pairs from processed stock data.

    In simple words: 
    If our seq_length is 60 (approx 3 months of trading days), this class acts like a sliding window.
    It looks at days 0 to 59, and sets the "target" as what happened on day 60 (Did it go UP or DOWN?).
    Then it slides one day forward: looks at days 1 to 60, and targets day 61.
    
    This matches exactly how trading works in real life: you look at the past N days to predict tomorrow.

    Args:
        features: NumPy array of shape (num_days, num_features). The raw numbers.
        targets: NumPy array of shape (num_days,). The 1s (UP) and 0s (DOWN).
        seq_length: Number of consecutive days per sequence (our "lookback" window).
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: int):
        # We convert standard Numpy arrays into PyTorch Tensors.
        # Tensors are just arrays that can be processed by a GPU very quickly.
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_length = seq_length

        # Calculate how many full sliding windows we can make.
        # If we have 100 days of data and a window of 60 days, we can only make 41 windows.
        # (days 0-59, 1-60, ... 40-99)
        self.num_sequences = len(features) - seq_length + 1

        # Safety check: ensure we have enough data to form at least one sequence.
        if self.num_sequences <= 0:
            raise ValueError(
                f"Not enough data for sequences: {len(features)} rows, "
                f"seq_length={seq_length}. Need at least {seq_length + 1} rows."
            )

    def __len__(self) -> int:
        # PyTorch needs to know exactly how many items are in this dataset.
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is the magic function PyTorch calls when it says "Give me the 5th sample".
        It returns the X (the past 60 days of features) and the Y (the target prediction).
        """
        # Slice the features array to get exactly 'seq_length' days starting from 'idx'.
        # For example, if idx=0 and seq_length=60, this gets days 0 through 59.
        x = self.features[idx : idx + self.seq_length]

        # The target corresponds to the LAST day in our input window.
        # target[t] = 1 if close[t+1] > close[t]. 
        # So by looking at features up to day t, we are predicting the direction of day t+1.
        y = self.targets[idx + self.seq_length - 1]

        return x, y
