"""
GRU + Multi-Head Attention + MC Dropout model for binary direction prediction.

In simple words: This is the brain of Stage 1 (The Analyst). 
1. GRU: Reads the 60 days of history sequentially, understanding trends over time.
2. Attention: Looks back at those 60 days and says "Which specific days were the most important?"
3. MC Dropout: Simulates uncertainty. It asks the network the same question 50 times to see if it's "sure" or "guessing".
4. Linear Head: Squashes all that complex thought into a single number: the probability the stock goes UP tomorrow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GRUAttentionModel(nn.Module):
    """
    The neural network architecture.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # --- GRU Encoder ---
        # The GRU (Gated Recurrent Unit) processes data step-by-step. 
        # It has "memory" of past days. 
        # batch_first=True just means our data is shaped like (Batch, Days, Features).
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- Multi-Head Self-Attention ---
        # While the GRU reads day-by-day, Attention can jump around and connect day 1 directly to day 59.
        # "Multi-Head" means it has 4 independent "brains" looking for different patterns.
        # (e.g., Head 1 looks at volume spikes, Head 2 looks at price drops).
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization smooths out the numbers so the network doesn't mathematically explode during training.
        self.layer_norm = nn.LayerNorm(hidden_size)

        # --- MC Dropout ---
        # Dropout randomly turns off neurons (brain cells) in the network.
        # Usually, this is just to prevent memorizing the training data.
        # But we keep it ON during inference (real trading) to measure uncertainty (Monte Carlo Dropout).
        self.mc_dropout1 = nn.Dropout(p=dropout)
        self.mc_dropout2 = nn.Dropout(p=dropout)

        # --- Classification Head ---
        # Final decision layers. Takes the complex 128-dimension thought and funnels it down to 64, then to 1 single output.
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        This defines how data flows through the network.
        Data (x) goes in -> GRU -> Attention -> Pooling -> Dropout -> Linear -> Output.
        """
        # 1. GRU encoding
        # gru_out is the "memory" state after reading the sequence.
        gru_out, _ = self.gru(x)

        # 2. Multi-Head Self-Attention
        # The network looks at its own GRU memory to find important connections.
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)

        # 3. Residual connection + Layer Normalization
        # A "residual connection" (adding gru_out + attn_out) is a trick to make training faster and more stable.
        attn_out = self.layer_norm(gru_out + attn_out)

        # 4. Attention-weighted pooling
        # We have 60 days of outputs, but we need 1 final decision.
        # Instead of just taking the last day, we calculate "weights" (importance) for every day.
        query = attn_out[:, -1:, :]  # Look at the very last day as our "query"
        energy = torch.bmm(query, attn_out.transpose(1, 2)) # Match the last day against all previous days
        pool_weights = F.softmax(energy, dim=-1)  # Convert matches into percentages (e.g., Day 5 is 20% important)

        # Multiply the days by their importance percentages to get one final "context" vector.
        context = torch.bmm(pool_weights, attn_out) 
        context = context.squeeze(1) 

        # 5. MC Dropout + Classification Head
        # Pass through the final linear layers with random neurons turned off (Dropout)
        out = self.mc_dropout1(context)
        out = F.relu(self.fc1(out))  # ReLU is an activation function that turns negative numbers to 0
        out = self.mc_dropout2(out)
        logits = self.fc2(out).squeeze(-1)  # Final raw number (called a logit)

        # If training with AMP (Automatic Mixed Precision - fast GPU mode), we need the raw logit.
        # If doing real predictions, we use Sigmoid to squash the logit into a percentage between 0% and 100% (Probability).
        if return_logits:
            return logits
        return torch.sigmoid(logits)

    def get_attention_weights(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as forward(), but it also hands back the internal "weights" 
        so we can visualize exactly which days the AI was looking at when it made its decision.
        """
        gru_out, _ = self.gru(x)

        # We specifically ask PyTorch to return the attention weights here (average_attn_weights=False)
        attn_out, mha_weights = self.attention(
            gru_out, gru_out, gru_out, average_attn_weights=False
        )

        attn_out = self.layer_norm(gru_out + attn_out)

        query = attn_out[:, -1:, :]
        energy = torch.bmm(query, attn_out.transpose(1, 2))
        pool_weights = F.softmax(energy, dim=-1).squeeze(1)

        context = torch.bmm(pool_weights.unsqueeze(1), attn_out).squeeze(1)

        out = self.mc_dropout1(context)
        out = F.relu(self.fc1(out))
        out = self.mc_dropout2(out)
        
        # Always return the 0-100% probability for visualizations
        prediction = torch.sigmoid(self.fc2(out)).squeeze(-1)

        return prediction, mha_weights, pool_weights
