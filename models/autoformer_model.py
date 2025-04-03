# models/autoformer_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# 1. Auto-Correlation Mechanism (Simplified)
################################################################################
class AutoCorrelationLayer(nn.Module):
    """
    A simplified version of the Auto-Correlation mechanism.
    The real Autoformer uses FFT-based correlation to capture
    periodic dependencies. This snippet is an illustrative version
    that demonstrates the structure but is not identical to the
    official code.
    """
    def __init__(self, d_model, n_heads):
        super(AutoCorrelationLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Projections for queries, keys, and values
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj   = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Final projection
        self.out_proj   = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [Batch, Time, D_model]
        Returns: same shape [Batch, Time, D_model]
        """
        B, T, D = x.shape
        
        # 1) Project inputs to Q, K, V
        Q = self.query_proj(x)  # [B, T, D]
        K = self.key_proj(x)    # [B, T, D]
        V = self.value_proj(x)  # [B, T, D]

        # 2) Reshape to (B, H, T, head_dim)
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1,2)  # [B, H, T, hd]
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1,2)  # [B, H, T, hd]
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1,2)  # [B, H, T, hd]

        # 3) Auto-Correlation idea (simplified):
        #    Instead of standard attention, we compute a correlation-like score
        #    between Q and K. In the real Autoformer, this is done via FFT.
        
        # naive "similarity" with broadcasting
        # Q and K each shape: [B, H, T, hd]
        # correlation dimension: T x T (per head)
        # We'll do a simple matmul-based correlation:
        scores = torch.einsum("bhth,bhTh->bhTT", Q, K) / (self.head_dim ** 0.5) 
        # ^ shape: [B, H, T, T]
        
        # 4) Softmax to get weights
        weights = F.softmax(scores, dim=-1)  # [B, H, T, T]

        # 5) Apply weights to V
        out = torch.einsum("bhTT,bhTh->bhth", weights, V)  # [B, H, T, hd]

        # 6) Reshape back to [B, T, D]
        out = out.transpose(1,2).contiguous().view(B, T, D)
        
        # 7) Final linear projection
        out = self.out_proj(out)  # [B, T, D]

        return out


################################################################################
# 2. Encoder & Decoder (Highly Simplified)
################################################################################
class AutoformerEncoderLayer(nn.Module):
    """
    A single encoder layer consisting of:
      - One AutoCorrelation layer
      - Feed-forward network
    """
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):
        super(AutoformerEncoderLayer, self).__init__()
        self.auto_corr = AutoCorrelationLayer(d_model, n_heads)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Auto-Correlation sub-layer
        new_x = self.auto_corr(x)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # Feed-forward sub-layer
        new_x = self.ff(x)
        x = x + self.dropout(new_x)
        x = self.norm2(x)

        return x


class AutoformerDecoderLayer(nn.Module):
    """
    A single decoder layer. In the official Autoformer, the decoder uses
    cross-auto-correlation with the encoder output. Here we show a simplified
    version that reuses the same auto-correlation mechanism, ignoring cross-attn.
    """
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):
        super(AutoformerDecoderLayer, self).__init__()
        self.self_auto_corr = AutoCorrelationLayer(d_model, n_heads)
        
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention would go here in a full sequence-to-sequence model
        # For demonstration, we'll skip it or treat it as a second correlation block:
        # self.cross_auto_corr = AutoCorrelationLayer(d_model, n_heads)
        # self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Auto-Correlation
        new_x = self.self_auto_corr(x)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # (Optional) Cross-Auto-Correlation would go here

        # Feed-forward
        new_x = self.ff(x)
        x = x + self.dropout(new_x)
        x = self.norm3(x)

        return x


##############################################################################
