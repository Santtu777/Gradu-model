# models/informer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# 1) ProbSparse Self-Attention (Simplified)
###############################################################################
class ProbSparseSelfAttention(nn.Module):
    """
    A simplified version of the ProbSparse mechanism from Informer:
      - Instead of computing all query-key similarities (O(L^2)), 
        we sample top-u queries for more efficient attention.
      - This code is only a demonstration and doesn't exactly 
        replicate all the official optimization tricks or distillation steps.
    """
    def __init__(self, d_model, n_heads, top_factor=5, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.top_factor = top_factor  # factor to determine how many queries to keep

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj   = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj   = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [Batch, Length, d_model]
        Returns: same shape [Batch, Length, d_model]
        """
        B, L, D = x.shape
        assert D == self.d_model, "Input hidden dimension must match d_model."

        # 1) Project Q, K, V
        Q = self.query_proj(x)  # [B, L, D]
        K = self.key_proj(x)    # [B, L, D]
        V = self.value_proj(x)  # [B, L, D]

        # 2) Reshape to (B, n_heads, L, head_dim)
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, hd]
        K = K.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, hd]
        V = V.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, hd]

        # 3) Compute queries' magnitude for top-u selection
        #    We'll flatten Q to shape [B*H, L, hd], compute the norm, 
        #    then pick top-u queries for each head.
        BH, _, hd = Q.shape[0], Q.shape[2], Q.shape[3]
        
        # Flatten Q for top-u selection
        Q_ = Q.contiguous().view(B * self.n_heads, L, hd)  # [B*H, L, hd]
        # L2 norm along the head_dim
        query_norms = torch.norm(Q_, dim=2)  # [B*H, L]

        # Determine how many queries to keep
        top_u = max(1, L // self.top_factor)  # e.g., keep L/5 queries
        top_values, top_indices = torch.topk(query_norms, k=top_u, dim=-1)
        # top_indices: [B*H, top_u]

        # We'll gather Q, K, V based on these top_u queries
        # But for simplicity, let's skip the full grouping mechanism
        # and do a naive approach: we only attend from the top_u queries 
        # to all keys. This is not the exact official approach, 
        # but demonstrates the idea.

        # Gather the selected Q rows
        Q_selected = []
        for i in range(B * self.n_heads):
            idxs = top_indices[i]  # shape [top_u]
            Q_selected.append(Q_[i, idxs, :])  # [top_u, hd]
        Q_selected = torch.stack(Q_selected, dim=0)  # [B*H, top_u, hd]

        # We'll do full K, but we need to match shapes
        K_ = K.contiguous().view(B * self.n_heads, L, hd)  # [B*H, L, hd]
        V_ = V.contiguous().view(B * self.n_heads, L, hd)  # [B*H, L, hd]

        # 4) Compute attention scores with only the selected queries
        #    scores shape: [B*H, top_u, L]
        scores = torch.bmm(Q_selected, K_.transpose(1, 2)) / (hd ** 0.5)

        # 5) Softmax
        attn_weights = F.softmax(scores, dim=-1)  # [B*H, top_u, L]

        # 6) Apply to V
        #    out_selected shape: [B*H, top_u, hd]
        out_selected = torch.bmm(attn_weights, V_)

        # 7) We must "place back" the attention outputs for the queries 
        #    that weren't selected. 
        #    For simplicity in this example, 
        #    we can just put zeros for the others or replicate them.
        #    A more correct approach would do interpolation or 
        #    a separate pass for "low energy" queries.
        out_full = torch.zeros(B * self.n_heads, L, hd, device=x.device)
        
        for i in range(B * self.n_heads):
            idxs = top_indices[i]
            out_full[i, idxs, :] = out_selected[i]

        # 8) Reshape back to [B, H, L, hd], then [B, L, D]
        out_full = out_full.view(B, self.n_heads, L, hd).transpose(1, 2).contiguous()
        out = out_full.view(B, L, D)

        # 9) Final projection
        out = self.out_proj(out)
        return self.dropout(out)


###############################################################################
# 2) Encoder / Decoder Layers with Distilling
###############################################################################
class InformerEncoderLayer(nn.Module):
    """
    A single Informer encoder layer:
      - ProbSparse Self-Attention
      - Feed-forward network
      - Distilling mechanism (downsampling) is often done 
        *between* encoder layers to reduce sequence length
    """
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.attn = ProbSparseSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        new_x = self.attn(x)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # FFN
        new_x = self.ff(x)
        x = x + self.dropout(new_x)
        x = self.norm2(x)

        return x


class InformerEncoder(nn.Module):
    """
    Stacked encoder layers with optional Distilling (reducing length by 1/2).
    """
    def __init__(self, d_model, d_ff, n_heads, num_layers, dropout=0.1, distil=True):
        super().__init__()
        self.layers = nn.ModuleList([
            InformerEncoderLayer(d_model, d_ff, n_heads, dropout) 
            for _ in range(num_layers)
        ])
        self.distil = distil

    def forward(self, x):
        """
        x: [B, L, d_model]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Distilling: reduce sequence length by half
            if self.distil and i != len(self.layers) - 1:
                x = x[:, ::2, :]  # downsample the time dimension by factor of 2
        return x


class InformerDecoderLayer(nn.Module):
    """
    A single Informer decoder layer:
      - Self ProbSparse Attention
      - Cross Attention (in the full version, we attend to encoder output)
      - Feed-forward network
    """
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):
        super().__init__()
        # Self-attention
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention (simplified)
        self.cross_attn = ProbSparseSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out):
        # Self-attn
        new_x = self.self_attn(x)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # Cross-attn
        # For cross-attn in Informer, we typically pass `enc_out` as K, V, 
        # and the decoder states as Q. For simplicity, we do a naive approach:
        # We'll just treat enc_out appended or something. Let's do a quick hack:
        # (One typical approach is: x queries enc_out, i.e., Q=x, K=enc_out, V=enc_out.)
        # But we've built ProbSparseSelfAttention in a purely self-attention style.
        # We'll just reuse that with concatenation for demonstration. 
        # Real code is more intricate.
        combined = torch.cat([x, enc_out], dim=1)  # naive approach
        new_x = self.cross_attn(combined)
        # We'll only take the portion corresponding to x's length.
        x_len = x.size(1)
        new_x = new_x[:, :x_len, :]  
        
        x = x + self.dropout(new_x)
        x = self.norm2(x)

        # FFN
        new_x = self.ff(x)
        x = x + self.dropout(new_x)
        x = self.norm3(x)

        return x


class InformerDecoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            InformerDecoderLayer(d_model, d_ff, n_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return x


###############################################################################
# 3) Full Informer Model (Simplified)
###############################################################################
class InformerModel(nn.Module):
    def __init__(
        self,
        input_dim=1,     # dimension of input features
        d_model=64,
        d_ff=128,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        dropout=0.1,
        distil=True,
        out_dim=1        # number of output features (e.g., 1 for univariate forecast)
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.enc_input_proj = nn.Linear(input_dim, d_model)
        self.dec_input_proj = nn.Linear(input_dim, d_model)

        # Encoder
        self.encoder = InformerEncoder(
            d_model=d_model, 
            d_ff=d_ff, 
            n_heads=n_heads,
            num_layers=e_layers,
            dropout=dropout,
            distil=distil
        )

        # Decoder
        self.decoder = InformerDecoder(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            num_layers=d_layers,
            dropout=dropout
        )

        # Final projection
        self.proj_out = nn.Linear(d_model, out_dim)

    def forward(self, enc_inputs, dec_inputs):
        """
        enc_inputs: [B, L_enc, input_dim]
        dec_inputs: [B, L_dec, input_dim]
        Returns: [B, L_dec, out_dim]
        """
        # 1) Encode
        enc_x = self.enc_input_proj(enc_inputs)  # [B, L_enc, d_model]
        enc_out = self.encoder(enc_x)            # [B, L_enc//(2^(layers-1)), d_model] if distil=True

        # 2) Decode
        dec_x = self.dec_input_proj(dec_inputs)  # [B, L_dec, d_model]
        dec_out = self.decoder(dec_x, enc_out)   # [B, L_dec, d_model]

        # 3) Project to final output
        out = self.proj_out(dec_out)            # [B, L_dec, out_dim]
        return out
