# models/tft_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# 1) Gated Residual Network (GRN)
###############################################################################
class GatedResidualNetwork(nn.Module):
    """
    A Gated Residual Network (GRN) as described in the TFT paper:
    - dense -> ELU -> dense
    - skip connection + gating mechanism

    GRN(x) = (Residual + GLU(FC2(ELU(FC1(x))))) layer-normed
    """
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Gating and residual
        self.fc_gate = nn.Linear(output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        self.activation = nn.ELU()

    def forward(self, x):
        # x shape: [Batch, ..., input_dim]
        residual = x
        
        # Step 1: FC -> ELU
        x = self.fc1(x)
        x = self.activation(x)
        
        # Step 2: FC -> dropout
        x = self.fc2(x)
        x = self.dropout(x)
        
        # Step 3: Gating mechanism (GLU)
        gating = self.fc_gate(x)
        gating = torch.sigmoid(gating)
        
        # Combine via Gated Linear Unit style gating
        x = x * gating
        
        # Residual connection
        x = x + residual
        
        # Layer norm
        x = self.layer_norm(x)
        
        return x


###############################################################################
# 2) Variable Selection Network (VSN)
###############################################################################
class VariableSelectionNetwork(nn.Module):
    """
    Learns soft weights for each input variable (static or time-varying).
    Each variable is passed through a GRN, then a softmax over them to 
    compute the importance of each variable in the "bundle".

    The output is a weighted combination of the variable embeddings.
    """
    def __init__(self, input_dim, num_vars, hidden_dim):
        super().__init__()
        # Each variable has its own GRN
        self.grn_list = nn.ModuleList([
            GatedResidualNetwork(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim)
            for _ in range(num_vars)
        ])
        # Final projection to a scalar "weight" for each variable
        self.weight_proj_list = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_vars)
        ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x shape: [Batch, Time, num_vars] (for time-varying) 
                 or [Batch, num_vars] (for static)
        We'll handle both by flattening if needed.
        """
        # We'll treat the last dimension as variables
        # For each var, pass its slice through a GRN
        # Then compute a weight, then apply softmax, then do a weighted sum.
        
        original_shape = x.shape
        
        # If 3D: [B, T, num_vars], we separate the variables along dim=-1
        # If 2D: [B, num_vars], we can treat T=1 conceptually
        if len(original_shape) == 3:
            B, T, num_vars = original_shape
            x_out = []
            weights = []
            for i in range(num_vars):
                # shape of var_i: [B, T]
                var_i = x[..., i].unsqueeze(-1)  # [B, T, 1]
                # pass var_i through GRN
                var_i = self.grn_list[i](var_i)
                # project to scalar weight
                w_i = self.weight_proj_list[i](var_i)  # [B, T, 1]
                
                x_out.append(var_i)
                weights.append(w_i)
            
            x_out = torch.cat(x_out, dim=-1)     # [B, T, num_vars * hidden_dim? Actually we used output_dim= input_dim?]
            weights = torch.cat(weights, dim=-1) # [B, T, num_vars]
            
            # compute normalized weights
            weights = self.softmax(weights)      # [B, T, num_vars]
            
            # Weighted sum over variables
            # But note that each var_i is shape [B, T, hidden_dim].
            # We'll gather them carefully:
            # Instead of combining along the hidden_dim dimension, 
            # let's do it per variable. We'll build a final representation 
            # of shape [B, T, hidden_dim].
            
            var_embeddings = []
            for i in range(num_vars):
                var_embeddings.append(x_out[..., i].unsqueeze(-1))  # [B, T, 1]
            
            var_embeddings = torch.cat(var_embeddings, dim=-1)  # [B, T, num_vars]
            
            # Weighted sum
            out = (var_embeddings * weights).sum(dim=-1, keepdim=True)  # [B, T, 1]
            # That is a simplified approach. In a more typical usage, you might keep 
            # a larger embedding dimension per variable. This is just for demonstration.
        
        else:  # 2D: [B, num_vars]
            B, num_vars = original_shape
            x_out = []
            weights = []
            for i in range(num_vars):
                var_i = x[..., i].unsqueeze(-1)  # [B, 1]
                var_i = self.grn_list[i](var_i)
                w_i = self.weight_proj_list[i](var_i)  # [B, 1]
                
                x_out.append(var_i)
                weights.append(w_i)
            
            x_out = torch.cat(x_out, dim=-1)     # [B, hidden_dim * num_vars] (conceptually)
            weights = torch.cat(weights, dim=-1) # [B, num_vars]
            weights = self.softmax(weights)      # [B, num_vars]
            
            # Weighted sum over variables
            # We'll do a simplified approach as above:
            # var_embeddings: each [B, hidden_dim]
            # For demonstration, we used output_dim = input_dim in GRN, so shape might mismatch.
            
            # Because we've used GatedResidualNetwork in a very simplified dimension scenario, 
            # we'll keep it consistent:
            var_embeddings = []
            hidden_dim = 1  # from how we constructed the GRNs
            for i in range(num_vars):
                var_i = x_out[..., i]  # [B]
                var_embeddings.append(var_i)
            var_embeddings = torch.stack(var_embeddings, dim=-1)  # [B, num_vars]
            
            out = (var_embeddings * weights).sum(dim=-1, keepdim=True)  # [B, 1]
        
        return out


###############################################################################
# 3) LSTM Encoder-Decoder
###############################################################################
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, x):
        """
        x shape: [B, T, input_dim]
        returns: output [B, T, hidden_dim], (h_n, c_n)
        """
        out, (h_n, c_n) = self.lstm(x)
        return out, (h_n, c_n)


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, x, hidden):
        """
        x shape: [B, T_dec, input_dim]
        hidden: (h, c), each [num_layers, B, hidden_dim]
        returns: out [B, T_dec, hidden_dim], (h_n, c_n)
        """
        out, (h_n, c_n) = self.lstm(x, hidden)
        return out, (h_n, c_n)


###############################################################################
# 4) Multi-Head Attention Fusion
###############################################################################
class TFTAttention(nn.Module):
    """
    Simple multi-head attention to fuse encoder & decoder outputs
    (the paper uses attention across time in the decoder stage).
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
    
    def forward(self, dec_out, enc_out):
        """
        dec_out: [B, T_dec, d_model]
        enc_out: [B, T_enc, d_model]
        returns: fused [B, T_dec, d_model]
        """
        # We let the queries = dec_out, keys=values=enc_out
        # so the decoder can attend over the encoder states
        out, _ = self.attn(query=dec_out, key=enc_out, value=enc_out)
        # Add residual
        out = out + dec_out
        return out


###############################################################################
# 5) Putting it all together in TFTModel
###############################################################################
class TFTModel(nn.Module):
    """
    A simplified Temporal Fusion Transformer-like architecture.
    Features:
     - Variable selection for static and time-varying inputs
     - LSTM encoder-decoder
     - Multi-head attention in the decoder ("fusion")
     - Gated Residual Networks throughout
    """
    def __init__(
        self,
        num_static_vars=2,
        num_time_varying_vars=3,
        hidden_dim=16,
        lstm_hidden_dim=16,
        n_heads=4,
        dropout=0.1,
        lstm_layers=1,
        out_dim=1
    ):
        super().__init__()
        
        # Variable selection networks
        self.static_vsn = VariableSelectionNetwork(input_dim=1, num_vars=num_static_vars, hidden_dim=hidden_dim)
        self.time_varying_vsn = VariableSelectionNetwork(input_dim=1, num_vars=num_time_varying_vars, hidden_dim=hidden_dim)
        
        # LSTM encoder-decoder
        self.encoder = LSTMEncoder(input_dim=1, hidden_dim=lstm_hidden_dim, num_layers=lstm_layers, dropout=dropout)
        self.decoder = LSTMDecoder(input_dim=1, hidden_dim=lstm_hidden_dim, num_layers=lstm_layers, dropout=dropout)
        
        # Attention fusion
        self.attention = TFTAttention(d_model=lstm_hidden_dim, n_heads=n_heads, dropout=dropout)
        
        # Final projection
        self.fc_out = nn.Linear(lstm_hidden_dim, out_dim)

    def forward(self, static_input, encoder_input, decoder_input):
        """
        static_input: [B, num_static_vars]
        encoder_input: [B, T_enc, num_time_varying_vars]
        decoder_input: [B, T_dec, num_time_varying_vars]

        returns: [B, T_dec, out_dim]
        """
        # 1) Variable selection for static
        static_embedding = self.static_vsn(static_input)  # shape [B, 1] (simplified)
        
        # 2) Variable selection for time-varying (enc)
        enc_var_selected = self.time_varying_vsn(encoder_input)  
        # shape [B, T_enc, 1] in this simplified approach
        
        # 3) Pass through LSTM encoder
        enc_out, (h_enc, c_enc) = self.encoder(enc_var_selected)
        # enc_out: [B, T_enc, lstm_hidden_dim]
        
        # 4) Variable selection for time-varying (dec)
        dec_var_selected = self.time_varying_vsn(decoder_input)
        # shape [B, T_dec, 1]
        
        # 5) Pass through LSTM decoder
        dec_out, (h_dec, c_dec) = self.decoder(dec_var_selected, (h_enc, c_enc))
        # dec_out: [B, T_dec, lstm_hidden_dim]
        
        # 6) Attention fusion over encoder+decoder
        fused_out = self.attention(dec_out, enc_out)  # [B, T_dec, lstm_hidden_dim]
        
        # 7) Project to final output
        out = self.fc_out(fused_out)  # [B, T_dec, out_dim]

        return out
