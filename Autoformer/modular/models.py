# models.py

from typing import Dict, Any
import torch
import torch.nn as nn
from layers import SeriesDecomp, AutoCorrelation, AutoformerEncoderLayer, AutoformerDecoderLayer

class Autoformer(nn.Module):
    """
    Autoformer model for time series forecasting.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing model parameters.
    """
    def __init__(self, config: Dict[str, Any]):
        super(Autoformer, self).__init__()
        
        self.d = config['input_dim']
        self.d_model = config['model_dim']
        self.h = config['num_heads']
        self.c = config['autocorrelation_factor']
        self.kernel_size = config['kernel_size']
        self.N = config['num_encoder_layers']
        self.M = config['num_decoder_layers']
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.use_layer_norm = config.get('use_layer_norm', False)

        self.embed = nn.Linear(self.d, self.d_model)
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(self.d_model, self.h, self.c, self.kernel_size, self.dropout_rate)
            for _ in range(self.N)
        ])
        self.decoder_layers = nn.ModuleList([
            AutoformerDecoderLayer(self.d_model, self.h, self.c, self.kernel_size, self.dropout_rate)
            for _ in range(self.M)
        ])
        self.mlp = nn.Linear(self.d_model, self.d)
        self.series_decomp = SeriesDecomp(self.kernel_size)

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, X: torch.Tensor, I: int, O: int) -> torch.Tensor:
        """
        Forward pass of the Autoformer model.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, input_length, input_dim)
            I (int): Input length
            O (int): Prediction length

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, prediction_length, input_dim)
        """
        if X.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {X.dim()}D")
        if I <= 0 or O <= 0:
            raise ValueError(f"I and O must be positive integers, got I={I}, O={O}")

        B, _, _ = X.shape

        X_en_s, X_en_t = self.series_decomp(X[:, I//2:])

        X0 = torch.zeros(B, O, self.d, device=X.device)
        Xmean = X[:, I//2:].mean(dim=1, keepdim=True).repeat(1, O, 1)

        X_de_s = torch.cat([X_en_s, X0], dim=1)
        X_de_t = torch.cat([X_en_t, Xmean], dim=1)

        X_en_s = self.embed(X)

        for layer in self.encoder_layers:
            X_en_s = layer(X_en_s)
            if self.use_layer_norm:
                X_en_s = self.layer_norm(X_en_s)

        X_de_s = self.embed(X_de_s)

        for layer in self.decoder_layers:
            X_de_s, X_de_t = layer(X_de_s, X_en_s, X_de_t)
            if self.use_layer_norm:
                X_de_s = self.layer_norm(X_de_s)

        X_pred = self.mlp(X_de_s) + X_de_t

        return X_pred[:, I//2:I//2+O, :]