# layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SeriesDecomp(nn.Module):
    """
    Series Decomposition layer.

    Args:
        kernel_size (int): Size of the moving average kernel.
    """
    def __init__(self, kernel_size: int):
        super(SeriesDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Series Decomposition layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Seasonal and trend components
        """
        x_t = self.avg_pool(x.transpose(1, 2)).transpose(1, 2)
        x_s = x - x_t
        return x_s, x_t

class AutoCorrelation(nn.Module):
    """
    AutoCorrelation layer.

    Args:
        d_model (int): Dimension of the model
        h (int): Number of attention heads
        c (int): Auto-correlation factor
    """
    def __init__(self, d_model: int, h: int, c: int):
        super(AutoCorrelation, self).__init__()
        self.d_model = d_model
        self.h = h
        self.c = c

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AutoCorrelation layer.

        Args:
            Q (torch.Tensor): Query tensor
            K (torch.Tensor): Key tensor
            V (torch.Tensor): Value tensor

        Returns:
            torch.Tensor: Output tensor
        """
        B, L, _ = Q.size()
        Q = self.q_proj(Q).view(B, L, self.h, -1).permute(0, 2, 1, 3)
        K = self.k_proj(K).view(B, L, self.h, -1).permute(0, 2, 1, 3)
        V = self.v_proj(V).view(B, L, self.h, -1).permute(0, 2, 1, 3)

        Q = torch.fft.rfft(Q, dim=2)
        K = torch.fft.rfft(K, dim=2)

        Corr = torch.fft.irfft(Q * K.conj(), dim=2)

        topk = int(self.c * torch.log(torch.tensor(L, dtype=torch.float32)))
        W_topk, I_topk = torch.topk(Corr, topk, dim=2)

        W_topk = F.softmax(W_topk, dim=2)

        Index = torch.arange(L).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, self.h, 1, 1).to(Q.device)
        V = V.repeat(1, 1, 2, 1)

        R = torch.zeros_like(V)
        for i in range(topk):
            R += W_topk[:, :, i, :].unsqueeze(2) * V.gather(2, (I_topk[:, :, i, :].unsqueeze(2) + Index).clamp(max=L-1))

        R = R.sum(dim=2)
        return R.permute(0, 2, 1, 3).contiguous().view(B, L, -1)

class AutoformerEncoderLayer(nn.Module):
    """
    Autoformer Encoder Layer.

    Args:
        d_model (int): Dimension of the model
        h (int): Number of attention heads
        c (int): Auto-correlation factor
        kernel_size (int): Kernel size for series decomposition
        dropout_rate (float): Dropout rate
    """
    def __init__(self, d_model: int, h: int, c: int, kernel_size: int, dropout_rate: float = 0.1):
        super(AutoformerEncoderLayer, self).__init__()
        self.series_decomp = SeriesDecomp(kernel_size)
        self.auto_correlation = AutoCorrelation(d_model, h, c)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Autoformer Encoder Layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x_s, _ = self.series_decomp(self.auto_correlation(x, x, x) + x)
        x_s, _ = self.series_decomp(self.dropout(self.feed_forward(x_s)) + x_s)
        return x_s

class AutoformerDecoderLayer(nn.Module):
    """
    Autoformer Decoder Layer.

    Args:
        d_model (int): Dimension of the model
        h (int): Number of attention heads
        c (int): Auto-correlation factor
        kernel_size (int): Kernel size for series decomposition
        dropout_rate (float): Dropout rate
    """
    def __init__(self, d_model: int, h: int, c: int, kernel_size: int, dropout_rate: float = 0.1):
        super(AutoformerDecoderLayer, self).__init__()
        self.series_decomp = SeriesDecomp(kernel_size)
        self.auto_correlation = AutoCorrelation(d_model, h, c)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Autoformer Decoder Layer.

        Args:
            x (torch.Tensor): Input tensor
            enc_output (torch.Tensor): Encoder output tensor
            x_t (torch.Tensor): Trend component tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Seasonal and trend components
        """
        s1, t1 = self.series_decomp(self.auto_correlation(x, x, x) + x)
        s2, t2 = self.series_decomp(self.auto_correlation(s1, enc_output, enc_output) + s1)
        s3, t3 = self.series_decomp(self.feed_forward(s2) + s2)
        
        t = x_t + self.dropout(t1) + self.dropout(t2) + self.dropout(t3)
        return s3, t