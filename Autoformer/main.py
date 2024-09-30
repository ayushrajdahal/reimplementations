import torch
import torch.nn as nn
import torch.nn.functional as F

class SeriesDecomp(nn.Module):

    """
    Returns the trend and the seasonal components of the time series.
    """

    def __init__(self, kernel_size:int):
        super(SeriesDecomp, self).__init__()

        # keep the series length unchanged
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        # match the expected input format for the average pooling layer.
        x_t = self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)

        # x_s is the seasonal and x_t is the trend-cyclical component
        x_s = x - x_t
        return x_s, x_t
    
class CustomLayerNorm(nn.Module):

    """
    Custom Layer Normalization layer for seasonal part. Taken from the original implementation.
    """

    def __init__(self, channels:int):
        super(CustomLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)
    
    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.size(1), 1)
        return x_hat - bias

class AutoCorrelation(nn.Module):

    """
    Computes the auto-correlation of the input sequence.
    """

    def __init__(self, d_model:int, h:int, c:int, d_keys:int=None):
        super(AutoCorrelation, self).__init__()

        self.d_model = d_model # dimension of the hidden state.
        self.d_keys = d_keys or (d_model // h) # dimension of the keys and queries.
        self.h = h # number of attention heads.
        self.c = c # hyper-parameter for selecting the top-k autocorrelations.

        self.k_proj = nn.Linear(d_model, self.d_keys*h)
        self.v_proj = nn.Linear(d_model, d_model*h)
        self.q_proj = nn.Linear(d_model, self.d_keys*h)

    def forward(self, Q, K, V):
        B, L, _ = Q.size() # batch size, sequence length, and hidden dimension.
        _, S, _ = K.size() # sequence length of the key-value pairs.

        # reshape and permute for multi-headed attention
        Q = Q.view(B, L, self.h, self.d_model // self.h).permute(0, 2, 1, 3)
        K = K.view(B, L, self.h, self.d_model // self.h).permute(0, 2, 1, 3)
        V = V.view(B, L, self.h, self.d_model // self.h).permute(0, 2, 1, 3)

        # applies FFT to Q and K along the sequence dimension.
        Q = torch.fft.fft(Q, dim=2)
        K = torch.fft.fft(K, dim=2)

        # computes the autocorrelation using the inverse FFT.
        Corr = torch.fft.ifft(Q * torch.conj(K), dim=2).real

        # calculates the number of top-k autocorrelations to consider.
        topk = int(self.c * torch.log(torch.tensor(L, dtype=torch.float32)))

        # selects the top-k autocorrelations and their indices.
        W_topk, I_topk = torch.topk(Corr, topk, dim=2)

        # applies softmax to the top-k autocorrelations.
        W_topk = F.softmax(W_topk, dim=2)

        # creates an index tensor for aggregation.
        Index = torch.arange(L).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, self.h, 1, 1)

        # repeats V to match the dimensions for aggregation.
        V = V.repeat(1, 1, 2, 1)

        # initialize the result tensor R.
        R = torch.zeros_like(V)

        # Aggregates the similar sub-series using the top-k autocorrelations.
        for i in range(topk):
            R += W_topk[:, :, i, :].unsqueeze(2) * V.gather(2, (I_topk[:, :, i, :].unsqueeze(2) + Index).clamp(max=L-1)) # clamping the index values to not exceed the length

        # Sums the aggregated sub-series.
        R = R.sum(dim=2)

        # Permutes and reshapes R to match the original format.
        return R.permute(0, 2, 1, 3).contiguous().view(B, L, self.d_model)

class AutoformerEncoderLayer(nn.Module):

    """
    The encoder layer of the Autoformer.
    """

    def __init__(self, d_model:int, h:int, c:int, kernel_size:int, dropout_rate:float=0.1):
        super(AutoformerEncoderLayer, self).__init__()
    
        """
        d_model (int): The dimension of the hidden state.
        h (int): The number of attention heads.
        c (int): A hyper-parameter for selecting the top-k autocorrelations.
        kernel_size (int): The size of the moving average window.
        dropout_rate (float): The dropout rate.
        """

        self.series_decomp = SeriesDecomp(kernel_size)
        self.auto_correlation = AutoCorrelation(d_model, h, c)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.dropout = nn.Dropout(dropout_rate)

    # algo 1: lines 5 to 8
    def forward(self, x):
        x_s,_ = self.series_decomp(self.auto_correlation(x, x, x) + x)
        x_s, _ = self.series_decomp(self.dropout(self.feed_forward(x_s)) + x_s)

        return x_s

class AutoformerDecoderLayer(nn.Module):

    """
    The decoder layer of the Autoformer.
    """

    def __init__(self, d_model:int, h:int, c:int, kernel_size:int, dropout_rate:float=0.1):
        super(AutoformerDecoderLayer, self).__init__()

        """
        d_model (int): The dimension of the hidden state.
        h (int): The number of attention heads.
        c (int): A hyper-parameter for selecting the top-k autocorrelations.
        kernel_size (int): The size of the moving average window.
        dropout_rate (float): The dropout rate.
        """

        self.series_decomp = SeriesDecomp(kernel_size)
        self.auto_correlation = AutoCorrelation(d_model, h, c)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.mlp1 = nn.Linear(d_model, d_model)
        self.mlp2 = nn.Linear(d_model, d_model)
        self.mlp3 = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, x_t):

        """
        x: The input data.
        enc_output: The output of the encoder.
        x_t: The trend-cyclical component.
        """
        
        s1, t1 = self.series_decomp(self.auto_correlation(x, x, x) + x)
        s2, t2 = self.series_decomp(self.auto_correlation(s1, enc_output, enc_output) + s1)
        s3, t3 = self.series_decomp(self.feed_forward(s2) + s2)
        
        # DIFFERENT: in Liu's implementation, t1, t2, and t3 are directly summed, followed by MLP and dropout.
        t = x_t + self.dropout(self.mlp1(t1)) + self.dropout(self.mlp2(t2)) + self.dropout(self.mlp3(t3))

        # output final seasonal and aggregated trend-cyclical components.
        return s3, t


class Autoformer(nn.Module):

    """
    The Autoformer model.
    """

    def __init__(self, d:int, d_model:int, h:int, c:int, kernel_size:int, N:int, M:int, dropout_rate:float=0.1):
        super(Autoformer, self).__init__()

        """
        d (int): The input dimension.
        d_model (int): The dimension of the hidden state.
        h (int): The number of attention heads.
        c (int): A hyper-parameter for selecting the top-k autocorrelations.
        kernel_size (int): The size of the moving average window.
        N (int): The number of encoder layers.
        M (int): The number of decoder layers.
        dropout_rate (float): The dropout rate.
        """

        self.embed = nn.Linear(d, d_model) # linear layer to embed the input time series.
        self.encoder_layers = nn.ModuleList([AutoformerEncoderLayer(d_model, h, c, kernel_size, dropout_rate) for _ in range(N)])
        self.decoder_layers = nn.ModuleList([AutoformerDecoderLayer(d_model, h, c, kernel_size, dropout_rate) for _ in range(M)])
        self.mlp = nn.Linear(d_model, d) # projects the hidden state to the output dimension.

        self.N = N
        self.series_decomp = SeriesDecomp(kernel_size)

    def forward(self, X, I, O):

        """
        X (int): The input data.
        I (int): The input length.
        O (int): The prediction length.
        """

        B, _, d = X.shape

        # decompose the embedded input into seasonal and trend-cyclical components.
        X_en_s, X_en_t = self.series_decomp(X[:, I//2:])

        # Step 2: Prepare X0 and Xmean
        X0 = torch.zeros(X.size(0), O, d, device=X.device) # d could be replaced by self.d from init
        Xmean = X[:, I//2:].mean(dim=1, keepdim=True).repeat(1, O, 1)

        # Prepare decoder input
        X_de_s = torch.cat([X_en_s, X0], dim=1)
        X_de_t = torch.cat([X_en_t, Xmean], dim=1)

        X_en_s = self.embed(X)

        # Encoder
        for layer in self.encoder_layers:
            X_en_s = layer(X_en_s)

        X_de_s = self.embed(X_de_s)

        # Decoder
        for layer in self.decoder_layers:
            X_de_s, X_de_t = layer(X_de_s, X_en_s, X_de_t)

        # Final prediction
        X_pred = self.mlp(X_de_s) + X_de_t

        return X_pred[:, I//2:I//2+O, :] # before: [:, -O, :]


# Parameters
d = 10          # example input dim
d_model = 512   # dimension of the hidden state
h = 8           # number of attention heads
c = 2           # hyper-parameter for selecting the top-k autocorrelations
kernel_size = 25
N = 2           # number of encoder layers
M = 1           # number of decoder layers
d = 10          # Example input dimension
I = 100         # Example input length
O = 10          # Example prediction length
B = 32          # Example batch size

# Create model
model = Autoformer(d, d_model, h, c, kernel_size, N, M)

# Example input
X = torch.randn(B, I, d)  # I is the input length, d is the input dimension

# Forward pass
output = model(X, I, O)  # O is the prediction length

print(output)