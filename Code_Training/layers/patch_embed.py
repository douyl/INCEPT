import torch
import torch.nn as nn
import logging
logger = logging.getLogger("incept")

    
class EEGResidualPatchEmbed(nn.Module):
    """
    Multi-modal EEG Patch Embedding using Constrained Residual Fusion.
    
    Design Philosophy: "Time is Master, Frequency is Helper"
    
    Structure:
    1. Time Branch: LayerNorm -> Linear (Main Stream)
    2. Freq Branch: LayerNorm -> Linear (Auxiliary Stream, simplified)
    3. Fusion: Time + alpha * Freq
    
    Constraints:
    - 'alpha' is initialized to 0.
    - 'alpha' is strictly bounded by max_freq_weight (e.g., 0.3) using tanh.
      This ensures frequency features can NEVER dominate the embedding.
    """

    def __init__(
        self,
        in_chans: int = 250,        # Time dimension T
        embed_dim: int = 768,       # Target embedding dimension
        flatten_embedding: bool = True,
        dropout: float = 0.0,       # Dropout on the final output
        max_freq_weight: float = 0.2 # [Modified] Maximum allowed weight for frequency
    ) -> None:
        super().__init__()

        logger.info(f"using Time-frequency Residual Fusion Patch Embedding (Init=0, Max={max_freq_weight})")

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.max_freq_weight = max_freq_weight
        
        self.freq_dim = in_chans // 2 + 1
        
        # --- 1. Time Domain Branch (The Main Highway) ---
        self.time_branch = nn.Sequential(
            nn.LayerNorm(in_chans),
            nn.Linear(in_chans, embed_dim)
        )
        
        # --- 2. Frequency Domain Branch (The Residual Path) ---
        self.freq_branch = nn.Sequential(
            nn.LayerNorm(self.freq_dim),
            nn.Linear(self.freq_dim, embed_dim)
        )

        # --- 3. Residual Weight Parameter ---
        # self.freq_weight = nn.Parameter(torch.zeros(1))
        self.freq_weight = nn.Parameter(torch.zeros(embed_dim))
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, N)
        """
        B, T, C, N = x.shape
        assert T == self.in_chans, f"Input time dim {T} != init dim {self.in_chans}"
        
        # Part A: Prepare Time Domain Features
        x_time = x.permute(0, 2, 3, 1).flatten(1, 2)  # (B, T, C, N) -> (B, C, N, T) -> (B, C*N, T)
        feat_time = self.time_branch(x_time)  # (B, C*N, D)
        

        # Part B: Prepare Frequency Domain Features
        x_fp32 = x.float()   # (B, T, C, N)
        # 1. Compute FFT along the Time axis (dim 1)
        x_fft = torch.fft.rfft(x_fp32, n=T, dim=1)  # (B, T, C, N) -> (B, Freq, C, N)
        
        # 2. Compute Amplitude Spectrum
        ampl = torch.abs(x_fft)  # (B, Freq, C, N)
        
        # [Modified] Added clamp for numerical stability before log
        ampl = torch.clamp(ampl, max=1e5) 

        # 3. Log Transform (Log-Norm) to compress dynamic range
        log_ampl = torch.log1p(ampl).to(dtype=x.dtype)  # x range [1e-5, 500] -> ln(1+x), range [0, 5+]
          
        # 4. Reshape to: (B, C*N, Freq)
        x_freq = log_ampl.permute(0, 2, 3, 1).flatten(1, 2)  # (B, Freq, C, N) -> (B, C, N, Freq) -> (B, C*N, Freq)
        
        # Process through Frequency Branch
        feat_freq = self.freq_branch(x_freq)  # (B, C*N, D)

        # --- Part C: Constrained Residual Fusion ---
        alpha = self.max_freq_weight * torch.tanh(self.freq_weight)
        
        # Output = Time + alpha * Freq
        out = feat_time + alpha * feat_freq  # (B, C*N, D)

        out = self.dropout(out)

        if not self.flatten_embedding:
            out = out.reshape(B, C, N, self.embed_dim)
            
        return out