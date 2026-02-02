# %% [code]
import numpy as np
import torch
import torch.nn as nn
import einops

class TemporalBlock(nn.Module):
    """
    A specific "1-D Conv Block" used in the separation network of Conv-TasNet.
    
    It employs a depthwise separable convolution structure with dilation to increase 
    the receptive field without significantly increasing computational cost.
    
    Structure:
        [1x1-conv] -> [GELU/Norm] -> [D-Conv] -> [GELU/Norm] -> [1x1-convs]
    
    Args:
        in_channels (int): Number of input channels (typically B).
        out_channels (int): Number of internal channels for the depthwise block (typically H).
        kernel_size (int): Kernel size for the depthwise convolution (typically P).
        dilation (int): Dilation factor for the depthwise convolution (2**i).
        skip_channels (int): Number of channels for the skip-connection output (typically Sc).
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, skip_channels, dropout = 0.0):
        super().__init__()

        # Pointwise convolution (1x1): Projects input to higher dimensional space (H)
        self.pw_conv = nn.Conv1d(in_channels, out_channels, kernel_size = 1)

        # Depthwise convolution: Spatial (temporal) filtering
        # Groups = out_channels ensures each channel is convolved independently (depthwise).
        self.d_conv = nn.Conv1d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            dilation = dilation,
            padding = (kernel_size - 1) * dilation // 2, # Padding is calculated to keep the output length the same as input length.
            groups = out_channels
        )

        # Residual path: Added back to input to flow to the next block.
        self.res_conv = nn.Conv1d(out_channels, in_channels, kernel_size = 1)
        # Skip path: Sent directly to the final summation at the end of the separation net.
        self.skip_conv = nn.Conv1d(out_channels, skip_channels, kernel_size = 1)

        # Normalization and Activation
        # Conv-TasNet uses Global Layer Norm (gLN) or Cumulative Layer Norm (cLN).
        # Here we use GroupNorm(1, ...) which is equivalent to LayerNorm in PyTorch for 1D data.
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(num_groups = 1, num_channels = out_channels)
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(num_groups = 1, num_channels = out_channels)

        # Dropout
        self.dropout = nn.Dropout(p = dropout)
                
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, B, Time)
            
        Returns:
            residual (torch.Tensor): Tensor to be fed to the next block (Batch, B, Time).
            skip (torch.Tensor): Tensor to be collected for global sum (Batch, Sc, Time).
        """
        # Expand channels: (Batch, B, L) -> (Batch, H, L)
        x_hat = self.pw_conv(x)
        x_hat = self.norm1(self.gelu1(x_hat))

        # Apply dilated depthwise convolution: (Batch, H, L) -> (Batch, H, L)
        x_hat = self.d_conv(x_hat)
        x_hat = self.norm2(self.gelu2(x_hat))

        # Apply dropout
        x_hat = self.dropout(x_hat)

        # Residual connection: Mixes original input with processed output
        residual = x + self.res_conv(x_hat) 
        
        # Skip connection: Extracts features for the final mask construction
        skip = self.skip_conv(x_hat) 

        return residual, skip


class TemporalConvNet(nn.Module):
    """
    The main separation module consisting of stacked TemporalBlocks.
    
    This creates the "receptive field" necessary to model long-term dependencies
    in the audio signal by stacking dilated convolutions.
    
    Args:
        B (int): Number of channels in the bottleneck (input to this net).
        H (int): Number of channels in the convolutional blocks.
        Sc (int): Number of channels in the skip connections.
        kernel_size (int): Kernel size (P).
        X (int): Number of convolutional blocks in one repeat.
        R (int): Number of repeats.
    """
    def __init__(self, B, H, Sc, kernel_size, X, R, dropout = 0.0):
        super().__init__()
        
        # Create a list of blocks with increasing dilation factors
        # Loop R times (repeats), and inside each repeat, loop X times (blocks)
        # Dilation increases as 2^0, 2^1, ..., 2^(X-1) inside each repeat.
        self.network = nn.ModuleList(
            TemporalBlock(
                in_channels=B, 
                out_channels=H, 
                kernel_size=kernel_size, 
                dilation=2**i, 
                skip_channels=Sc,
                dropout=dropout
            ) 
            for _ in range(R) for i in range(X)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor from bottleneck (Batch, B, Time)
        
        Returns:
            skip_sum (torch.Tensor): Aggregated features (Batch, Sc, Time)
        """
        skip_sum = 0
        # Pass input through each block sequentially
        for block in self.network:
            x, skip = block(x)
            # Accumulate skip connections from EVERY block
            skip_sum += skip
            
        return skip_sum


class ConvTasNet(nn.Module):
    """
    Conv-TasNet: Convolutional Time-domain Audio Separation Network. For our case, the number of
    speakers will always be 2.
    
    Paper: "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation"
    
    Architecture:
        1. Encoder: Converts raw waveform to a latent feature representation.
        2. Separation Net: Calculates masks for each speaker in the latent space.
        3. Decoder: Reconstructs the separated waveforms from masked features.
        
    Args:
        N (int): Number of filters in autoencoder (encoder output channels).
        L (int): Length of the filters (window size in samples).
        B (int): Number of channels in bottleneck and residual paths.
        H (int): Number of channels in convolutional blocks.
        Sc (int): Number of channels in skip-connection paths.
        P (int): Kernel size in convolutional blocks.
        R (int): Number of repeats.
        X (int): Number of convolutional blocks in each repeat.
    """
    def __init__(self, N = 512, L = 16, B = 128, H = 512, Sc = 128, P = 3, R = 3, X = 8, dropout = 0.0):
        super().__init__()

        self.num_speakers = 2
        self.embed_dim = N
        
        # Encoder
        # Acts like a learnable STFT. 
        # Stride = L//2 implies 50% overlap, ensuring smooth reconstruction.
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=N,
            kernel_size=L,
            stride=L // 2,
            bias = False
        )

        # Bottleneck
        # Compresses the high-dimensional encoder output (N) to a smaller dimension (B)
        # for processing in the separation network.
        self.layernorm = nn.GroupNorm(num_groups=1, num_channels=N)
        self.bottleneck = nn.Conv1d(in_channels=N, out_channels=B, kernel_size=1)

        # Separation Network
        # The TCN that estimates the masks.
        self.separation_net = TemporalConvNet(B, H, Sc, P, X, R, dropout=dropout)

        # Mask Generation
        # Projects the accumulated skip connections back to the encoder dimension (N).
        # We output (N * 2) channels to create 2 distinct masks (one per speaker).
        self.gelu = nn.GELU()
        self.mask_conv = nn.Conv1d(Sc, N * 2, 1)
        self.sigmoid = nn.Sigmoid()

        # Decoder
        # Acts like a learnable Inverse STFT.
        # Transposed convolution reconstructs the waveform from the masked features.
        self.decoder = nn.ConvTranspose1d(N, 1, L, stride=L//2, bias = False)

        self.apply(self._init_weights)
        torch.nn.init.xavier_normal_(self.mask_conv.weight)
        torch.nn.init.xavier_normal_(self.decoder.weight)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity = 'relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, x):
        """
        Forward pass for source separation.
        
        Args:
            x (torch.Tensor): Mixture waveform of shape (Batch, 1, Time_Samples)
        
        Returns:
            output (torch.Tensor): Separated sources of shape (Batch, C, Time_Samples)
        """
        batch_size = x.size(0)
        
        # 1. Encoding
        # Shape: (Batch, 1, T) -> (Batch, N, L_frames)
        w = self.encoder(x)
        
        # 2. Separation / Mask Estimation
        # Apply Norm -> Bottleneck -> TCN
        bottleneck = self.bottleneck(self.layernorm(w))
        skip_sum = self.separation_net(bottleneck)
        
        # Generate masks
        # Shape: (Batch, Sc, L_frames) -> (Batch, N*2, L_frames)
        masks = self.mask_conv(self.gelu(skip_sum)) 
        
        # Reshape to isolate speakers: (Batch, N*C, L) -> (Batch, C, N, L)
        # This splits the big channel dimension into [Number of Speakers] x [Embedding Dim]
        masks = einops.rearrange(
            masks, 
            'Batch (N C) L -> Batch C N L', 
            Batch=batch_size, N=self.embed_dim, C=self.num_speakers
        )
        masks = self.sigmoid(masks) # Constrain masks to [0, 1]
        
        # Apply masks
        # Element-wise multiplication of the Encoder Output (w) with the 2 masks.
        # w is unsqueezed to (Batch, 1, N, L) to broadcast across the speaker dimension.
        masked_w = w.unsqueeze(1) * masks  # Shape: (Batch, 2, N, L)

        # Decoding
        # We must flatten (Batch, C) into a single dimension to pass through ConvTranspose1d
        # Shape: (Batch * 2, N, L)
        decoder_input = masked_w.reshape(batch_size * self.num_speakers, self.embed_dim, -1)
        
        output = self.decoder(decoder_input) # Shape: (Batch * 2, 1, T)
        
        # Reshape back to separate batch and speakers
        output = output.view(batch_size, self.num_speakers, -1) # (Batch, 2, T)
        
        return output