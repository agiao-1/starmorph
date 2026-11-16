import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from torch.distributions.normal import Normal

# ==========================================
# 基础组件 (Basic Components)
# ==========================================

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij') 
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """
    def __init__(self, inshape, nsteps=7):
        super().__init__()
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ConvInsBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()
        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )
    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)


class ResBlock(nn.Module):
    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
        )
    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)


class Encoder(nn.Module):
    def __init__(self, in_channel=1, first_out_channel=16):
        super(Encoder, self).__init__()
        c = first_out_channel
        self.conv0 = ConvInsBlock(in_channel, c, 3, 1)

        self.conv1 = nn.Sequential(
            nn.Conv3d(c, 2*c, kernel_size=3, stride=2, padding=1),
            ResBlock(2*c)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(2*c, 4*c, kernel_size=3, stride=2, padding=1),
            ResBlock(4*c)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(4*c, 8*c, kernel_size=3, stride=2, padding=1),
            ResBlock(8*c)
        )

    def forward(self, x):
        out0 = self.conv0(x) 
        out1 = self.conv1(out0) 
        out2 = self.conv2(out1) 
        out3 = self.conv3(out2) 
        return [out0, out1, out2, out3]


class CConv(nn.Module):
    """
    Standard Convolutional Block for feature fusion
    """
    def __init__(self, in_channels, out_channels):
        super(CConv, self).__init__()
        self.conv = nn.Sequential(
            ConvInsBlock(in_channels, out_channels, 3, 1),
            ConvInsBlock(out_channels, out_channels, 3, 1)
        )

    def forward(self, x):
        return self.conv(x)


# ==========================================
# 创新模块 (Innovation Modules)
# ==========================================

# [创新点 1] 记忆门控机制
class ConvGRU3D(nn.Module):
    """
    Convolutional Gated Recurrent Unit for 3D data.
    Allows the model to maintain a 'memory' of the registration state across iterations.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3, padding=1):
        super(ConvGRU3D, self).__init__()
        self.hidden_channels = hidden_channels
        
        # Reset Gate (r)
        self.conv_r = nn.Conv3d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        # Update Gate (z)
        self.conv_z = nn.Conv3d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        # New Content (h_tilde)
        self.conv_h = nn.Conv3d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, x, h_prev=None):
        if h_prev is None:
            # Init hidden state with zeros if not provided
            batch_size, _, d, h, w = x.size()
            h_prev = torch.zeros(batch_size, self.hidden_channels, d, h, w).to(x.device)
        
        combined = torch.cat([x, h_prev], dim=1)
        
        r = torch.sigmoid(self.conv_r(combined))
        z = torch.sigmoid(self.conv_z(combined))
        
        combined_r = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_r))
        
        h_next = (1 - z) * h_prev + z * h_tilde
        return h_next


# [创新点 2] 相关性层
class CorrelationLayer(nn.Module):
    """
    Computes local correlation (cosine similarity) between Fixed and Warped features.
    Provides an explicit "matching cost" map to the network.
    """
    def __init__(self):
        super(CorrelationLayer, self).__init__()

    def forward(self, fixed_feat, warped_moving_feat):
        # Normalize features along channel dimension
        f_norm = nnf.normalize(fixed_feat, dim=1)
        w_norm = nnf.normalize(warped_moving_feat, dim=1)
        # Pixel-wise dot product (Cosine Similarity)
        # Output shape: (Batch, 1, D, H, W)
        correlation = torch.sum(f_norm * w_norm, dim=1, keepdim=True)
        return correlation


# [创新点 3] 小波高频分支
class WaveletHighFreq(nn.Module):
    """
    Simulates high-frequency extraction (similar to Haar Wavelet detail coefficients).
    Helping the network focus on edges and textures during fine-tuning stages.
    """
    def __init__(self):
        super(WaveletHighFreq, self).__init__()
        # Use average pooling to simulate low-pass filter
        self.avg_pool = nn.AvgPool3d(kernel_size=2, stride=1, padding=1)

    def forward(self, x):
        # Low freq part
        # Padding to match size when stride=1
        pad_x = nnf.pad(x, (0,1,0,1,0,1), mode='replicate') 
        low_freq = nnf.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        
        # High freq = Original - Low Freq
        high_freq = x - low_freq
        return high_freq


# ==========================================
# 完整的 RDP Advanced 模型
# ==========================================

class RDP_Advanced(nn.Module):
    def __init__(self, inshape=(160,192,160), flow_multiplier=1., in_channel=1, channels=16):
        super(RDP_Advanced, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.inshape = inshape
        c = self.channels

        # --- Encoders ---
        self.encoder_moving = Encoder(in_channel=in_channel, first_out_channel=c)
        self.encoder_fixed = Encoder(in_channel=in_channel, first_out_channel=c)

        # --- Helper Modules ---
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(4):
            self.warp.append(SpatialTransformer([s // 2**i for s in inshape]))
            self.diff.append(VecInt([s // 2**i for s in inshape]))

        # --- Innovation Modules Initialization ---
        self.corr_layer = CorrelationLayer()
        self.wavelet = WaveletHighFreq()

        # ================= Stage 4 (Bottleneck / Coarse) =================
        # 160/8 = 20. Keep simple Conv structure here.
        self.cconv_4 = nn.Sequential(
            ConvInsBlock(16 * c, 8 * c, 3, 1),
            ConvInsBlock(8 * c, 8 * c, 3, 1)
        )
        self.defconv4 = self._make_defconv(8*c)
        self.dconv4 = nn.Sequential(
            ConvInsBlock(3*8*c, 8*c),
            ConvInsBlock(8*c, 8*c)
        )

        # ================= Stage 3 (Medium Resolution) =================
        # 160/4 = 40. Start using GRU.
        self.upconv3 = UpConvBlock(8*c, 4*c, 4, 2)
        
        # Scale 3 GRU Setup
        # Inputs: Fixed(4c) + Warped(4c) + Correlation(1) = 8c + 1
        gru3_in_dim = 2 * (4*c) + 1 
        self.gru3 = ConvGRU3D(in_channels=gru3_in_dim, hidden_channels=4*c)
        
        self.cconv_3 = CConv(4*c, 4*c) # Process GRU output
        self.defconv3 = self._make_defconv(4*c)

        # ================= Stage 2 (Fine Resolution - Main Refinement) =================
        # 160/2 = 80. Use GRU + Wavelet here.
        self.upconv2 = UpConvBlock(4*c, 2*c, 4, 2)
        
        # Scale 2 GRU Setup
        # Inputs: Fixed(2c) + Warped(2c) + Correlation(1) + Wavelet(1) = 4c + 2
        gru2_in_dim = 2 * (2*c) + 1 + 1 
        self.gru2 = ConvGRU3D(in_channels=gru2_in_dim, hidden_channels=2*c)
        
        self.cconv_2 = CConv(2*c, 2*c)
        self.defconv2 = self._make_defconv(2*c)

        # ================= Stage 1 (Full Resolution) =================
        # 160/1 = 160. Keep memory efficient (Standard Conv).
        self.upconv1 = UpConvBlock(2*c, c, 4, 2)
        
        # Inputs: Fixed(c) + Warped(c) + Prev(c) + Corr(1) = 3c + 1
        self.cconv_1 = CConv(3*c + 1, c)
        self.defconv1 = self._make_defconv(c)

    def _make_defconv(self, in_channel):
        # Helper to create flow prediction layer with small init
        conv = nn.Conv3d(in_channel, 3, 3, 1, 1)
        conv.weight = nn.Parameter(Normal(0, 1e-5).sample(conv.weight.shape))
        conv.bias = nn.Parameter(torch.zeros(conv.bias.shape))
        return conv

    def forward(self, moving, fixed):
        # --- Encoding ---
        M1, M2, M3, M4 = self.encoder_moving(moving)
        F1, F2, F3, F4 = self.encoder_fixed(fixed)
        # Channels: M4=8c, M3=4c, M2=2c, M1=c

        # ================= Level 4 (Coarsest) =================
        # Standard Concat & Refine
        C4 = torch.cat([F4, M4], dim=1)
        C4 = self.cconv_4(C4) # Feature extraction

        # Initial Flow
        flow = self.defconv4(C4)
        flow = self.diff[3](flow) # Integrate

        # Refinement loop (Simple Conv)
        warped = self.warp[3](M4, flow)
        C4 = self.dconv4(torch.cat([F4, warped, C4], dim=1)) # Update State
        v = self.defconv4(C4)
        w = self.diff[3](v)
        
        # ================= Level 3 (GRU + Correlation) =================
        # 2 Iterations
        
        # 1. Upsample State & Flow
        D3 = self.upconv3(C4) # Hidden state initialization
        flow = self.upsample_trilin(2 * (self.warp[3](flow, w) + w))
        
        # 2. Iterative Refinement
        gru_state = D3 # Initialize hidden state
        
        for _ in range(2):
            warped_feat = self.warp[2](M3, flow)
            
            # [Innovation] Correlation Map
            corr = self.corr_layer(F3, warped_feat)
            
            # Prepare GRU Input (Fixed + Warped + Corr)
            gru_input = torch.cat([F3, warped_feat, corr], dim=1)
            
            # [Innovation] ConvGRU Update
            gru_state = self.gru3(gru_input, gru_state)
            
            # Predict Flow Delta
            feat = self.cconv_3(gru_state)
            v = self.defconv3(feat)
            w = self.diff[2](v)
            
            # Compose Flow
            flow = self.warp[2](flow, w) + w
        
        D3 = gru_state # Pass final state to next level

        # ================= Level 2 (GRU + Corr + Wavelet) =================
        # 3 Iterations (Most intensive stage)

        D2 = self.upconv2(D3)
        flow = self.upsample_trilin(2 * (self.warp[2](flow, w) + w))
        
        gru_state = D2
        
        # Compute Wavelet (High Freq) for Fixed Image once to save time
        fixed_high_freq = self.wavelet(F2)

        for _ in range(3):
            warped_feat = self.warp[1](M2, flow)
            
            # [Innovation] Correlation
            corr = self.corr_layer(F2, warped_feat)
            
            # [Innovation] Wavelet (High Freq Difference)
            # Calculate high freq of warped feature on the fly
            warped_high_freq = self.wavelet(warped_feat)
            high_freq_diff = fixed_high_freq - warped_high_freq # Difference map
            
            # Prepare GRU Input
            gru_input = torch.cat([F2, warped_feat, corr, high_freq_diff], dim=1)
            
            # [Innovation] ConvGRU Update
            gru_state = self.gru2(gru_input, gru_state)
            
            # Predict Flow Delta
            feat = self.cconv_2(gru_state)
            v = self.defconv2(feat)
            w = self.diff[1](v)
            
            flow = self.warp[1](flow, w) + w
            
        D2 = gru_state

        # ================= Level 1 (Finest) =================
        # 1 Iteration (Memory optimized)

        D1 = self.upconv1(D2)
        flow = self.upsample_trilin(2 * (self.warp[1](flow, w) + w))
        
        warped_feat = self.warp[0](M1, flow)
        
        # Simple Correlation here
        corr = self.corr_layer(F1, warped_feat)
        
        C1 = torch.cat([F1, warped_feat, D1, corr], dim=1)
        C1 = self.cconv_1(C1)
        
        v = self.defconv1(C1)
        w = self.diff[0](v)
        flow = self.warp[0](flow, w) + w

        # Final Image Warp
        y_moved = self.warp[0](moving, flow)

        return y_moved, flow


if __name__ == '__main__':
    # ================= Test Block =================
    # Check if model builds and runs on random data
    print("Initializing RDP Advanced Model...")
    
    # Batch=1, Channel=1, D=80, H=96, W=80 (Half resolution test for speed)
    size = (1, 1, 80, 96, 80) 
    
    model = RDP_Advanced(inshape=size[2:])
    
    if torch.cuda.is_available():
        model = model.cuda()
        A = torch.randn(size).cuda()
        B = torch.randn(size).cuda()
        print("Running on GPU...")
    else:
        A = torch.randn(size)
        B = torch.randn(size)
        print("Running on CPU...")

    # Forward Pass
    try:
        out, flow = model(A, B)
        print("-" * 30)
        print("Forward Pass Successful!")
        print(f"Input Shape:  {size}")
        print(f"Output Shape: {out.shape}")
        print(f"Flow Shape:   {flow.shape}")
        print("-" * 30)
        print("Innovations Integrated:")
        print("1. ConvGRU3D (Memory) in Stages 2 & 3")
        print("2. CorrelationLayer (Correspondence) in Stages 1, 2, 3")
        print("3. WaveletHighFreq (Detail) in Stage 2")
    except Exception as e:
        print(f"Error during forward pass: {e}")