from .TM_ssm import BiMamba, GMamba
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, dim, hidden_features, squeeze_factor=16):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv3d(dim, hidden_features, 3, 1, 1, groups=dim),
            nn.GELU(),
            nn.Conv3d(hidden_features, hidden_features, kernel_size=1),
            ChannelAttention(hidden_features, squeeze_factor),
            nn.Conv3d(hidden_features, dim, kernel_size=1),
        )
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x, nf, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().reshape(B, C, nf, H, W)
        x = self.cab(x)
        x = x.contiguous().flatten(2).transpose(1, 2)
        return x
    

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, nf, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().reshape(B, C, nf, H, W)
        x = self.dwconv(x)
        x = x.contiguous().flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, nf, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, nf, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MambaLayerglobal(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU, 
                 spatial_first=True, channel_mixer="mlp"):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.spatial_first = spatial_first
        
        self.mamba =BiMamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # use_fast_path=False,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if channel_mixer=="mlp":
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif channel_mixer=="cab":
            self.mlp = CAB(dim=dim, hidden_features=mlp_hidden_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x):
        B, nf, C, H, W = x.shape
        if self.spatial_first:
            x = x.permute(0, 2, 1, 3, 4) # T H W
        else:
            x = x.permute(0, 2, 4, 3, 1) # W H T

        assert C == self.dim

        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, -1).transpose(-1, -2)

        # Bi-Mamba layer
        if self.spatial_first:
            x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat)))
            x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), nf, H, W))
        else:
            x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat)))
            x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), W, H, nf))
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        if self.spatial_first:
            out = out.permute(0, 2, 1, 3, 4)
        else:
            out = out.permute(0, 4, 1, 3, 2)
        return out

    
class MambaLayerlocal(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU,
                 reverse=True, channel_mixer="mlp"):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = BiMamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # use_fast_path=False,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if channel_mixer=="mlp":
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif channel_mixer=="cab":
            self.mlp = CAB(dim=dim, hidden_features=mlp_hidden_dim)
        self.reverse = reverse
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, hilbert_curve):

        x = x.permute(0, 2, 1, 3, 4)
        B, C, nf, H, W = x.shape
        
        if self.reverse:
            x = x.permute(0, 1, 3, 4, 2)
        assert C == self.dim

        img_dims = x.shape[2:]
        x_hw = x.flatten(2).contiguous()

        x_hil = x_hw.index_select(dim=-1, index=hilbert_curve)
        x_flat = x_hil.transpose(-1, -2)

        # Bi-Mamba layer
        x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat)))
        x_mamba_out = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), nf, H, W))
        outmamba = x_mamba_out.transpose(-1, -2)

        sum_out = torch.zeros_like(outmamba)
        hilbert_curve_re = repeat(hilbert_curve, 'hw -> b c hw', b=outmamba.shape[0], c=outmamba.shape[1])
        assert outmamba.shape == hilbert_curve_re.shape

        sum_out.scatter_add_(dim=-1, index=hilbert_curve_re, src=outmamba)
        sum_out = sum_out.reshape(B, C, *img_dims).contiguous()

        if self.reverse:
            out = sum_out.permute(0, 1, 4, 2, 3)

        out = out.permute(0, 2, 1, 3, 4)

        return out
    
    
class MambaLayerglobalRef(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU,
                 spatial_first=True, channel_mixer="mlp"):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm1_ref = nn.LayerNorm(dim)
        self.spatial_first = spatial_first
        self.mamba = GMamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # use_fast_path=False,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if channel_mixer=="mlp":
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif channel_mixer=="cab":
            self.mlp = CAB(dim=dim, hidden_features=mlp_hidden_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x, ref):
        assert x.shape == ref.shape
        B, nf, C, H, W = x.shape
        if self.spatial_first:
            x = x.permute(0, 2, 1, 3, 4) # T H W
            ref = ref.permute(0, 2, 1, 3, 4) # T H W
        else:
            x = x.permute(0, 2, 4, 3, 1) # W H T
            ref = ref.permute(0, 2, 4, 3, 1) # T H W

        assert C == self.dim

        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, -1).transpose(-1, -2)
        ref_flat = ref.reshape(B, C, -1).transpose(-1, -2)

        # Bi-Mamba layer
        if self.spatial_first:
            x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat), self.norm1_ref(ref_flat)))
            x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), nf, H, W))
        else:
            x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat), self.norm1_ref(ref_flat)))
            x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), W, H, nf))
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        if self.spatial_first:
            out = out.permute(0, 2, 1, 3, 4)
        else:
            out = out.permute(0, 4, 1, 3, 2)
        return out


class MambaLayerlocalRef(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU,
                 reverse=True, channel_mixer="mlp"):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm1_ref = nn.LayerNorm(dim)
        self.mamba = GMamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # use_fast_path=False,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if channel_mixer=="mlp":
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif channel_mixer=="cab":
            self.mlp = CAB(dim=dim, hidden_features=mlp_hidden_dim)
        self.reverse = reverse
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, ref, hilbert_curve):
        assert x.shape == ref.shape
        x = x.permute(0, 2, 1, 3, 4)
        ref = ref.permute(0, 2, 1, 3, 4)
        B, C, nf, H, W = x.shape
        
        if self.reverse:
            x = x.permute(0, 1, 3, 4, 2)
            ref = ref.permute(0, 1, 3, 4, 2)
        assert C == self.dim

        img_dims = x.shape[2:]
        x_hw = x.flatten(2).contiguous()
        ref_hw = ref.flatten(2).contiguous()
        x_hil = x_hw.index_select(dim=-1, index=hilbert_curve)
        ref_hil = ref_hw.index_select(dim=-1, index=hilbert_curve)
        
        x_flat = x_hil.transpose(-1, -2)
        ref_flat = ref_hil.transpose(-1, -2)
        # Bi-Mamba layer
        x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat), self.norm1_ref(ref_flat)))
        x_mamba_out = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), nf, H, W))
        outmamba = x_mamba_out.transpose(-1, -2)

        sum_out = torch.zeros_like(outmamba)
        hilbert_curve_re = repeat(hilbert_curve, 'hw -> b c hw', b=outmamba.shape[0], c=outmamba.shape[1])
        assert outmamba.shape == hilbert_curve_re.shape

        sum_out.scatter_add_(dim=-1, index=hilbert_curve_re, src=outmamba)
        sum_out = sum_out.reshape(B, C, *img_dims).contiguous()

        if self.reverse:
            out = sum_out.permute(0, 1, 4, 2, 3)

        out = out.permute(0, 2, 1, 3, 4)

        return out