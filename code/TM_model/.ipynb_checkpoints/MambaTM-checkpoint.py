import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import argparse, math
from thop import profile, clever_format
from .modules.Hilbert3d import Hilbert3d
from .modules.mambablock import MambaLayerglobal, MambaLayerlocal, MambaLayerglobalRef, MambaLayerlocalRef
from .archs import conv1x1, conv3x3, conv5x5, actFunc


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, activation='gelu'):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
    
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size=[3,3], reduction=16, activation='gelu'):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size[0], padding=kernel_size[0]//2, bias=True))
        modules_body.append(actFunc(activation))
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size[1], padding=kernel_size[1]//2, bias=True))

        self.CA = CALayer(n_feat, reduction)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
    
# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate, activation))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out

    
class EncodeCell(nn.Module):
    def __init__(self, para, in_channel=3):
        super(EncodeCell, self).__init__()
        self.n_feats = para.n_features
        self.conv = conv5x5(in_channel, self.n_feats, stride=1)
        self.down1 = conv5x5(self.n_feats, 2*self.n_feats, stride=2)
        self.down2 = conv5x5(2*self.n_feats, 4*self.n_feats, stride=2)
        self.down3 = conv5x5(4*self.n_feats, 8*self.n_feats, stride=2)
        self.enc_l1 = RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3)
        self.enc_l2 = RDB(in_channels=2 * self.n_feats, growthRate=int(self.n_feats * 3 / 2), num_layer=3)
        self.enc_l3 = RDB(in_channels=4 * self.n_feats, growthRate=self.n_feats * 2, num_layer=3)
        self.enc_h = RDB(in_channels=8 * self.n_feats, growthRate=self.n_feats * 2, num_layer=3)

    def forward(self, x):
        '''
        out1: torch.Size([B, 16, 256, 256])
        out2: torch.Size([B, 32, 128, 128])
        out3: torch.Size([B, 64, 64, 64])
        h: torch.Size([B, 128, 32, 32])
        '''
        b,c,h,w = x.shape
        
        out1 = self.enc_l1(self.conv(x))
        out2 = self.enc_l2(self.down1(out1))
        out3 = self.enc_l3(self.down2(out2))
        h = self.enc_h(self.down3(out3))
        
        return out1, out2, out3, h
        
class DecodeCell(nn.Module):
    def __init__(self, para, out_dim=3):
        super(DecodeCell, self).__init__()
        self.n_feats = para.n_features
        self.uph = nn.ConvTranspose2d(8 * self.n_feats, 4 * self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1)
        self.fusion3 = CAB(8*self.n_feats, [1,3])
        
        self.up3 = nn.ConvTranspose2d(8*self.n_feats, 2*self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1)    
        self.fusion2 = CAB(4*self.n_feats, [1,3]) 
        
        self.up2 = nn.ConvTranspose2d(4*self.n_feats, self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1)                
        self.fusion1 = CAB(2*self.n_feats, [1,3])   
        
        self.output = nn.Sequential(
            conv3x3(2*self.n_feats, self.n_feats, stride=1),
            conv3x3(self.n_feats, out_dim, stride=1)
        )

    def forward(self, h, x3, x2, x1):
        # channel: 8, 4, 2 * n_feat
        
        h_decode = self.uph(h)
        x3 = self.fusion3(torch.cat([h_decode, x3], dim=1))
        
        x3_up = self.up3(x3)
        x2 = self.fusion2(torch.cat([x3_up, x2], dim=1))
        
        x2_up = self.up2(x2)
        x1 = self.fusion1(torch.cat([x2_up, x1], dim=1))
        
        return self.output(x1)
    
    
    
class Model(nn.Module):
    def __init__(self, para, input_size=(192, 192, 16)):
        super(Model, self).__init__()
        self.para = para
        self.n_feats = para.n_features
        self.input_size = input_size
        self.n_blocks = para.n_blocks
        self.n_ref_block = 1
        assert self.n_ref_block < self.n_blocks
        
        self.encoder = EncodeCell(para)
        self.ref_encoder = EncodeCell(para, 4)
        self.convert_ref = nn.Conv2d(128, 128, 1)
        H, W, T = input_size
        self.set_h_curve(H, W, T, "cuda")
        self.SGlobalMambaBlocks = nn.ModuleList()
        self.TGlobalMambaBlocks = nn.ModuleList()
        self.LocalMambaBlocks = nn.ModuleList()
        for _ in range(self.n_blocks-self.n_ref_block):
            self.SGlobalMambaBlocks.append(MambaLayerglobal(dim=self.n_feats*8))
            self.TGlobalMambaBlocks.append(MambaLayerglobal(dim=self.n_feats*8, spatial_first=False))
            self.LocalMambaBlocks.append(MambaLayerlocal(dim=self.n_feats*8))
        for _ in range(self.n_ref_block):
            self.SGlobalMambaBlocks.append(MambaLayerglobalRef(dim=self.n_feats*8))
            self.TGlobalMambaBlocks.append(MambaLayerglobalRef(dim=self.n_feats*8, spatial_first=False))
            self.LocalMambaBlocks.append(MambaLayerlocalRef(dim=self.n_feats*8))    
        self.ref_decoder = DecodeCell(para, 4)
        self.decoder = DecodeCell(para)
    
    def set_h_curve(self, H, W, T, device):
        SH = math.ceil(H/8)
        SW = math.ceil(W/8)
        h_curve = list(Hilbert3d(width=SW, height=SH, depth=T))
        h_curve = torch.tensor(h_curve).long().to(device)
        self.h_curve = h_curve[:, 0] * SW * T + h_curve[:, 1] * T + h_curve[:, 2]
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        if (H, W, T) != self.input_size or self.h_curve.device != x.device:
            self.set_h_curve(H, W, T, x.device)
            self.input_size = (H, W, T)

        enc1, enc2, enc3, h = self.encoder(x.contiguous().view(-1, C, H, W))
        h = rearrange(h, '(b t) c h w -> b t c h w', t=T)

        for i in range(self.n_blocks-self.n_ref_block):
            h = self.SGlobalMambaBlocks[i](h)
            h = self.TGlobalMambaBlocks[i](h)
            h = self.LocalMambaBlocks[i](h, self.h_curve)
            
        h_ref = self.convert_ref(h.flatten(0,1))
        LPD = self.ref_decoder(h_ref, enc3, enc2, enc1)
        _, _, _, ref = self.ref_encoder(LPD)
        ref = rearrange(ref, '(b t) c h w -> b t c h w', t=T)
        for i in range(self.n_blocks-self.n_ref_block, self.n_blocks):
            h = self.SGlobalMambaBlocks[i](h, ref)
            h = self.TGlobalMambaBlocks[i](h, ref)
            h = self.LocalMambaBlocks[i](h, ref, self.h_curve)
        h = rearrange(h, 'b t c h w -> (b t) c h w')  
        output = self.decoder(h, enc3, enc2, enc1)
        return output.view(B, T, C, H, W), LPD.view(B, T, -1, H, W)


def feed(model, iter_samples):
    inputs = iter_samples
    outputs = model(inputs)
    return outputs

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and restoration')
    parser.add_argument('--model', type=str, default='MambaTM', help='type of model to construct')
    parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
    parser.add_argument('--n_blocks', type=int, default=6, help='# of blocks in middle part of the model')
    return parser.parse_args()

if __name__ == "__main__":
    from thop import profile, clever_format
    from torchsummary import summary
    from utils_fvscore import FLOPs
    params = get_args()
    B, T, H, W = 1, 8, 256, 256 
    device = torch.device('cuda')
    n_repeat = 10
    model = Model(params, input_size=(H, W, T)).cuda()
    data = torch.randn((1,T,3,H,W)).to(device=device)

    # 58 FPS under 512x512
    with torch.no_grad():
        summary(model, (T,3,H,W))
        
    flops, params = profile(model, inputs=(data,), verbose=False)
    
    macs, params = clever_format([flops, params], "%.3f") # 960x540 ~ 8x256x256
    print(macs, params)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True) 

    out = model(data)
    start_event.record()
    for i in range(n_repeat):
        out, PD = model(data)
        out = torch.sum(out) + torch.sum(PD)
        out.backward()
    end_event.record()
    
    with torch.no_grad():
        FLOPs.fvcore_flop_count(model, inputs=data)
    # with torch.no_grad():
    #     out, _ = model(data)
    #     start_event.record()
    #     for i in range(n_repeat):
    #         out = model(data)
    #     end_event.record()

    # Wait for everything to finish running
    torch.cuda.synchronize()

    # Calculate elapsed time in milliseconds
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average elapsed time: {elapsed_time_ms/n_repeat} ms")
    
