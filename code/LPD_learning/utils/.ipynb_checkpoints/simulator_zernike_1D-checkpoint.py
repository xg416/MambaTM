import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.turb import z_space, nollCovMat
# from scipy.stats import truncnorm 
# import matplotlib.pyplot as plt

class Simulator(nn.Module):
    def __init__(self, path, turb_param_dict, ks=65, Batch_unrelated=1):
        super().__init__()
        
        self.turb_params = turb_param_dict
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.Batch_unrelated = Batch_unrelated # if 1, frames in one batch is not in sequential order

        # Loading the P2S module, integral_path, blur kernels
        self.integral_path = os.path.join(path, 'precomputed/correlation_integrals.pth')
        self.corr_integral = torch.load(self.integral_path, map_location=self.device)
        dict_psf = torch.load(os.path.join(path, 'precomputed/kernel_2DPCA.pt'), map_location=self.device)
        # dict_psf = torch.load(os.path.join(path, '/home/xingguang/Documents/turb/TurbulenceSim_P2S/P2Sv3/data/dictionary.pt'), map_location=self.device)
        self.mu = dict_psf['mu_comp'].unsqueeze(0).unsqueeze(0).permute(3,1,2,0).to(self.device, dtype=torch.float32)
        self.n_mu = dict_psf['mu_comp'].shape[-1]
        self.basis_psf_left = dict_psf['Z'].unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
        self.basis_psf_right = dict_psf['X'].unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32) 
        self.size_feat = dict_psf['K']
        self.mapping = _P2S(output_dim=self.size_feat**2)
        self.mapping.load_state_dict(
            torch.load(os.path.join(path, 'precomputed/P2S_state_2DPCA.pt'), map_location=self.device))
        self.kernel_size = 65
        self.p2s_blur_left, self.p2s_blur_right = self._blur_construct(ks)


    def _blur_construct(self, ksize=65):
        n_coeff = self.size_feat**2
        if self.kernel_size == ksize:
            local_basis_psf_l = self.basis_psf_left
            local_basis_psf_r = self.basis_psf_right
            local_mu = self.mu
        else:
            local_mu = F.interpolate(self.mu, size=(ksize, 1), mode='bilinear')
            local_basis_psf_l = F.interpolate(self.basis_psf_left, size=(ksize, self.size_feat), mode='bilinear')
            local_basis_psf_r = F.interpolate(self.basis_psf_right, size=(ksize, self.size_feat), mode='bilinear')
        self.kernel_size = ksize

        # Turn P2S spatial blurry kernels to nn.conv2d
        p2s_blur_left = nn.Conv2d(n_coeff + self.n_mu, n_coeff + self.n_mu, (ksize, 1), groups=n_coeff + self.n_mu, padding='same', \
                                  padding_mode='reflect', bias=False, device=self.device, dtype=torch.float32)
        p2s_blur_right = nn.Conv2d(n_coeff + self.n_mu, n_coeff + self.n_mu, (1, ksize), groups=n_coeff + self.n_mu, padding='same', \
                                   padding_mode='reflect', bias=False, device=self.device, dtype=torch.float32) 
        p2s_blur_left.weight.data[:n_coeff, ...] = local_basis_psf_l.permute(3,1,2,0).repeat(1, self.size_feat, 1, 1) \
                                                                                    .reshape(n_coeff, 1, ksize, 1) # 1,1,ks,K -> K**2, 1, ks, 1
        p2s_blur_left.weight.data[n_coeff:, ...] = local_mu
        p2s_blur_right.weight.data[:n_coeff, ...] = local_basis_psf_r.permute(0,3,2,1).repeat(self.size_feat, 1, 1, 1) \
                                                                                    .reshape(n_coeff, 1, ksize, 1).permute(0,1,3,2) # 1,1,ks,K -> K**2, 1, 1, ks
        p2s_blur_right.weight.data[n_coeff:, ...] = local_mu.permute(0, 1, 3, 2)
        return p2s_blur_left, p2s_blur_right
    
    def forward(self, img, zernike, ksize=55):
        """function that does the core of the dfp2s simulation

        Args:
            img (tensor): input image tensor
            see sim_examples directory for details on how to set this up

        Returns:
            tensor: simulated version of input img
        """
        # adapt to new input size
        batchN, channelN, H, W = img.shape
        
        if self.kernel_size != ksize:
            self.p2s_blur_left, self.p2s_blur_right = self._blur_construct(ksize)

        # Generating the pixel-shift values
        yy, xx = torch.meshgrid(torch.arange(0, H, device=self.device), torch.arange(0, W, device=self.device))
        grid = torch.stack((xx, yy), -1).unsqueeze(0).to(dtype=torch.float)
        
        pos = zernike[...,:2]

        # setting up the flow array
        flow = 2.0 * (grid + pos*1)/(torch.tensor((W, H), device=self.device)-1) - 1.0
        # applying the flow array
        tilt_img = F.grid_sample(img, flow, 'bilinear', padding_mode='border', align_corners=False)
        # tilt_img = tilt_img.view((-1, 1, H, W))

        # Convolving the image with the dictionary, weights will be applied later
        # Forming the blurred image based off of p2s weights
        # Computing the weights of the image dictionary using p2s
        weight = torch.ones((batchN, H, W, self.size_feat**2+self.n_mu), dtype=torch.float32, device=self.device)
        weight[..., :self.size_feat**2] = self.mapping(zernike[...,2:])


        ones_img = torch.ones_like(tilt_img)
        big_img = torch.cat((tilt_img.view(batchN, channelN, H, W).unsqueeze(4), 
                                ones_img.view(batchN, channelN, H, W).unsqueeze(4)), 1)
        big_img = big_img * weight.unsqueeze(1)
        # print(big_img.shape)
        dict_img = self.p2s_blur_right(self.p2s_blur_left(big_img.view(-1, H, W, self.size_feat**2+self.n_mu).permute(0,3,1,2)))
        dict_img = dict_img.view(batchN, -1, self.size_feat**2+self.n_mu, H, W)
        norm_img = dict_img[:, 3:]
        out = torch.sum(dict_img[:, :3], dim=2) / torch.sum(norm_img, dim=2)
        
        # tilt_img = tilt_img.view((batchN, channelN, H, W)).squeeze()
        
        return out


def ar1_white_noise(in_arr, param):
    """function to generation ar1 (correlated) white noise.

    Note: be sure the in_arr is unit variance white noise!!!!!

    Args:
        in_arr (tensor): input unit variance white noise
        param (_type_): correlation value [0, 1]
        device (_type_): torch device

    Returns:
        _type_: output correlated white noise tensor
    """
    return param*torch.real(in_arr) + (1 - param**2)**(1/2)*torch.randn_like(in_arr) + \
        1j*param*torch.imag(in_arr) + 1j*(1 - param**2)**(1/2)*torch.randn_like(in_arr)
        
        
class _P2S(nn.Module):
    def __init__(self, input_dim=33, hidden_dim=200, output_dim=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        y = self.act(self.fc1(x))
        y = self.act(self.fc2(y))
        out = self.fc3(y)
        return out
