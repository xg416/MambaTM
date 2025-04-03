import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.turb import z_space, nollCovMat
from scipy.stats import truncnorm 

class Simulator(nn.Module):
    def __init__(self, path, turb_param_dict, Batch_unrelated=1):
        super().__init__()
        
        self.turb_params = turb_param_dict
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.Batch_unrelated = Batch_unrelated # if 1, frames in one batch is not in sequential order

        # Loading the P2S module, integral_path, blur kernels
        self.integral_path = os.path.join(path, 'precomputed/correlation_integrals.pth')
        self.corr_integral = torch.load(self.integral_path, map_location=self.device)
        dict_psf = torch.load(os.path.join(path, 'precomputed/kernel_2DPCA.pt'), map_location=self.device)
        self.mu = dict_psf['mu_comp'].unsqueeze(0).unsqueeze(0).permute(3,1,2,0).to(self.device, dtype=torch.float32)
        self.n_mu = dict_psf['mu_comp'].shape[-1]
        self.basis_psf_left = dict_psf['Z'].unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
        self.basis_psf_right = dict_psf['X'].unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32) 
        self.size_feat = dict_psf['K']
        self.mapping = _P2S(output_dim=self.size_feat**2)
        self.mapping.load_state_dict(
            torch.load(os.path.join(path, 'precomputed/P2S_state_2DPCA.pt'), map_location=self.device))
        self._initialize_all()

    def _initialize_all(self):
        # Generating a few parameters and making them accessible
        # H, W: image shape in pixel
        # h, w: image shape in meter
        # HH, WW: image shape to achieve no aliasing 
        # pr, PR: no aliasing range in physics(meters) and image(pixel) 
        self.H, self.W = self.turb_params['img_shape']
        self.batchN = self.turb_params['batch_size']
        self.L = self.turb_params['L']
        self.f = self.turb_params['f']
        self.wvl = self.turb_params['wvl']
        self.width = self.turb_params['width']
        self.D = self.turb_params['D']
        self.pr = self.turb_params['anti_aliasing_range']
        self.dx = max(self.turb_params['width'], self.turb_params['anti_aliasing_range']) / self.W

        # self.weight = torch.ones((self.batchN, self.H, self.W, self.size_feat**2+self.n_mu), dtype=torch.float32, device=self.device)

        
        self.PR = int(self.pr / self.dx)
        self.HH = min(int(self.H / (self.turb_params['anti_aliasing_range'] / self.turb_params['width'])), self.H)
        self.WW = min(int(self.W / (self.turb_params['anti_aliasing_range'] / self.turb_params['width'])), self.W)
        if self.HH == self.H or self.WW == self.W:
            self.resample = False
        else:
            self.resample = True
        # Zernike space covariance generation.
        z_cov, self.r0 = z_space(self.turb_params, self.corr_integral, self.dx, (self.H, self.W), self.device).generate()
        self.turb_params['r0'] = self.r0 # z_space returns a r0 value based on Cn2 profile, save to params
        self.Dr0 = (self.D / self.r0)
        if self.Dr0 > 10:
            self.turb_params['D'] /= 1.5
            self._initialize_all() 
            
        self.temp_corr = self.turb_params['temp_corr']
        self.Z = self.turb_params['num_zern'] - 1

        # A grid of positions in the image, will be used for warping later
        yy, xx = torch.meshgrid(torch.arange(0, self.H, device=self.device), torch.arange(0, self.W, device=self.device))
        self.grid = torch.stack((xx, yy), -1).unsqueeze(0).to(dtype=torch.float)
        
        # define low pass filtering for the PSD of random field
        k = min(self.H, self.W)/5
        yy, xx = torch.meshgrid(torch.arange(0, self.H), torch.arange(0, self.W))
        dist = torch.sqrt((xx - self.W/2)**2 + (yy - self.H/2)**2) / k # use k to control the filter range
        filter = torch.exp(-dist**2).to(self.device, dtype=torch.float)
        self.filter = torch.fft.fftshift(filter)
        
        # We need to normalize the Zernike space covariance. The Noll matrix will
        # weight them accordingly. Then form the PSDs
        z_cov = z_cov / torch.amax(z_cov, dim=(0,1)).unsqueeze(0).unsqueeze(0)

        self.psd_stack = torch.abs(torch.fft.fft2(z_cov, dim=(0,1)))
        psd = self.psd_stack[..., :2] * self.filter.unsqueeze(2)
        self.psd_stack[..., :2] = psd * self.psd_stack[..., :2].sum(dim=(0,1))/psd.sum(dim=(0,1))
        
        # construct the blur kernel
        self.p2s_blur_left, self.p2s_blur_right = self._blur_construct()

        # Generating the cholesky(Noll), which is just called Noll
        self.nollCovMat =torch.linalg.cholesky(nollCovMat(self.Z+1).to(self.device, dtype=torch.float32))
        self.Noll = self.nollCovMat * (self.Dr0)**(5/6)
        
        # A scaling constant for the tilts (Zernike to spatial units to pixels conversion)
        tilt_factor = 1 / min(2, max(1, self.Dr0.item() / 4))
        self.tilt_const = (self.wvl / self.D) * self.L * (self.W/ self.width) * tilt_factor
        
        # Initialize the random seed
        self.temp_corr_zer = torch.ones((self.Z), device=self.device)
        self.temp_corr_zer[2:] = 1/1.4
        self.rnd_seed = self._init_rnd_seed((self.batchN, self.H, self.W, self.Z)) 
        
        
    def _blur_construct(self):
        og_ksize = 65
        n_coeff = self.size_feat**2
        
        # third_diff_ring = 3 * 1.22 * self.wvl * self.L / self.D                 # distance from center to third ring of diffraction pattern
        # factor = min(2, max(1, self.Dr0.item() / 4))
        # kernel_size = max(round(((third_diff_ring / self.dx) / 10) * og_ksize * factor), 1) 
        # if self.resample:
        #     kernel_size *= (self.W / self.WW)
        #     kernel_size = int(kernel_size)
        # if kernel_size % 2 == 0:
        #     kernel_size += 1
        
        ks = self.turb_params['kernel_size']
        # self.turb_params['kernel_size'] = ks
        if ks == og_ksize:
            local_basis_psf_l = self.basis_psf_left
            local_basis_psf_r = self.basis_psf_right
            local_mu = self.mu
        else:
            local_mu = F.interpolate(self.mu, size=(ks, 1), mode='bilinear')
            local_basis_psf_l = F.interpolate(self.basis_psf_left, size=(ks, self.size_feat), mode='bilinear')
            local_basis_psf_r = F.interpolate(self.basis_psf_right, size=(ks, self.size_feat), mode='bilinear')

        # Turn P2S spatial blurry kernels to nn.conv2d
        p2s_blur_left = nn.Conv2d(n_coeff + self.n_mu, n_coeff + self.n_mu, (ks, 1), groups=n_coeff + self.n_mu, padding='same', \
                                  padding_mode='reflect', bias=False, device=self.device, dtype=torch.float32)
        p2s_blur_right = nn.Conv2d(n_coeff + self.n_mu, n_coeff + self.n_mu, (1, ks), groups=n_coeff + self.n_mu, padding='same', \
                                   padding_mode='reflect', bias=False, device=self.device, dtype=torch.float32) 
        p2s_blur_left.weight.data[:n_coeff, ...] = local_basis_psf_l.permute(3,1,2,0).repeat(1, self.size_feat, 1, 1) \
                                                                                    .reshape(n_coeff, 1, ks, 1) # 1,1,ks,K -> K**2, 1, ks, 1
        p2s_blur_left.weight.data[n_coeff:, ...] = local_mu
        p2s_blur_right.weight.data[:n_coeff, ...] = local_basis_psf_r.permute(0,3,2,1).repeat(self.size_feat, 1, 1, 1) \
                                                                                    .reshape(n_coeff, 1, ks, 1).permute(0,1,3,2) # 1,1,ks,K -> K**2, 1, 1, ks
        p2s_blur_right.weight.data[n_coeff:, ...] = local_mu.permute(0, 1, 3, 2)
        self.kernel_size = ks
        return p2s_blur_left, p2s_blur_right
        
    # initialize random seed
    def _init_rnd_seed(self, seed_shape):
        rnd_seed = (torch.randn(seed_shape, device=self.device, dtype=torch.float32) + \
                                1j * torch.randn(seed_shape, device=self.device, dtype=torch.float32))
        if not (self.batchN == 1 or self.Batch_unrelated):
            for i in range(1, self.batchN):
                rnd_seed[i] = ar1_white_noise(rnd_seed[i-1], self.temp_corr*self.temp_corr_zer)
        # Set a frame counter
        self.counter = 0
        return rnd_seed
        
    # small method to update random seed               
    def _rnd_seed_update(self):
        if self.batchN == 1 or self.Batch_unrelated:
            # update temporally correlated random seed (do the blur and tilt seperately via an approximation)
            self.rnd_seed = ar1_white_noise(self.rnd_seed, self.temp_corr*self.temp_corr_zer)
        else:
            # -1 for the first one is chosen INTENTIONALLY!! Not an error (as far as I know :P)
            self.rnd_seed[0] = ar1_white_noise(self.rnd_seed[-1], self.temp_corr*self.temp_corr_zer)
            for i in range(1, self.batchN):
                self.rnd_seed[i] = ar1_white_noise(self.rnd_seed[i-1], self.temp_corr*self.temp_corr_zer)

    
    def change_param(self, new_turb_params):
        self.turb_params = new_turb_params
        self._initialize_all()

    def print_param(self):
        tp = self.turb_params
        tp['Cn2'] = self.turb_params['Cn2'].tolist()
        tp['r0'] = self.turb_params['r0'] if type(self.turb_params['r0'])==float else self.turb_params['r0'].tolist()
        tp['img_shape'] = (self.H, self.W)
        return tp
    
    def sample_zernike(self, B, H, W):
        if B!=self.batchN:
            self.turb_params['batch_size'] = B
        if H > self.H or W > self.W:
            self.turb_params['img_shape'] = (H, W)
        self._initialize_all()
        zer = torch.fft.ifft2(torch.sqrt(self.psd_stack * self.W * self.H) * self.rnd_seed, dim=(1,2)).real
        zer = torch.einsum('...wz,zx->...wx', zer[:, :H, :W, :], self.Noll)
        zer[...,:2] *= self.tilt_const
        return zer
        
    def forward(self, img, min_random=1, max_random=1, out_params=False, require_tilt=False):
        """function that does the core of the dfp2s simulation

        Args:
            img (tensor): input image tensor
            see sim_examples directory for details on how to set this up

        Returns:
            tensor: simulated version of input img
        """

        # A technicality for CPU simulation
        if self.device == 'cpu':
            torch.set_flush_denormal(True)
        
        # adapt to new input size
        batchN, channelN, H, W = img.shape
        if batchN!=self.batchN:
            self.turb_params['batch_size'] = batchN
            self._initialize_all()
        
        # Generating the Z-index-wise coefficient planes independently (not yet mixed)
        if self.counter > 0:
            self._rnd_seed_update()
            
        zer = torch.fft.ifft2(torch.sqrt(self.psd_stack * self.W * self.H) * self.rnd_seed, dim=(1,2)).real
        self.counter += self.batchN
        
        # Mixing the Zernike coefficients wrt the Noll matrix (a matrix multiplication)
        # If zer padded because of anti-aliasing, first crop it
        # print(zer.shape, self.W, self.H, W, H)
        
        zer = torch.einsum('...wz,zx->...wx', zer[:, :H, :W, :], self.Noll)
        # zer[...,2:] *= truncnorm.rvs(min_random, max_random)

        # Generating the pixel-shift values
        zer[...,:2] *= self.tilt_const
        pos = zer[...,:2] 

        # setting up the flow array
        yy, xx = torch.meshgrid(torch.arange(0, H, device=self.device), torch.arange(0, W, device=self.device))
        flow = 2.0 * (torch.stack((xx, yy), -1).unsqueeze(0).float() + pos)/(torch.tensor((W, H), device=self.device)-1) - 1.0
        # applying the flow array
        tilt_img = F.grid_sample(img, flow, 'bilinear', padding_mode='border', align_corners=False)
        tilt_img = tilt_img.view(batchN, channelN, H, W)
        # self.turb_params['tilt'] = pos.abs().sum().item() / (batchN * channelN * H * W)
        
        if out_params:
            return self.print_param()
        # Convolving the image with the dictionary, weights will be applied later
        # Forming the blurred image based off of p2s weights

        # Computing the weights of the image dictionary using p2s
        weight_norm = torch.ones((batchN, H, W, self.n_mu), dtype=torch.float32, device=self.device)
        weight = self.mapping(zer[...,2:])
        weight = torch.cat([weight, weight_norm], dim=-1)
        
        ones_img = torch.ones_like(tilt_img)
        big_img = torch.cat((tilt_img.unsqueeze(4), ones_img.unsqueeze(4)), 1)
        big_img = big_img * weight.unsqueeze(1)
        # print(big_img.shape)
        dict_img = self.p2s_blur_right(self.p2s_blur_left(big_img.view(-1, H, W, self.size_feat**2+self.n_mu).permute(0,3,1,2)))
        dict_img = dict_img.view(batchN, -1, self.size_feat**2+self.n_mu, H, W)
        norm_img = dict_img[:, 3:]
        out = torch.sum(dict_img[:, :3], dim=2) / torch.sum(norm_img, dim=2)
        if require_tilt:
            return out, tilt_img, zer
        else:
            return out, zer


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
        
        
# class _P2S(nn.Module):
#     def __init__(self, input_dim=33, hidden_dim=200, output_dim=100):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)
#         self.out = nn.Linear(hidden_dim, output_dim)
#         self.act = nn.LeakyReLU()
    
#     def forward(self, x):
#         y = self.act(self.fc1(x))
#         y = self.act(self.fc2(y))
#         y = self.act(self.fc3(y))
#         out = self.out(y)
#         return out
    
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