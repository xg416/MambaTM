import numpy as np
import math
import torch

class z_space:
    def __init__(self, params, precomputed, dx, zern_shape, device):
        # H, W: image shape in pixel
        # h, w: image shape in meter
        # HH, WW: image shape to achieve no aliasing 
        # pr, PR: no aliasing range in physics(meters) and image(pixel) 
        self.Cn2 = params['Cn2'].to(device)
        self.L = params['L']
        self.wvl = params['wvl']
        self.D = params['D']
        self.Z = params['num_zern']
        self.input_H, self.input_W = zern_shape[0], zern_shape[1]
        # make sure H and W are odd
        self.H = self.input_H // 2 * 2 + 1
        self.W = self.input_W // 2 * 2 + 1
        self.device = device

        self.zern_space = torch.zeros(self.H, self.W, self.Z-1, device=device)
        
        self.tilt_coeff_final = (self.wvl / (self.D / 2)) ** 2
        self.our_coeff_final = 0.0096932 * (2*np.pi/self.wvl)**2 * 2**(14/3) * \
                                np.pi**(8/3) * (self.D / 2)**(5/3) / np.pi**2

        # Let Cn2 to be tensor forever
        self.M = len(self.Cn2)
        
        self.ints1, self.ints2 = precomputed['ints1'], precomputed['ints2']
        self.int_sample = precomputed['s']
        self.exp_low = precomputed['exp_low']
        self.int_range = precomputed['s_range']
        
        # xx,yy = torch.meshgrid(torch.linspace(-self.W/2, self.W/2, self.W)*dx,
        #                         torch.linspace(-self.H/2, self.H/2, self.H)*dx, indexing='xy')  
        yy, xx = torch.meshgrid(torch.linspace(-self.H/2, self.H/2, self.H)*dx, torch.linspace(-self.W/2, self.W/2, self.W)*dx)      
        self.dist = torch.sqrt(xx**2 + yy**2).to(device)
        self.coeff1, self.coeff2, self.ni = self._func_coeff(self.W, self.H)
        
    def _find_corr(self, idx_map):
        idx_map = idx_map.unsqueeze(0).expand(self.ints1.size(0), idx_map.size(0), idx_map.size(1))
        ints1 = self.ints1.unsqueeze(2).expand(self.ints1.size(0), self.ints1.size(1), idx_map.size(2))
        ints2 = self.ints2.unsqueeze(2).expand(self.ints2.size(0), self.ints2.size(1), idx_map.size(2))
        # out[i][j][k] = ints[i][idx_map[i][j][k]][k]  # if dim == 1
        # 35, 3000, N2
        corr1 = torch.gather(ints1, 1, idx_map)
        corr2 = torch.gather(ints2, 1, idx_map)
        return corr1, corr2
    
    def _find_idx(self, dist_map):
        norm_dist = torch.log((dist_map/100 * (1-torch.exp(-self.exp_low.float())) \
                               + torch.exp(-self.exp_low.float())))/self.exp_low.float() + 1
        dist_idx = norm_dist * (self.int_range - 1)
        return dist_idx.long(), dist_idx-dist_idx.long()

    def _func_coeff(self, N1, N2):
        # x, y = torch.meshgrid(torch.arange(0, N1).to(self.device), torch.arange(0, N2).to(self.device), indexing='xy')
        y, x = torch.meshgrid(torch.arange(0, N2).to(self.device), torch.arange(0, N1).to(self.device))
        thetamat = torch.atan2((y - N2 / 2), (x - N1 / 2))
        # h1: for parallalization, if h1, h==1, else h==5
        ni = torch.zeros((self.Z-1,1,1), device=self.device)
        mi = torch.zeros((self.Z-1,1,1), device=self.device)
        h1 = torch.zeros((self.Z-1,1,1), device=self.device)
        mult = torch.zeros((self.Z-1,1,1), device=self.device)
        for i in range(2, self.Z+1):
            ni[i-2], mi[i-2] = nollToZernInd(i)
            h = tak_indicator(i, i, mi[i-2], mi[i-2]) # only 1 or 5 will be used here
            if h==1:
                h1[i-2] = 1
                mult[i-2] = 1 if i%2 else -1
            else:
                h1[i-2] = 0
                mult[i-2] = 1
        tm = thetamat.unsqueeze(0).expand(self.Z-1, thetamat.shape[0], thetamat.shape[1])
        coeff1 = mult * (-1)**ni * torch.cos(mi * 2 * tm)
        coeff2 = h1 * (-1)**(ni + mi)
        return coeff1, coeff2, ni
            
    def _fijs(self, s_arr):
        s_arr[s_arr >= 99] = 99
        # s_idx is integer, w is the residual
        s_idx, w = self._find_idx(s_arr) 
        corr1, corr2 = self._find_corr(s_idx)
        corr1_, corr2_ = self._find_corr(s_idx+1)
        # bilinear interpolation
        corr1 = corr1 * (1-w) + corr1_ * w
        corr2 = corr2 * (1-w) + corr2_ * w
        return self.coeff1 * corr1 + self.coeff2 * corr2
    
    def _corr_zern(self):
        c_out = torch.zeros_like(self.zern_space)
        for l in range(self.M):
            # multiple ni before temp swap axes
            temp = self._fijs(self.dist * (l+1) / (self.D * (self.M-l))) * torch.sqrt((self.ni+1)**2)
            c_out += temp.permute(1,2,0) * self.Cn2[l] * ((self.M - l) / (self.M + 1)) ** (5 / 3)
        return self.our_coeff_final * c_out * (self.L/self.M) * self.tilt_coeff_final 
    
    def _get_r0(self):
        constant = 0.423 * (2*math.pi/self.wvl)**2 * self.L / self.M
        lm = torch.arange(1, self.M+1, dtype=torch.float32, device=self.device)/(self.M+1) 
        r0 = constant * (lm**(5/3) * self.Cn2).sum()
        return r0**(-3/5)
    
    def generate(self):
        self.zern_space = self._corr_zern()
        return self.zern_space[:self.input_H, :self.input_W, :], self._get_r0()


def tak_indicator(i, j, mi, mj):  
    if (mi != 0) and (mj != 0) and ((i + j) % 2 == 0):
        return 1
    if (mi != 0) and (mj != 0) and ((i + j) % 2 != 0):
        return 2
    if logical_xor((mi == 0) and (j % 2 == 0), (mj == 0) and (i % 2 == 0)):
        return 3
    if logical_xor((mi == 0) and (j % 2 != 0), (mj == 0) and (i % 2 != 0)):
        return 4
    if (mi == 0) and (mj == 0):
        return 5

def logical_xor(str1, str2):
    return bool(str1) ^ bool(str2)

def nollToZernInd(j):
    """
    This function maps the input "j" to the (row, column) of the Zernike pyramid using the Noll numbering scheme.
    Authors: Tim van Werkhoven, Jason Saredy
    See: https://github.com/tvwerkhoven/libtim-py/blob/master/libtim/zern.py
    """
    if (j == 0):
        raise ValueError("Noll indices start at 1, 0 is invalid.")
    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n
    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))

    return n, m
    
    
def nollCovMat(Z):
    """
    This function generates the covariance matrix for a single point source. See the associated paper for details on
    the matrix itself.
    :param Z: Number of Zernike basis functions/coefficients, determines the size of the matrix.
    :param D: The diameter of the aperture (meters)
    :param fried: The Fried parameter value
    :return:
    """
    C = torch.zeros((Z,Z))
    ijs = []
    for i in range(Z):
        ijs.append(nollToZernInd(i+1))
    for i in range(Z):
        ni, mi = ijs[i]
        for j in range(Z):
            nj, mj = ijs[j]
            if (abs(mi) == abs(mj)) and (np.mod(i + j, 2) == 0 or (mi == 0) or (mj == 0)):
                kzz = 2.2698 * (-1)**((ni + nj - 2*mi)/2) * math.sqrt(ni + 1)  * math.sqrt(nj + 1)
                den = math.gamma((-ni + nj + 17.0/3.0)/2.0) * math.gamma((ni - nj + 17.0/3.0)/2.0) * math.gamma((ni + nj + 23.0/3.0)/2.0)
                C[i, j] = kzz * math.gamma((ni + nj - 5/3)/2) / den
            else:
                C[i, j] = 0
    C[0,0] = 1
    return C[1:,1:]