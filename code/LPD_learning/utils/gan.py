import torch
import torch.nn as nn
import numpy as np

class WGAN_loss():
    def compute_G_loss(self, net, fake):
        self.fake_D = net(fake)
        return -self.fake_D.mean()

    def compute_pixel_loss(self,real,fake):
        mse_loss = nn.MSELoss()
        self.pixel_loss = mse_loss(real,fake)
        return self.pixel_loss

    def compute_gradient_penalty(self, real, fake, net_D):
        alpha = torch.rand(real.shape[0], 1, 1, 1).cuda()
        alpha = alpha.expand_as(real)
        compose = alpha * real + (1 - alpha) * fake
        compose = compose.cuda().requires_grad_()
        D_compose = net_D.forward(compose)
        gradients = torch.autograd.grad(outputs=D_compose, inputs=compose, grad_outputs=torch.ones(D_compose.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)
        gradient_penalty = ((gradients[0].norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def compute_D_loss(self, net_D, fake, real):
        self.Lambda = 10
        self.gradient_penalty = self.compute_gradient_penalty(real.data, fake.data, net_D)
        self.D_fake = net_D.forward(fake.detach())
        self.D_real = net_D.forward(real)

        self.loss_D = self.D_fake.mean() - self.D_real.mean()
        self.loss_GP = self.loss_D + self.gradient_penalty * self.Lambda
        return self.loss_GP
    
class Discriminator(nn.Module):
    def __init__(self, input_nc=33, ndf=32, norm=nn.InstanceNorm2d):
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding=1, bias=True), nn.GELU()]
        mult = 1
        for idx in range(4):
            mult_prev = mult
            mult = min(2 ** idx, 8)
            model += [nn.Conv2d(ndf * mult_prev, ndf * mult, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),
                        norm(ndf * mult), nn.GELU()]
        model += [nn.Conv2d(ndf * mult, 8, kernel_size=3, padding=1, bias=True)]
        self.global_model = nn.Sequential(*model)

    def forward(self, inp):
        global_input = inp
        out = self.global_model(inp)
        return out