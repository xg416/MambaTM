from data.LSDIR_data import LSDIRDataset
from utils.NAFNet_arch import NAFNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import numpy as np
# from tqdm import tqdm
# import matplotlib
# import matplotlib.pyplot as plt
from PIL import Image
import imageio
import argparse
import os, glob, json, random
from typing import List
from utils.NAFNet_arch import NAFNet, NAFNet_zer
from utils.simulator import Simulator
# from utils.simulator_zernike import Simulator as Simulator_zer

# torch.autograd.detect_anomaly()

def tensor_psnr(img1, img2, eps=1e-8):
    mse = torch.mean((img1 - img2)**2)
    # print(mse.max(), mse.min())
    return min(100.0, 20 * torch.log10(1.0 / torch.sqrt(mse+eps)).item())

def eval_function(VAELossParams, kld_weight):
    recons, inp, _, _, mu, log_var = VAELossParams
    recons_loss = F.l1_loss(recons, inp)
    PSNR = tensor_psnr(recons, inp)
    mu = mu.flatten(start_dim=1, end_dim=-1)
    log_var = log_var.flatten(start_dim=1, end_dim=-1)
    # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
    kld_loss = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())
    loss = recons_loss + kld_weight * kld_loss
    
    return {
        "loss": loss,
        "Reconstruction_Loss": recons_loss.detach(),
        "KLD": -kld_loss.detach(),
        "PSNR": PSNR
    }


class ZernikeVAE(nn.Module):
    def __init__(self, train_params: str, val_params: str, sim_path: str, zer_channels: int, img_size: int, init_ks: int, nblocks: int, width: int) -> None:
        super(ZernikeVAE, self).__init__()
        enc_blks = [1, 1, 1, nblocks]
        middle_blk_num = 1
        dec_blks = [1, 1, 1, 1]
        self.img_size = img_size
        
        self.param_list_train = []
        for param_path in train_params:
            for jp in os.listdir(param_path):
                if jp.endswith(".json"):
                    sample = json.load(open(os.path.join(param_path, jp), "r"))
                    cn2 = torch.tensor(sample['Cn2'])
                    sample['Cn2'] = cn2
                    self.param_list_train.append(sample)
                    
        self.param_list_val = []
        for param_path in val_params:
            for jp in os.listdir(param_path):
                if jp.endswith(".json"):
                    sample = json.load(open(os.path.join(param_path, jp), "r"))
                    cn2 = torch.tensor(sample['Cn2'])
                    sample['Cn2'] = cn2
                    self.param_list_val.append(sample)
                    
        self.param_list = self.param_list_train
        param_init = self.param_list[0]
        param_init["kernel_size"] = init_ks
        self.simulator = Simulator(sim_path, param_init).cuda()
        for prm in self.simulator.parameters():
            prm.requires_grad = False

        # build the encoder
        self.encoder = NAFNet(img_channel=zer_channels+1, out_channel=2, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        
        # build the decoder
        self.decoder = NAFNet_zer(in_channels=1, out_channel=1, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    def to_eval(self):
        self.param_list = self.param_list_val
    
    def to_train(self):
        self.param_list = self.param_list_train
        
    # encoding function to map the input to the latent space
    def encode(self, input: Tensor) -> List[Tensor]:
        # pass the input through the encoder
        result = self.encoder(input)
        n_channel = result.shape[1]
        mu = result[:, :n_channel//2, ...]
        log_var = result[:, n_channel//2:, ...]
        return mu, log_var
    
    # decoding function to map the latent space to the reconstructed input
    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z)
        return result
    
    # reparameterization trick to sample from the latent space
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # compute the standard deviation from the log variance
        std = torch.exp(0.5 * logvar)
        # sample random noise
        eps = torch.randn_like(std)
        # compute the sample from the latent space
        return eps * std + mu
    
    def simulation(self, inp_imgs: Tensor) -> List[Tensor]:
        outputs = []
        zers = []
        all_ks = []
        tilt_imgs = []
        with torch.no_grad():
            for i in range(inp_imgs.shape[0]):
                param = random.choice(self.param_list)
                ks = param["kernel_size"] / 100.0
                param["img_shape"] = (self.img_size, self.img_size)
                self.simulator.change_param(param)
                out, ti, zer = self.simulator(inp_imgs[i:i+1], require_tilt=True)
                ks_embedding = torch.ones((*zer.shape[:-1], 1), device=zer.device).float() * ks
                outputs.append(out.detach())
                zers.append(zer.detach())
                tilt_imgs.append(ti.detach())
                all_ks.append(ks_embedding.detach())
                # print(ks)
        return torch.cat(outputs, dim=0), torch.cat(tilt_imgs, dim=0), torch.cat(zers, dim=0), torch.cat(all_ks, dim=0)
    
    # forward pass of the vae
    def forward(self, inp_imgs: Tensor) -> List[Tensor]:
        # encode the input to the latent space
        re_degraded = []
        degraded, tilt_img, zer, ks_map = self.simulation(inp_imgs)
        rec_zer = torch.zeros_like(zer)
        rec_zer[..., :2] = zer[..., :2]
        zer_w_size = torch.cat([zer[..., 2:], ks_map], dim=-1)
        mu, log_var = self.encode(zer_w_size.permute(0,3,1,2))
        log_var = torch.clamp(log_var, min=-10.0, max = 10.0)
        # sample from the latent space
        z = self.reparameterize(mu, log_var)
        re_degraded = self.decoder(tilt_img, z)
        return degraded, re_degraded, zer, z, mu, log_var


def validate(model, dataloader, device):
    running_loss = 0
    model.to_eval()
    model.eval()
    with torch.no_grad():
        for i, x in enumerate(dataloader):
            x = x.to(device)
            predictions = model(x)
            total_loss = eval_function(predictions, KLD_WEIGHT)
            running_loss += total_loss["loss"].item()
    model.train()
    model.to_train()
    return running_loss / len(dataloader)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and restoration')
    parser.add_argument('--iters', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--patch-size', '-ps', dest='patch_size', type=int, default=256, help='patch size')
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=32)
    parser.add_argument('--enc_blocks', type=int, default=8)
    parser.add_argument('--net_width', type=int, default=16)
    parser.add_argument('--load', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    args = get_args()
    # define model hyperparameters
    LR = 0.001 # this is good, no need to change
    IMAGE_SIZE = args.patch_size
    CHANNELS = 3
    BATCH_SIZE = args.batch_size
    ITERS = args.iters
    KLD_WEIGHT = 0.0002  # learning rate down, kld_weight up
    PRINT_FREQ = 100
    ENC_BLOCKS = args.enc_blocks
    NET_WIDTH = args.net_width
    # define the dataset path
    DATASET_PATH = "/home/zhan3275/data/LSDIR/LSDIR"
    train_params = ["/home/zhan3275/data/syn_hybrid_2/train/turb_param", "/home/zhan3275/lab/data/TurbulenceData/static_new/train_params"]
    val_params = ["/home/zhan3275/data/syn_hybrid_2/test/turb_param", "/home/zhan3275/lab/data/TurbulenceData/static_new/test_params"]
    sim_path = "/home/zhan3275/turb/recon/semi/utils"
    EXP_DIR = f"/home/zhan3275/data/log_rnn/VAE/blur_{ENC_BLOCKS}blks_{NET_WIDTH}width_NAF_allparam"
    MODEL_WEIGHTS_PATH = f"{EXP_DIR}/latest.pth"
    MODEL_BEST_WEIGHTS_PATH = f"{EXP_DIR}/best.pth"  
    os.makedirs(EXP_DIR, exist_ok = True) 

    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )
    
    zernike_dataset = LSDIRDataset(DATASET_PATH, transform=train_transforms)
    # Define the size of the validation set
    val_size = int(len(zernike_dataset) * 0.1)  # 10% for validation
    train_size = len(zernike_dataset) - val_size
    torch.manual_seed(416)
    train_dataset, val_dataset = random_split(zernike_dataset, [train_size, val_size])
    # Define the data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    if args.load:
        loaded = torch.load(args.load)
        init_ks = loaded["vae-zernike"]["simulator.p2s_blur_left.weight"].shape[2]
        model = ZernikeVAE(train_params, val_params, sim_path, 33, IMAGE_SIZE, init_ks, ENC_BLOCKS, NET_WIDTH)
        model = model.to(DEVICE)
        try:
            model.load_state_dict(loaded["vae-zernike"])
        except:
            change_checkpoint(model, loaded)
            model.load_state_dict(loaded["vae-zernike"])
    else:
        model = ZernikeVAE(train_params, val_params, sim_path, 33, IMAGE_SIZE, 65, ENC_BLOCKS, NET_WIDTH)
        model = model.to(DEVICE)
        
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ITERS, eta_min=1e-7)

    best_val_loss = float("inf")
    print("Training Started!!")
    current_iter = 1
    # start training by looping over the number of epochs
    running_loss = 0.0
    rec_running_loss = 0.0
    KL_running_loss = 0.0
    PSNR = 0.0
    for epoch in range(ITERS):
        for x in train_dataloader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(x)
            total_loss = eval_function(predictions, KLD_WEIGHT)
            # Backward pass
            total_loss["loss"].backward()
            # Optimizer variable updates
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += total_loss["loss"].item()
            rec_running_loss += total_loss['Reconstruction_Loss'].item()
            KL_running_loss += total_loss['KLD'].item()
            PSNR += total_loss['PSNR']
            if current_iter % PRINT_FREQ == 0:
                # compute average loss for the epoch, reduction="mean", no need average over batch size
                train_loss = running_loss / PRINT_FREQ 
                rec_running_loss = rec_running_loss / PRINT_FREQ
                KL_running_loss = KL_running_loss / PRINT_FREQ
                PSNR = PSNR / PRINT_FREQ
                # compute validation loss for the epoch
                val_loss = validate(model, val_dataloader, DEVICE)
                # save best vae model weights based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {"vae-zernike": model.state_dict()},
                        MODEL_BEST_WEIGHTS_PATH,
                    )
                torch.save(
                    {"vae-zernike": model.state_dict()},
                    MODEL_WEIGHTS_PATH,
                )
                if current_iter % 1000 == 0:
                    torch.save(
                        {"vae-zernike": model.state_dict()},
                        f"{EXP_DIR}/iter_{current_iter}.pth",
                    )
                print(
                    f"Iteration {current_iter}/{ITERS}, "
                    f"Total Loss: {train_loss:.6f}, "
                    f"Reconstruction Loss: {rec_running_loss:.6f}, "
                    f"PSNR: {PSNR:.6f}, "
                    f"KL Loss: {KL_running_loss:.6f}",
                    f"Val Loss: {val_loss:.6f}",
                )
                running_loss = 0.0
                rec_running_loss = 0.0
                KL_running_loss = 0.0
                PSNR = 0.0
            current_iter += 1
            scheduler.step()
