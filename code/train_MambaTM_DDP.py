import argparse
from PIL import Image
import time
import logging
import os
import numpy as np
import random
from datetime import datetime
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from utils.scheduler import GradualWarmupScheduler
from TM_model import Model
import utils.losses as losses
from utils.utils_image import eval_tensor_imgs
from utils.general import create_log_folder, get_cuda_info, find_latest_checkpoint, change_checkpoint
from utils.NAFNet_arch import NAFNet_zer
from TM_model.archs import flow_warp
from data.dataset_LMDB_train import DataLoaderTurbVideo
import lpips


def init_dist(launcher="slurm", backend='gloo', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Set environment variables needed for PyTorch distributed
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(port)
            
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
        print(f"Process {rank}/{world_size} using GPU {local_rank}")
        return local_rank, world_size

    elif launcher == 'slurm':
        # Get SLURM variables
        rank = int(os.environ.get("SLURM_PROCID", 0))
        # Calculate world_size from SLURM variables
        if "SLURM_NTASKS" in os.environ:
            world_size = int(os.environ["SLURM_NTASKS"])
        else:
            world_size = int(os.environ.get("SLURM_NNODES", 1)) * int(os.environ.get("SLURM_TASKS_PER_NODE", 1).split("(")[0])
            
        # Set environment variables needed for PyTorch distributed
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(port)

        # With SLURM, each process should only see its assigned GPU through CUDA_VISIBLE_DEVICES
        # So we can simply use device 0 for each process
        local_rank = 0
        torch.cuda.set_device(local_rank)

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        return local_rank, world_size
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')

def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and restoration')
    parser.add_argument('--iters', type=int, default=400000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--patch-size', '-ps', dest='patch_size', type=int, default=256, help='Patch size')
    parser.add_argument('--print-period', '-pp', dest='print_period', type=int, default=1000, help='Iterations between checkpointing')
    parser.add_argument('--val-period', '-vp', dest='val_period', type=int, default=5000, help='Iterations for validation')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001, help='Learning rate', dest='lr')
    parser.add_argument('--warmup_iters', type=int, default=5000, help='Warm up iterations')
    parser.add_argument('--lpips_weight', type=float, default=0.01, help='lpips weight')
    parser.add_argument('--rt_weight', type=float, default=0.2, help='Weight of returb loss')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames for the model')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers in dataloader per process')
    parser.add_argument('--train_path', type=str, default='/home/zhan3275/data/lmdb_ATSyn/train_lmdb/', help='Path of training imgs')
    parser.add_argument('--train_info', type=str, default='/home/zhan3275/data/lmdb_ATSyn/train_lmdb/train_info.json', help='Info of training imgs')
    parser.add_argument('--val_path', type=str, default='/home/zhan3275/data/lmdb_ATSyn/test_lmdb/', help='Path of validation imgs')
    parser.add_argument('--val_info', type=str, default='/home/zhan3275/data/lmdb_ATSyn/test_lmdb/test_info.json', help='Info of testing imgs')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--log_path', type=str, default='/home/zhan3275/data/log_rnn/Mamba/', help='Path to save logging files and images')
    parser.add_argument('--reblur_path', type=str, default='/home/zhan3275/turb/recon/semi/model_zoo/NAF_decoder.pth', help='Path to vae reblur model')
    parser.add_argument('--task', type=str, default='turb', help='Choose turb or blur or both')
    parser.add_argument('--run_name', type=str, default='MambaTM3Ref', help='Name of this run')
    parser.add_argument('--start_over', action='store_true')
    parser.add_argument('--model', type=str, default='MambaTM', help='Type of model to construct')
    parser.add_argument('--output_full', action='store_true', help='Output # of frames is the same as the input')
    parser.add_argument('--n_features', type=int, default=16, help='Base # of channels for Conv')
    parser.add_argument('--n_blocks', type=int, default=6, help='# of blocks in middle part of the model')
    parser.add_argument('--future_frames', type=int, default=2, help='Use # of future frames')
    parser.add_argument('--past_frames', type=int, default=2, help='Use # of past frames')
    parser.add_argument('--seed', type=int, default=3275, help='Random seed')
    return parser.parse_args()

class TurbSim(object):
    def __init__(self, nb, width, ckpt_path):
        self.blur_model = NAFNet_zer(in_channels=1, out_channel=1, width=width, middle_blk_num=1,
                                     enc_blk_nums=[1,1,1,nb], dec_blk_nums=[1,1,1,1]).cuda()
        loaded = torch.load(ckpt_path)
        self.blur_model.load_state_dict(loaded["decoder"])
        for prm in self.blur_model.parameters():
            prm.requires_grad = False
        print("TurbSim loaded!")
    
    def reblur(self, img, LPD):
        B, T, C, H, W = img.shape
        img = img.flatten(0, 1)
        LPD = LPD.flatten(0, 1)
        tilt, mu, logvar = LPD[:, :2, :, :], LPD[:, 2:3, :, :], LPD[:, 3:4, :, :]
        log_var = torch.clamp(logvar, min=-10.0, max=10.0)
        tilt_img = flow_warp(img, tilt)
        z = torch.exp(0.5 * logvar) * torch.randn_like(logvar) + mu
        kld = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())
        turb_img = self.blur_model(tilt_img.mean(1, keepdim=True), z)
        return tilt_img.view(B, T, C, H, W), turb_img.view(B, T, 1, H, W), kld

def validate(args, model, val_loader, criterion, c_lpips, TS, iter_count, im_save_freq, im_save_path, device, level):
        test_results_folder = OrderedDict()
        test_results_folder['psnr'] = []
        test_results_folder['ssim'] = []
        eval_loss = 0
        eval_rt_loss = 0
        eval_kld = 0
        eval_lpips = 0
        
        model.eval()
        for s, data in enumerate(val_loader):
            input_ = data[0].to(device)
            tilt_ = data[1].to(device)
            if args.output_full:
                target = data[2].to(device)
            else:
                target = data[2][:, args.past_frames:args.num_frames-args.future_frames, ...].to(device)
                tilt_ = tilt_[:, args.past_frames:args.num_frames-args.future_frames, ...]
            with torch.no_grad():
                output, lpd = model(input_)

                if not args.output_full:
                    input_ = input_[:, args.past_frames:args.num_frames-args.future_frames, ...]
                retilt, returb, kld = TS.reblur(target, lpd)
                loss_returb = criterion(tilt_, retilt) + criterion(returb, input_.mean(2, keepdim=True))
                loss = criterion(output, target) 
                loss_lpips = c_lpips(output.flatten(0,1)*2-1, target.flatten(0,1)*2-1).mean()
                loss_all = loss + args.rt_weight * (loss_returb + 0.0005 * kld)
                eval_loss += loss.item()
                eval_lpips += loss_lpips.item()
                eval_rt_loss += loss_returb.item()
                eval_kld += kld.item()
            
            if s % im_save_freq == 0:
                psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_, save_path=im_save_path, kw=level+'val', iter_count=iter_count)
            else:
                psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_)
            test_results_folder['psnr'] += psnr_batch
            test_results_folder['ssim'] += ssim_batch
                        
        psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
        ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
        eval_loss /= (s + 1)
        eval_rt_loss /= (s + 1)
        eval_kld /= (s + 1)
        eval_lpips /= (s + 1)
        return psnr, ssim, eval_loss, eval_rt_loss, eval_kld, eval_lpips

def main():
    args = get_args()
    local_rank, world_size = init_dist(launcher="slurm", backend='gloo')
    # With SLURM, each process should only see its assigned GPU through CUDA_VISIBLE_DEVICES
    # So we can simply use device 0 for each process
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(local_rank)

    if torch.cuda.is_available():
        # Get the UUID of the GPU (unique per physical device)
        import subprocess
        gpu_uuid = subprocess.check_output('nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader', shell=True).decode().strip()
        print(f"Rank {local_rank} / {world_size} using physical GPU with UUID: {gpu_uuid}")

    # Create logging folder only on rank 0
    run_name = args.run_name + '_' + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    run_path = os.path.join(args.log_path, run_name)
    if dist.get_rank() == 0:
        result_img_path, path_ckpt, path_scripts = create_log_folder(run_path)
        logging.basicConfig(filename=f'{run_path}/recording.log',
                            level=logging.INFO,
                            format='%(levelname)s: %(message)s')
    dist.barrier()
    # Initialize datasets and distributed sampler for training set
    train_dataset = DataLoaderTurbVideo(args.train_path, args.train_info, turb=True, tilt=True, blur=False,
                                        num_frames=args.num_frames, patch_size=args.patch_size, noise=0.0001, is_train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True, prefetch_factor=3)

    # Validation datasets (no sampler; order is not crucial)
    val_dataset_weak = DataLoaderTurbVideo(args.val_path, args.val_info, turb=True, tilt=True, blur=False, level='weak',
                                        num_frames=args.num_frames, patch_size=args.patch_size, noise=0.0001, is_train=False)
    val_loader_weak = DataLoader(dataset=val_dataset_weak, batch_size=args.batch_size*2, shuffle=False,
                                 num_workers=args.num_workers, drop_last=False, pin_memory=True, prefetch_factor=2)

    val_dataset_medium = DataLoaderTurbVideo(args.val_path, args.val_info, turb=True, tilt=True, blur=False, level='medium',
                                        num_frames=args.num_frames, patch_size=args.patch_size, noise=0.0001, is_train=False)
    val_loader_medium = DataLoader(dataset=val_dataset_medium, batch_size=args.batch_size*2, shuffle=False,
                                   num_workers=args.num_workers, drop_last=False, pin_memory=True, prefetch_factor=2)

    val_dataset_strong = DataLoaderTurbVideo(args.val_path, args.val_info, turb=True, tilt=True, blur=False, level='strong',
                                        num_frames=args.num_frames, patch_size=args.patch_size, noise=0.0001, is_train=False)
    val_loader_strong = DataLoader(dataset=val_dataset_strong, batch_size=args.batch_size*2, shuffle=False,
                                   num_workers=args.num_workers, drop_last=False, pin_memory=True, prefetch_factor=2)

    # Build model and wrap with DDP
    model = Model(args, input_size=(args.patch_size, args.patch_size, args.num_frames)).to(device)
    
    new_lr = args.lr
    lpips_weight = args.lpips_weight
    total_iters = args.iters
    warmup_iter = args.warmup_iters

    TS = TurbSim(8, 16, args.reblur_path)
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ######### Resume if needed ###########
    start_iter = 0
    if args.load:
        if args.load == 'latest':
            load_path = find_latest_checkpoint(args.log_path, args.run_name)
            if not load_path:
                print(f'search for the latest checkpoint of {args.run_name} failed!')
        else:
            load_path = args.load
        checkpoint = torch.load(load_path, map_location=device)
        try:
            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
        except:
            change_checkpoint(model, checkpoint, logging)
            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint, strict=False)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.99), eps=1e-8)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, total_iters - warmup_iter, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_iter, after_scheduler=scheduler_cosine)
    if args.load and not args.start_over:
        if 'epoch' in checkpoint.keys():
            start_iter = checkpoint["epoch"] * len(train_dataset)
        elif 'iter' in checkpoint.keys():
            start_iter = checkpoint["iter"]
        if checkpoint.get('optimizer') is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint.keys():
            for i in range(0, start_iter):
                scheduler.step()
        else:
            for i in range(0, start_iter):
                scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        if dist.get_rank() == 0:
            print('------------------------------------------------------------------------------')
            print("==> Resuming Training with learning rate:", new_lr)
            logging.info(f'==> Resuming Training with learning rate: {new_lr}')
            print('------------------------------------------------------------------------------')
    # Loss functions
    criterion_char = losses.CharbonnierLoss()
    criterion_lpips = lpips.LPIPS(net='vgg').to(device)

    if dist.get_rank() == 0:
        logging.info(f'''Starting training:
            Total_iters:     {total_iters}
            Start_iters:     {start_iter}
            Batch size:      {args.batch_size}
            Learning rate:   {new_lr}
            Training size:   {len(train_dataset)}
            val_dataset_weak size: {len(val_loader_weak.dataset)}
            val_dataset_medium size: {len(val_loader_medium.dataset)}
            val_dataset_strong size: {len(val_loader_strong.dataset)}
            Checkpoints:     {path_ckpt}
        ''')

    ######### Train Loop ###########
    best_psnr = 0
    iter_count = start_iter
    current_start_time = time.time()
    current_loss_char = 0
    current_returb_loss = 0
    current_kld_loss = 0
    current_lpips = 0
    train_results_folder = OrderedDict({'psnr': [], 'ssim': []})
    model.train()
    for epoch in range(1000000):
        train_sampler.set_epoch(epoch)
        for data in train_loader:
            model.zero_grad()
            input_ = data[0].to(device)
            tilt_ = data[1].to(device)
            if args.output_full:
                target = data[2].to(device)
            else:
                target = data[2][:, args.past_frames:args.num_frames-args.future_frames, ...].to(device)
                input_ = input_[:, args.past_frames:args.num_frames-args.future_frames, ...]
                tilt_ = tilt_[:, args.past_frames:args.num_frames-args.future_frames, ...]
            
            output, lpd = model(input_)
            retilt, returb, kld = TS.reblur(target, lpd)

            loss_returb = criterion_char(retilt, tilt_) + criterion_char(returb, input_.mean(2, keepdim=True))
            loss = criterion_char(output, target)
            loss_lpips = criterion_lpips(output.flatten(0,1)*2-1, target.flatten(0,1)*2-1).mean()
            loss_all = loss + args.rt_weight * (loss_returb + 0.001 * kld) + lpips_weight * loss_lpips
            loss_all.backward()
            clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

            optimizer.step()
            scheduler.step()
            current_loss_char += loss.item()
            current_returb_loss += loss_returb.item()
            current_kld_loss += kld.item()
            current_lpips += loss_lpips.item()
            iter_count += 1
            
            if iter_count % 500 == 0:
                psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_, save_path=result_img_path if dist.get_rank()==0 else None, kw='train', iter_count=iter_count)
            else:
                psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_)
            train_results_folder['psnr'] += psnr_batch
            train_results_folder['ssim'] += ssim_batch

            if iter_count > start_iter and iter_count % args.print_period == 0:
                psnr_avg = sum(train_results_folder['psnr']) / len(train_results_folder['psnr'])
                ssim_avg = sum(train_results_folder['ssim']) / len(train_results_folder['ssim'])
                if dist.get_rank() == 0:
                    logging.info('Training: iters {:d}/{:d} -Time:{:.6f} -LR:{:.7f} -CharLoss {:8f} -RTLoss {:8f} -KLD {:8f} -PSNR: {:.2f} dB; SSIM: {:.4f}; LPIPS {:8f}'.format(
                        iter_count, total_iters, time.time()-current_start_time, optimizer.param_groups[0]['lr'], \
                        current_loss_char/args.print_period, current_returb_loss/args.print_period, current_kld_loss/args.print_period, psnr_avg, ssim_avg, current_lpips/args.print_period))

                    torch.save({'iter': iter_count, 
                                'psnr': psnr_avg,
                                'state_dict': model.module.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                                'scheduler' : scheduler.state_dict()},
                               os.path.join(path_ckpt, f"model_{iter_count}.pth"))
    
                    torch.save({'iter': iter_count, 
                                'psnr': psnr_avg,
                                'state_dict': model.module.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                                'scheduler' : scheduler.state_dict()},
                               os.path.join(path_ckpt, "latest.pth"))
                current_start_time = time.time()
                current_loss_char = 0
                current_returb_loss = 0
                current_kld_loss = 0
                current_lpips = 0
                train_results_folder = OrderedDict({'psnr': [], 'ssim': []})
                                          
            #### Evaluation ####
            if iter_count > 0 and iter_count % args.val_period == 0 and dist.get_rank() == 0:
                psnr_w, ssim_w, loss_w, loss_rt_w, loss_kld_w, lpips_w = validate(args, model, val_loader_weak, criterion_char, criterion_lpips, TS, iter_count, 200, result_img_path, device, 'weak')
                logging.info('Validation W: Iters {:d}/{:d} - Loss {:8f} - RTLoss {:8f} - KLD {:8f} - PSNR: {:.2f} dB; SSIM: {:.4f}; LPIPS: {:8f}'.format(iter_count, total_iters, loss_w, loss_rt_w, loss_kld_w, psnr_w, ssim_w, lpips_w))
                
                psnr_m, ssim_m, loss_m,loss_rt_m, loss_kld_m, lpips_m = validate(args, model, val_loader_medium, criterion_char, criterion_lpips, TS, iter_count, 200, result_img_path, device, 'medium')
                logging.info('Validation M: Iters {:d}/{:d} - Loss {:8f} - RTLoss {:8f} - KLD {:8f} - PSNR: {:.2f} dB; SSIM: {:.4f}; LPIPS: {:8f}'.format(iter_count, total_iters, loss_m, loss_rt_m, loss_kld_m, psnr_m, ssim_m, lpips_m))
                
                psnr_s, ssim_s, loss_s, loss_rt_s, loss_kld_s, lpips_s = validate(args, model, val_loader_strong, criterion_char, criterion_lpips, TS, iter_count, 200, result_img_path, device, 'strong')
                logging.info('Validation S: Iters {:d}/{:d} - Loss {:8f} - RTLoss {:8f} - KLD {:8f} - PSNR: {:.2f} dB; SSIM: {:.4f}; LPIPS: {:8f}'.format(iter_count, total_iters, loss_s, loss_rt_s, loss_kld_s, psnr_s, ssim_s, lpips_s))
                val_psnr = (psnr_w + psnr_m + psnr_s) / 3
                val_ssim = (ssim_w + ssim_m + ssim_s) / 3
                val_lpips = (lpips_w + lpips_m + lpips_s) / 3
                val_loss = (loss_w + loss_m + loss_s) / 3
                val_rt_loss = (loss_rt_w + loss_rt_m + loss_rt_s) / 3
                val_kld_loss = (loss_kld_w + loss_kld_m + loss_kld_s) / 3

                logging.info('Validation All: Iters {:d}/{:d} - Loss {:8f} - RTLoss {:8f} - KLD {:8f} - PSNR: {:.2f} dB; SSIM: {:.4f}; LPIPS: {:8f}'.format(iter_count, total_iters, val_loss, val_rt_loss, val_kld_loss, val_psnr, val_ssim, val_lpips))
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    torch.save({'iter': iter_count,
                                'psnr': val_psnr,
                                'state_dict': model.module.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                                'scheduler' : scheduler.state_dict(),
                                "after_scheduler" : scheduler.after_scheduler.state_dict()},
                               os.path.join(path_ckpt, "model_best.pth"))
                model.train()
    
    cleanup_distributed()

if __name__ == '__main__':
    main()