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
from utils.scheduler import GradualWarmupScheduler
from TM_model import Model
import utils.losses as losses
from utils.utils_image import eval_tensor_imgs
from utils.general import create_log_folder, get_cuda_info, find_latest_checkpoint, change_checkpoint
from utils.NAFNet_arch import NAFNet_zer
from TM_model.archs import flow_warp
from data.dataset_video_train import DataLoaderTurbImage, DataLoaderTurbImageTest


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and restoration')
    parser.add_argument('--iters', type=int, default=400000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--patch-size', '-ps', dest='patch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--print-period', '-pp', dest='print_period', type=int, default=1000, help='number of iterations to save checkpoint')
    parser.add_argument('--val-period', '-vp', dest='val_period', type=int, default=5000, help='number of iterations for validation')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001, help='Learning rate', dest='lr')
    parser.add_argument('--warmup_iters', type=int, default=5000, help='warm up iterations')
    parser.add_argument('--rt_weight', type=float, default=0.2, help='returb module LR weight')
    parser.add_argument('--num_frames', type=int, default=16, help='number of frames for the model')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers in dataloader')
    parser.add_argument('--train_path', type=str, default='~/lab/data/TurbulenceData/static_new/train_static', help='path of training imgs')
    parser.add_argument('--val_path', type=str, default='~/lab/data/TurbulenceData/static_new/test_static', help='path of validation imgs')  
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--log_path', type=str, default='~/data/log_rnn/Mamba/', help='path to save logging files and images')
    parser.add_argument('--reblur_path', type=str, default='~/turb/recon/semi/model_zoo/NAF_decoder.pth', help='path to vae reblur model')
    parser.add_argument('--task', type=str, default='turb', help='choose turb or blur or both')
    parser.add_argument('--run_name', type=str, default='MambaTM_NOLPD', help='name of this running')
    parser.add_argument('--start_over', action='store_true')

    parser.add_argument('--model', type=str, default='MambaTM_NOLPD', help='type of model to construct')
    parser.add_argument('--output_full', action='store_true', help='output # of frames is the same as the input')
    parser.add_argument('--version', type=str, default='v2', help='version of Mamba')
    parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
    parser.add_argument('--n_blocks', type=int, default=4, help='# of blocks in middle part of the model')
    parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
    parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
    parser.add_argument('--seed', type=int, default=3275, help='random seed')
    return parser.parse_args()


def validate(args, model, val_loader, criterion, TS, iter_count, im_save_freq, im_save_path, device, level):
        test_results_folder = OrderedDict()
        test_results_folder['psnr'] = []
        test_results_folder['ssim'] = []
        eval_loss = 0
        eval_rt_loss = 0
        eval_kld = 0
        model.eval()
        for s, data in enumerate(val_loader):
            input_ = data[0].to(device)
            tilt_ = data[1].to(device)
            if args.output_full:
                target = data[2].to(device)
            else:
                target = data[2][:, args.past_frames:args.num_frames-args.future_frames, ...].to(device)
            with torch.no_grad():
                output, lpd = model(input_)
                if not args.output_full:
                    input_ = input_[:, args.past_frames:args.num_frames-args.future_frames, ...]
                    output = output[:, args.past_frames:args.num_frames-args.future_frames, ...]
                    tilt_ = tilt_[:, args.past_frames:args.num_frames-args.future_frames, ...]
    
                retilt, returb, kld = TS.reblur(target, lpd)
                loss_returb = criterion(tilt_, retilt) + criterion(returb, input_.mean(2, keepdim=True))
                loss = criterion(output, target) 
                loss_all = loss + args.rt_weight * (loss_returb + 0.0002 * kld)
                eval_loss += loss.item()
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
        return psnr, ssim, eval_loss, eval_rt_loss, eval_kld

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
        tilt, mu, logvar = LPD[:,:2,:,:], LPD[:,2:3,:,:], LPD[:,3:4,:,:]
        log_var = torch.clamp(logvar, min=-10.0, max = 10.0)
        tilt_img = flow_warp(img, tilt)
        z = torch.exp(0.5 * logvar) * torch.randn_like(logvar) + mu
        kld = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())
        turb_img = self.blur_model(tilt.mean(1, keepdim=True), z)
        return tilt_img.view(B, T, C, H, W), turb_img.view(B, T, 1, H, W), kld
    
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = args.run_name + '_' + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    run_path = os.path.join(args.log_path, run_name)
    if not os.path.exists(run_path):
        result_img_path, path_ckpt, path_scipts = create_log_folder(run_path)
    logging.basicConfig(filename=f'{run_path}/recording.log', \
                        level=logging.INFO, format='%(levelname)s: %(message)s')
    gpu_count = torch.cuda.device_count()
    get_cuda_info(logging)
    
    train_dataset = DataLoaderTurbImage(rgb_dir=args.train_path, num_frames=args.num_frames, total_frames=50, \
        im_size=args.patch_size, noise=0.0004, other_mod='tilt',is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, \
                              drop_last=True, pin_memory=True, prefetch_factor=3)

    val_dataset = DataLoaderTurbImage(rgb_dir=args.val_path, num_frames=args.num_frames, total_frames=50, \
        im_size=args.patch_size, noise=0.0004, other_mod='tilt', is_train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size*2, shuffle=True, num_workers=args.num_workers, \
                            drop_last=True, pin_memory=True, prefetch_factor=3)

    model = Model(args, input_size=(args.patch_size, args.patch_size, args.num_frames)).cuda()
    TS = TurbSim(8, 16, args.reblur_path)
    
    new_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.99), eps=1e-8)
    ######### Scheduler ###########
    total_iters = args.iters
    start_iter = 0
    warmup_iter = args.warmup_iters
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, total_iters-warmup_iter, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_iter, after_scheduler=scheduler_cosine)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ######### Resume ###########
    if args.load:
        if args.load == 'latest':
            load_path = find_latest_checkpoint(args.log_path, args.run_name)
            if not load_path:
                print(f'search for the latest checkpoint of {args.run_name} failed!')
        else:
            load_path = args.load
        checkpoint = torch.load(load_path)
        try:
            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
        except:
            change_checkpoint(model, checkpoint, logging)
            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
        if not args.start_over:
            if 'epoch' in checkpoint.keys():
                start_iter = checkpoint["epoch"] * len(train_dataset)
            elif 'iter' in checkpoint.keys():
                start_iter = checkpoint["iter"] 
            if checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            for i in range(0, start_iter):
                scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            print('------------------------------------------------------------------------------')
            print("==> Resuming Training with learning rate:", new_lr)
            logging.info(f'==> Resuming Training with learning rate: {new_lr}')
            print('------------------------------------------------------------------------------')
            


    if gpu_count > 1:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(gpu_count)]).cuda()

    ######### Loss ###########
    criterion_char = losses.CharbonnierLoss()
    # criterion_edge = losses.EdgeLoss3D()
    
    logging.info(f'''Starting training:
        Total_iters:     {total_iters}
        Start_iters:     {start_iter}
        Batch size:      {args.batch_size}
        Learning rate:   {new_lr}
        Training size:   {len(train_dataset)}
        val_dataset size: {len(val_dataset)}
        Checkpoints:     {path_ckpt}
    ''')
    
    ######### train ###########
    best_psnr = 0
    iter_count = start_iter

    current_start_time = time.time()
    current_loss = 0
    current_returb_loss = 0
    current_kld_loss = 0
    train_results_folder = OrderedDict()
    train_results_folder['psnr'] = []
    train_results_folder['ssim'] = []
    
    model.train()
    for epoch in range(1000000):
        for data in train_loader:
            model.zero_grad()
            
            input_ = data[0].to(device)
            tilt_ = data[1].to(device)
            output, lpd = model(input_)
            
            
            if args.output_full:
                target = data[2].to(device)
            else:
                target = data[2][:, args.past_frames:args.num_frames-args.future_frames, ...].to(device)
                output = output[:, args.past_frames:args.num_frames-args.future_frames, ...]
                input_ = input_[:, args.past_frames:args.num_frames-args.future_frames, ...]
                tilt_ = tilt_[:, args.past_frames:args.num_frames-args.future_frames, ...]
            
            retilt, returb, kld = TS.reblur(target, lpd)
            
            loss_returb = criterion_char(tilt_, retilt) + criterion_char(returb, input_.mean(2, keepdim=True))
            loss = criterion_char(output, target) 
            loss_all = loss + args.rt_weight * (loss_returb + 0.0002 * kld)
            # loss = criterion_char(output, target) + 0.05*criterion_edge(output, target)
            loss_all.backward()
            clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

            optimizer.step()
            scheduler.step()
            current_loss += loss.item()
            current_returb_loss += loss_returb.item()
            current_kld_loss += kld.item()
            # print(scheduler.get_lr()[0], scheduler.after_scheduler.get_last_lr()[0], optimizer.param_groups[0]['lr'], scheduler.after_scheduler.last_epoch
            
            iter_count += 1
            if iter_count % 500 == 0:
                psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_, save_path=result_img_path, kw='train', iter_count=iter_count)
            else:
                psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_)
            train_results_folder['psnr'] += psnr_batch
            train_results_folder['ssim'] += ssim_batch

            if iter_count>start_iter and iter_count % args.print_period == 0:
                psnr = sum(train_results_folder['psnr']) / len(train_results_folder['psnr'])
                ssim = sum(train_results_folder['ssim']) / len(train_results_folder['ssim'])
                
                logging.info('Training: iters {:d}/{:d} -Time:{:.6f} -LR:{:.7f} -Loss {:8f} -RTLoss {:8f} -KLD {:8f} -PSNR: {:.2f} dB; SSIM: {:.4f}'.format(
                    iter_count, total_iters, time.time()-current_start_time, optimizer.param_groups[0]['lr'], \
                    current_loss/args.print_period, current_returb_loss/args.print_period, current_kld_loss/args.print_period, psnr, ssim))

                torch.save({'iter': iter_count, 
                            'psnr': psnr,
                            'state_dict': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'scheduler' : scheduler.state_dict()
                            }, os.path.join(path_ckpt, f"model_{iter_count}.pth")) 

                torch.save({'iter': iter_count, 
                            'psnr': psnr,
                            'state_dict': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'scheduler' : scheduler.state_dict()
                            }, os.path.join(path_ckpt, "latest.pth")) 
                current_start_time = time.time()
                current_loss = 0
                current_returb_loss = 0
                current_kld_loss = 0
                train_results_folder = OrderedDict()
                train_results_folder['psnr'] = []
                train_results_folder['ssim'] = []
                                          
            #### Evaluation ####
            if iter_count>0 and iter_count % args.val_period == 0:
                psnr, ssim, val_loss, val_rt_loss, val_kld_loss = validate(args, model, val_loader, criterion_char, TS, iter_count, 200, result_img_path, device, 'static')
                logging.info('Validation All: Iters {:d}/{:d} - Loss {:8f} - RTLoss {:8f} - KLD {:8f} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(iter_count, total_iters, val_loss, val_rt_loss, val_kld_loss, psnr, ssim))
                if psnr > best_psnr:
                    best_psnr = psnr
                    torch.save({'iter': iter_count,
                                'psnr': psnr,
                                'state_dict': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                                'scheduler' : scheduler.state_dict()
                                }, os.path.join(path_ckpt, "model_best.pth"))
                model.train()
                
if __name__ == '__main__':
    main()
