from __future__ import division
import os
import time
import glob
import datetime
import argparse
import numpy as np
import json
import random
from pprint import pprint
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import cv2
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from models.arch_unet import UNet
from utils.metrics import cal_psnr
from utils.data_loader import (load_denoising_n2n_train, 
                               load_denoising_test_mix, fluore_to_tensor)
from utils.practices import OneCycleScheduler, adjust_learning_rate, find_lr
from utils.misc import mkdirs, module_size
from utils.plot import save_samples, save_stats


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Training Ne2Ne')
        self.add_argument('--exp-name', type=str, default='ne2ne', help='experiment name')
        self.add_argument('--exp-dir', type=str, default="./experiments", help='directory to save experiments')        
        self.add_argument('--post', action='store_true', default=False, help='post proc mode')
        self.add_argument('--debug', action='store_true', default=False, help='verbose stdout')
        self.add_argument('--net', type=str, default='UNet')
        # data
        self.add_argument('--data-root', type=str, default="./dataset", help='directory to dataset root')
        self.add_argument('--imsize', type=int, default=256)
        self.add_argument('--in-channels', type=int, default=1)
        self.add_argument('--out-channels', type=int, default=1)
        self.add_argument('--transform', type=str, default='four_crop', choices=['four_crop', 'center_crop'])
        self.add_argument('--noise-levels-train', type=list, default=[1, 2, 4, 8, 16])
        self.add_argument('--noise-levels-test', type=list, default=[1])
        self.add_argument('--test-group', type=int, default=19)
        self.add_argument('--captures', type=int, default=50, help='how many captures in each group to load')
        self.add_argument('--n_feature', type=int, default=48)
        self.add_argument('--parallel', action='store_true')
        # training
        self.add_argument('--epochs', type=int, default=400, help='number of iterations to train')
        self.add_argument('--batch-size', type=int, default=4, help='input batch size for training')
        self.add_argument('--lr', type=float, default=3e-4, help='learnign rate')
        self.add_argument('--wd', type=float, default=0., help="weight decay")
        self.add_argument('--gamma', type=float, default=0.5)
        self.add_argument('--Lambda1', type=float, default=1.0)
        self.add_argument('--Lambda2', type=float, default=1.0)
        self.add_argument('--increase_ratio', type=float, default=2.0)
        self.add_argument('--test-batch-size', type=int, default=2, help='input batch size for testing')
        self.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')
        self.add_argument('--cuda', type=int, default=0, help='cuda number')
        # logging
        self.add_argument('--ckpt-freq', type=int, default=50, help='how many epochs to wait before saving model')
        self.add_argument('--print-freq', type=int, default=100, help='how many minibatches to wait before printing training status')
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
        self.add_argument('--plot-epochs', type=int, default=50, help='how many epochs to wait before plotting test output')
        self.add_argument('--cmap', type=str, default='inferno', help='attach notes to the run dir')

    def parse(self):
        args = self.parse_args()
        date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        # 构建运行目录路径
        args.run_dir = args.exp_dir + '/' + args.exp_name + '_' + date
        
        # 构建检查点目录路径
        args.ckpt_dir = args.run_dir + '/checkpoints'
        args.train_dir = args.run_dir + "/training"
        args.pred_dir = args.run_dir + "/predictions"
        
        
        if not args.post:
            mkdirs([args.run_dir, args.ckpt_dir])
            
        mkdirs([args.train_dir, args.pred_dir])

        # seed,设置随机种子，以确保结果的可复现性
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark=True
        
        print('Arguments:')
        pprint(vars(args))

        if not args.post:
            # 将所有解析的参数以JSON格式保存,以便后续查阅和恢复实验配置
            with open(args.run_dir + "/args.txt", 'w') as args_file:
                json.dump(vars(args), args_file, indent=4)

        return args

args = Parser().parse()

device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
model = UNet(args.in_channels, args.out_channels).to(device)
    
if args.debug:
    print(model)
    print(f"Model size: {module_size(model)}")

if args.transform == 'four_crop':
    # wide field images may have complete noise in center-crop case
    transform = transforms.Compose([
        transforms.FiveCrop(args.imsize),
        transforms.Lambda(lambda crops: torch.stack([
            fluore_to_tensor(crop) for crop in crops[:4]])),
        transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
        ])
elif args.transform == 'center_crop':
    # default transform
    transform = None

systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0


def checkpoint(net, epoch, name):
    save_model_path = args.ckpt_dir
    mkdirs(save_model_path)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


class DataLoader_Imagenet_val(Dataset):
    def __init__(self, data_dir, patch=256):
        super(DataLoader_Imagenet_val, self).__init__()
        self.data_dir = data_dir
        self.patch = patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        # random crop
        H = im.shape[0]
        W = im.shape[1]
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx:xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy:yy + self.patch, :]
        # np.ndarray to torch.tensor
        transformer = transforms.Compose([transforms.ToTensor()])
        im = transformer(im)
        return im

    def __len__(self):
        return len(self.train_fns)


def validation_kodak(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_bsd300(dataset_dir):
    fns = []
    fns.extend(glob.glob(os.path.join(dataset_dir, "test", "*")))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_Set14(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr


# Training Set
TrainingDataset = DataLoader_Imagenet_val(args.data_dir, patch=args.patchsize)
TrainingLoader = DataLoader(dataset=TrainingDataset,
                            num_workers=8,
                            batch_size=args.batchsize,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

# Validation Set
Kodak_dir = os.path.join(args.val_dirs, "Kodak")
BSD300_dir = os.path.join(args.val_dirs, "BSD300")
Set14_dir = os.path.join(args.val_dirs, "Set14")
valid_dict = {
    "Kodak": validation_kodak(Kodak_dir),
    "BSD300": validation_bsd300(BSD300_dir),
    "Set14": validation_Set14(Set14_dir)
}

# Noise adder
noise_adder = AugmentNoise(style=args.noisetype)

# Network
network = UNet(in_nc=args.n_channel,
               out_nc=args.n_channel,
               n_feature=args.n_feature)
if args.parallel:
    network = torch.nn.DataParallel(network)
network = network.cuda()

# about training scheme
num_epoch = args.n_epoch
ratio = num_epoch / 100
optimizer = optim.Adam(network.parameters(), lr=args.lr)
scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     milestones=[
                                         int(20 * ratio) - 1,
                                         int(40 * ratio) - 1,
                                         int(60 * ratio) - 1,
                                         int(80 * ratio) - 1
                                     ],
                                     gamma=args.gamma)
print("Batchsize={}, number of epoch={}".format(args.batchsize, args.n_epoch))

checkpoint(network, 0, "model")
print('init finish')

for epoch in range(1, args.n_epoch + 1):
    cnt = 0

    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

    network.train()
    for iteration, clean in enumerate(TrainingLoader):
        st = time.time()
        clean = clean / 255.0
        clean = clean.cuda()
        noisy = noise_adder.add_train_noise(clean)

        optimizer.zero_grad()

        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)
        with torch.no_grad():
            noisy_denoised = network(noisy)
        noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
        noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

        noisy_output = network(noisy_sub1)
        noisy_target = noisy_sub2
        Lambda = epoch / args.n_epoch * args.increase_ratio
        diff = noisy_output - noisy_target
        exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

        loss1 = torch.mean(diff**2)
        loss2 = Lambda * torch.mean((diff - exp_diff)**2)
        loss_all = args.Lambda1 * loss1 + args.Lambda2 * loss2

        loss_all.backward()
        optimizer.step()
        print(
            '{:04d} {:05d} Loss1={:.6f}, Lambda={}, Loss2={:.6f}, Loss_Full={:.6f}, Time={:.4f}'
            .format(epoch, iteration, np.mean(loss1.item()), Lambda,
                    np.mean(loss2.item()), np.mean(loss_all.item()),
                    time.time() - st))

    scheduler.step()

    if epoch % args.n_snapshot == 0 or epoch == args.n_epoch:
        network.eval()
        # save checkpoint
        checkpoint(network, epoch, "model")
        # validation
        save_model_path = os.path.join(args.save_model_path, args.log_name,
                                       systime)
        validation_path = os.path.join(save_model_path, "validation")
        os.makedirs(validation_path, exist_ok=True)
        np.random.seed(101)
        valid_repeat_times = {"Kodak": 10, "BSD300": 3, "Set14": 20}

        for valid_name, valid_images in valid_dict.items():
            psnr_result = []
            ssim_result = []
            repeat_times = valid_repeat_times[valid_name]
            for i in range(repeat_times):
                for idx, im in enumerate(valid_images):
                    origin255 = im.copy()
                    origin255 = origin255.astype(np.uint8)
                    im = np.array(im, dtype=np.float32) / 255.0
                    noisy_im = noise_adder.add_valid_noise(im)
                    if epoch == args.n_snapshot:
                        noisy255 = noisy_im.copy()
                        noisy255 = np.clip(noisy255 * 255.0 + 0.5, 0,
                                           255).astype(np.uint8)
                    # padding to square
                    H = noisy_im.shape[0]
                    W = noisy_im.shape[1]
                    val_size = (max(H, W) + 31) // 32 * 32
                    noisy_im = np.pad(
                        noisy_im,
                        [[0, val_size - H], [0, val_size - W], [0, 0]],
                        'reflect')
                    transformer = transforms.Compose([transforms.ToTensor()])
                    noisy_im = transformer(noisy_im)
                    noisy_im = torch.unsqueeze(noisy_im, 0)
                    noisy_im = noisy_im.cuda()
                    with torch.no_grad():
                        prediction = network(noisy_im)
                        prediction = prediction[:, :, :H, :W]
                    prediction = prediction.permute(0, 2, 3, 1)
                    prediction = prediction.cpu().data.clamp(0, 1).numpy()
                    prediction = prediction.squeeze()
                    pred255 = np.clip(prediction * 255.0 + 0.5, 0,
                                      255).astype(np.uint8)
                    # calculate psnr
                    cur_psnr = calculate_psnr(origin255.astype(np.float32),
                                              pred255.astype(np.float32))
                    psnr_result.append(cur_psnr)
                    cur_ssim = calculate_ssim(origin255.astype(np.float32),
                                              pred255.astype(np.float32))
                    ssim_result.append(cur_ssim)

                    # visualization
                    if i == 0 and epoch == args.n_snapshot:
                        save_path = os.path.join(
                            validation_path,
                            "{}_{:03d}-{:03d}_clean.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(origin255).convert('RGB').save(
                            save_path)
                        save_path = os.path.join(
                            validation_path,
                            "{}_{:03d}-{:03d}_noisy.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(noisy255).convert('RGB').save(
                            save_path)
                    if i == 0:
                        save_path = os.path.join(
                            validation_path,
                            "{}_{:03d}-{:03d}_denoised.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(pred255).convert('RGB').save(save_path)

            psnr_result = np.array(psnr_result)
            avg_psnr = np.mean(psnr_result)
            avg_ssim = np.mean(ssim_result)
            log_path = os.path.join(validation_path,
                                    "A_log_{}.csv".format(valid_name))
            with open(log_path, "a") as f:
                f.writelines("{},{},{}\n".format(epoch, avg_psnr, avg_ssim))
