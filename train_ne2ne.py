# python train_ne2ne.py --data-dir ./dataset --batchsize 4 --n-epoch 100

from __future__ import division
import os
import time
import datetime
import argparse
import cv2
from PIL import Image
import numpy as np
import json
from pprint import pprint

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor, _is_pil_image
from torchvision.datasets.folder import has_file_allowed_extension


from arch_unet import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--log-name', type=str, default='ne2ne_unet_b4e400r02')
parser.add_argument('--exp-dir', type=str, default='./results')
parser.add_argument('--data-dir', type=str, default='./dataset')
parser.add_argument('--gpu-devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n-feature', type=int, default=48)
parser.add_argument('--n-channel', type=int, default=1)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n-epoch', type=int, default=400)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--test-batchsize', type=int, default=1)
parser.add_argument('--patchsize', type=int, default=512)
parser.add_argument('--noise-levels', type=int, nargs='+', default=[1, 2, 4, 8, 16])
parser.add_argument('--noise-levels-test', type=int, nargs='+', default=[1])
# 每ckpt-epochs次迭代保存一次模型
parser.add_argument('--ckpt-epochs', type=int, default=50)
# 每print-freq次迭代打印一次
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--plot-epochs', type=int, default=10)
parser.add_argument('--Lambda1', type=float, default=1.0)
parser.add_argument('--Lambda2', type=float, default=1.0)
parser.add_argument('--increase-ratio', type=float, default=2.0)

opt = parser.parse_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices

# 在文件开头添加全局变量
operation_seed_counter = 0

def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.exp_dir, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{:03d}_{}.pth'.format(epoch,name)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))
    
def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)

IMG_EXTENSIONS = ['.png']

def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def pil_loader(path):
    img = Image.open(path)
    return img

def fluore_to_tensor(pic):
    """Convert a ``PIL Image`` to tensor. Range stays the same.
    Only output one channel, if RGB, convert to grayscale as well.
    Currently data is 8 bit depth.
    
    Args:
        pic (PIL Image): Image to be converted to Tensor.
    Returns:
        Tensor: only one channel, Tensor type consistent with bit-depth.
    """
    if not(_is_pil_image(pic)):
        raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        # all 8-bit: L, P, RGB, YCbCr, RGBA, CMYK
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    # PIL image mode: L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)

    img = img.view(pic.size[1], pic.size[0], nchannel)
    
    if nchannel == 1:
        img = img.squeeze(-1).unsqueeze(0)
    elif pic.mode in ('RGB', 'RGBA'):
        # RBG to grayscale: 
        # https://en.wikipedia.org/wiki/Luma_%28video%29
        ori_dtype = img.dtype
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140])
        img = (img[:, :, [0, 1, 2]].float() * rgb_weights).sum(-1).unsqueeze(0)
        img = img.to(ori_dtype)
    else:
        # other type not supported yet: YCbCr, CMYK
        raise TypeError('Unsupported image type {}'.format(pic.mode))

    return img

class DenoisingFolderNe2Ne(torch.utils.data.Dataset):
    """Data loader for denoising dataset for Ne2Ne.
    This loader fetches noisy and clean images, but does not return a noisy target image.
    
    Args:
        root (str): Root directory to the dataset.
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders.
        types (seq, optional): e.g. ['TwoPhoton_BPAE_B', 'Confocal_MICE`].
        test_fov (int, optional): Default 19. 19th fov is test fov.
        captures (int): Number of images within one folder.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        loader (callable, optional): Image loader.
    """
    def __init__(self, root, noise_levels, types=None, test_fov=19,
                 captures=50, transform=None, target_transform=None, loader=pil_loader):
        super().__init__()
        all_noise_levels = [1, 2, 4, 8, 16]
        all_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B',
                     'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R',
                     'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH',
                     'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
        assert all([level in all_noise_levels for level in noise_levels])
        self.noise_levels = noise_levels
        if types is None:
            self.types = all_types
        else:
            assert all([img_type in all_types for img_type in types])
            self.types = types
        self.root = root
        fovs = list(range(1, 20+1))
        fovs.remove(test_fov)
        self.fovs = fovs
        self.captures = captures
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = self._gather_files()

        dataset_info = {'Dataset': 'train Ne2Ne',
                        'Noise levels': self.noise_levels,
                        f'{len(self.types)} Types': self.types,
                        'Fovs': self.fovs,
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))

    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        subdirs = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
                   if (os.path.isdir(os.path.join(root_dir, name)) and name in self.types)]

        for subdir in subdirs:
            gt_dir = os.path.join(subdir, 'gt')
            for noise_level in self.noise_levels:
                if noise_level == 1:
                    noise_dir = os.path.join(subdir, 'raw')
                elif noise_level in [2, 4, 8, 16]:
                    noise_dir = os.path.join(subdir, f'avg{noise_level}')
                for i_fov in self.fovs:
                    noisy_fov_dir = os.path.join(noise_dir, f'{i_fov}')
                    clean_file = os.path.join(gt_dir, f'{i_fov}', 'avg50.png')
                    noisy_captures = []
                    for fname in sorted(os.listdir(noisy_fov_dir))[:self.captures]:
                        if is_image_file(fname):
                            noisy_captures.append(os.path.join(noisy_fov_dir, fname))
                    # randomly select one noisy capture when loading
                    samples.append((noisy_captures, clean_file))
        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        noisy_captures, clean_file = self.samples[index]
        idx = np.random.choice(len(noisy_captures), 1)
        noisy_file = noisy_captures[idx[0]]
        noisy, clean = self.loader(noisy_file), self.loader(clean_file)
        if self.transform is not None:
            noisy = self.transform(noisy)
        if self.target_transform is not None:
            clean = self.target_transform(clean)

        return noisy, clean

    def __len__(self):
        return len(self.samples)

class DenoisingTestMixFolder(torch.utils.data.Dataset):
    """Data loader for the denoising mixed test set.
        data_root/test_mix/noise_level/imgae.png
        type:           test_mix
        noise_level:    5 (+ 1: ground truth)
        captures.png:   48 images in each fov
    Args:
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders
    """

    def __init__(self, root, loader, noise_levels, transform, target_transform):
        super().__init__()
        all_noise_levels = [1, 2, 4, 8, 16] 
    
        assert all([level in all_noise_levels for level in all_noise_levels])
        self.noise_levels = noise_levels

        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._gather_files()

        dataset_info = {'Dataset': 'test_mix',
                        'Noise levels': self.noise_levels,
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))


    
    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        test_mix_dir = os.path.join(root_dir, 'test_mix')
        gt_dir = os.path.join(test_mix_dir, 'gt')
        
        for noise_level in self.noise_levels:
            if noise_level == 1:
                noise_dir = os.path.join(test_mix_dir, 'raw')
            elif noise_level in [2, 4, 8, 16]:
                noise_dir = os.path.join(test_mix_dir, f'avg{noise_level}')

            for fname in sorted(os.listdir(noise_dir)):
                if is_image_file(fname):
                    noisy_file = os.path.join(noise_dir, fname)
                    clean_file = os.path.join(gt_dir, fname)
                    samples.append((noisy_file, clean_file))

        return samples
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        noisy_file, clean_file = self.samples[index]
        noisy, clean = self.loader(noisy_file), self.loader(clean_file)
        if self.transform is not None:
            noisy = self.transform(noisy)
        if self.target_transform is not None:
            clean = self.target_transform(clean)

        return noisy, clean

    def __len__(self):
        return len(self.samples)


def load_denoising_ne2ne_train(root, batch_size, noise_levels, types=None,
    patch_size=256, transform=None, target_transform=None, loader=pil_loader,
    test_fov=19):
    """For Ne2Ne model, use all captures in each fov, randomly select 1 when
    loading.
    files: root/type/noise_level/fov/captures.png
        total 12 x 5 x 20 x 50 = 60,000 images
        raw: 12 x 20 x 50 = 12,000 images
    
    Args:
        root (str):
        batch_size (int): 
        noise_levels (seq): e.g. [1, 2, 4], or [1, 2, 4, 8]
        types (seq, None): e.g. [`microscopy_cell`]
        transform (torchvision.transform): transform to noisy images
        target_transform (torchvision.transform): transforms to clean images
    """
    if transform is None:
        # default to center crop the image from 512x512 to 256x256
        transform = transforms.Compose([
            transforms.CenterCrop(patch_size),
            fluore_to_tensor,
            transforms.Lambda(lambda x: x.float().div(255))
            ])
    target_transform = transform
    dataset = DenoisingFolderNe2Ne(root, noise_levels, types=types, 
        test_fov=test_fov, transform=transform, 
        target_transform=target_transform, loader=pil_loader)
    kwargs = {'num_workers': 0, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        shuffle=True, drop_last=False, **kwargs)

    return data_loader

def load_denoising_test_mix(root, batch_size, noise_levels, loader=pil_loader, 
    transform=None, target_transform=None, patch_size=256):
    """
    files: root/test_mix/noise_level/captures.png
        
    Args:
        root (str):
        batch_size (int): 
        noise_levels (seq): e.g. [1, 2, 4], or [1, 2, 4, 8]
        types (seq, None): e.g.     [`microscopy_cell`]
        transform (torchvision.transform): transform to noisy images
        target_transform (torchvision.transform): transforms to clean images
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.CenterCrop(patch_size),
            fluore_to_tensor,
            transforms.Lambda(lambda x: x.float().div(255))
            ])
    # the same
    target_transform = transform
    dataset = DenoisingTestMixFolder(root, loader, noise_levels, transform, 
        target_transform)
    kwargs = {'num_workers': 0, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        shuffle=True, drop_last=False, **kwargs)

    return data_loader

def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

def generate_mask_pair(img):
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ), dtype=torch.bool, device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ), dtype=torch.bool, device=img.device)
    idx_pair = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]], dtype=torch.int64, device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ), dtype=torch.int64, device=img.device)
    torch.randint(low=0, high=8, size=(n * h // 2 * w // 2, ), generator=get_generator(), out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0, end=n * h // 2 * w // 2 * 4, step=4, dtype=torch.int64, device=img.device).reshape(-1, 1)
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2

def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n, c, h // 2, w // 2, dtype=img.dtype, layout=img.layout, device=img.device)
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage

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


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    # Initialize lists to store validation metrics
    psnr_list = []
    ssim_list = []
    
    # Create output directory for validation results
    output_dir = os.path.join(opt.exp_dir, opt.log_name, systime, 'validation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    train_loader = load_denoising_ne2ne_train(opt.data_dir, batch_size=opt.batchsize, 
                                            noise_levels=opt.noise_levels, 
                                            patch_size=opt.patchsize)
    test_loader = load_denoising_test_mix(opt.data_dir, batch_size=opt.test_batchsize, 
                                      noise_levels=opt.noise_levels_test, 
                                      patch_size=opt.patchsize)

    # Network
    network = UNet(in_nc=opt.n_channel, out_nc=opt.n_channel, n_feature=opt.n_feature)
    if opt.parallel:
        network = torch.nn.DataParallel(network)
    network = network.cuda()

    # Optimizer and scheduler
    optimizer = optim.Adam(network.parameters(), lr=opt.lr)
    num_epoch = opt.n_epoch
    ratio = num_epoch / 100
    scheduler = lr_scheduler.MultiStepLR(optimizer, 
                                        milestones=[
                                            int(20 * ratio) - 1, 
                                            int(40 * ratio) - 1, 
                                            int(60 * ratio) - 1, 
                                            int(80 * ratio) - 1
                                            ], 
                                        gamma=opt.gamma)

    tic = time.time()
    # Training loop
    for epoch in range(1, opt.n_epoch + 1):
        network.train()
        for iteration, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.cuda(), clean.cuda()
            
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
            Lambda = epoch / opt.n_epoch * opt.increase_ratio
            diff = noisy_output - noisy_target
            exp_diff = noisy_sub1_denoised - noisy_sub2_denoised
            
            loss1 = torch.mean(diff**2)
            loss2 = Lambda * torch.mean((diff - exp_diff)**2)
            loss_all = opt.Lambda1 * loss1 + opt.Lambda2 * loss2
            
            loss_all.backward()
            optimizer.step()
            
            if iteration % opt.print_freq == 0:
                print(f'Epoch [{epoch}/{opt.n_epoch}], Iter [{iteration}/{len(train_loader)}], Loss: {loss_all.item():.4f}')

        scheduler.step()

        if epoch % opt.ckpt_epochs == 0 or epoch == opt.n_epoch:
            checkpoint(network, epoch, "model")

        # Validation
        if epoch % opt.plot_epochs == 0:
            network.eval()
            psnr_val = 0.0
            ssim_val = 0.0
            with torch.no_grad():
                for i, (noisy, clean) in enumerate(test_loader):
                    noisy, clean = noisy.cuda(), clean.cuda()
                    denoised = network(noisy)
                    
                    
                    noisy_im = noisy.permute(0,2, 3, 1)
                    denoised_im = denoised.permute(0,2, 3, 1)
                    clean_im = clean.permute(0,2, 3, 1)
                    
                    noisy_im = noisy_im.squeeze(0)
                    denoised_im = denoised_im.squeeze(0)
                    clean_im = clean_im.squeeze(0)
                    
                    noisy255 = (noisy_im * 255).cpu().numpy().astype(np.uint8)
                    denoised255 = (denoised_im * 255).cpu().numpy().astype(np.uint8)
                    clean255 = (clean_im * 255).cpu().numpy().astype(np.uint8)
                    
                    
                    
                    # 计算 PSNR 和 SSIM
                    psnr_val += calculate_psnr(clean255.astype(np.float32), 
                                               denoised255.astype(np.float32))
                    ssim_val += calculate_ssim(clean255.astype(np.float32), 
                                               denoised255.astype(np.float32))
                    
                    # 保存第一张噪声图像和降噪结果
                    if i == 0:
                        
                        # print('clean.shape',clean.shape)    # torch.Size([1, 1, 512, 512])
                        # print('clean_im.shape',clean_im.shape)  #clean255.shape (512 , 512, 1)
                        # print('clean255.shape',clean255.shape)   # clean_im.shape (512, 512, 1)
                        
                        clean_img = clean[0].cpu().numpy().copy()
                        noisy_img = noisy[0].cpu().numpy().copy()
                        denoised_img = denoised[0].cpu().numpy().copy()
                        
                        clean_img = (clean_img * 255).astype(np.uint8)
                        noisy_img = (noisy_img * 255).astype(np.uint8)
                        denoised_img = (denoised_img * 255).astype(np.uint8)
                        # 保存图像
                        # 将 NumPy 数组转换为 PIL 图像
                        clean_img = Image.fromarray(clean_img[0])  # 选择第一个通道
                        noisy_img = Image.fromarray(noisy_img[0])  # 选择第一个通道
                        denoised_img = Image.fromarray(denoised_img[0])  # 选择第一个通道
                        # 保存图像
                        clean_img.save(os.path.join(output_dir, f'epoch_{epoch}_clean.png'))
                        noisy_img.save(os.path.join(output_dir, f'epoch_{epoch}_noisy.png'))
                        denoised_img.save(os.path.join(output_dir, f'epoch_{epoch}_denoised.png'))
            
            psnr_val /= len(test_loader)
            ssim_val /= len(test_loader)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            print(f'Epoch [{epoch}/{opt.n_epoch}], PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}')
            
    tic2 = time.time()
    print("Finished training {} epochs using {} seconds"
        .format(opt.n_epoch, tic2 - tic))

    # 保存所有 PSNR 和 SSIM 结果
    results_file = os.path.join(output_dir, "results.txt")
    with open(results_file, "w") as f:
        for epoch_idx, (psnr, ssim) in enumerate(zip(psnr_list, ssim_list), start=1):
            f.write(f"Epoch {epoch_idx * opt.plot_epochs}: PSNR: {psnr:.2f}, SSIM: {ssim:.4f}\n")

    print(f"Results saved to {results_file}")
