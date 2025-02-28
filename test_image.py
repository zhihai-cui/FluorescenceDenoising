import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from arch_unet import UNet
import cv2

def load_model(check_epochs):
    """加载预训练模型"""
    model = UNet(in_nc=1, out_nc=1, n_feature=48)
    model_names = {
        "e50": "epoch_050_model",
        "e100": "epoch_100_model",
        "e150": "epoch_150_model",
        "e200": "epoch_200_model",
        "e250": "epoch_250_model",
        "e300": "epoch_300_model",
        "e350": "epoch_350_model",
        "e400": "epoch_400_model"
    }

    model_path = os.path.join("results", "ne2ne_unet_b4e400r02", "2025-02-27-18-41", f"{model_names[check_epochs]}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No pretrained model found for {check_epochs}")
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load(model_path))
    return model

def calculate_psnr(img1, img2):
    """计算PSNR值"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """计算SSIM值"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def process_image(model, img, device):
    """处理图像"""
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        
    # 检查是否为RGB图像
    is_rgb = img.mode == 'RGB'
    
    # 转换为tensor
    transform = transforms.Compose([transforms.ToTensor()])
    
    # 如果是灰度图像
    if not is_rgb:
                
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)

        img_tensor = img_tensor.to(device)
        denoised = model(img_tensor)
        
    if is_rgb:
        # 分离RGB通道
        r, g, b = img.split()
        channels = [r, g, b]
        # 处理每个通道
        processed_channels = []
        for channel in channels:
            img_tensor = transform(channel)
            img_tensor = img_tensor.unsqueeze(0)
            # 处理单个通道
            img_tensor = img_tensor.to(device)
            denoised_channel = model(img_tensor)
            processed_channels.append(denoised_channel)
        # 合并处理后的通道
        denoised = Image.merge('RGB', processed_channels)
    
    denoised = denoised.cpu().squeeze(0).clamp(0, 1)
    denoised_img = transforms.ToPILImage()(denoised)
    # 返回处理后的图像
    return denoised_img

def main():
    # 设置输出目录
    output_dir = r"test\output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入图像路径
    input_path = input("\nEnter the path to the noisy image: ")
    if not os.path.exists(input_path):
        print("Input image does not exist!")
        return
        
    # 获取ground truth图像路径
    gt_path = input("\nEnter the path to the ground truth image: ")
    if not os.path.exists(gt_path):
        print("Ground truth image does not exist!")
        return

    # 读取输入图像
    img = Image.open(input_path)
    gt_img = Image.open(gt_path)
    
    # 确保图像尺寸是32的倍数
    w, h = img.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32
    if w != new_w or h != new_h:
        img = img.resize((new_w, new_h), Image.LANCZOS)
        gt_img = gt_img.resize((new_w, new_h), Image.LANCZOS)
    
    
    # 使用GPU如果可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # 定义所有噪声类型
    check_epochs  = ["e50", "e100", "e150", "e200", "e250", "e300", "e350", "e400"]
    
    # 获取基本文件名（不包含扩展名）
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # 定义所有模型的名称和路径类型
    check_epochs  = ["e50", "e100", "e150", "e200", "e250", "e300", "e350", "e400"]
    
    print("\nProcessing with all models:")
    print("-" * 50)
    
    # 使用所有模型处理图像
    for check_epoch in check_epochs:
        try:
            # 加载模型
            model = load_model(check_epoch)
            
            # 处理图像
            denoised_img = process_image(model, img, device)
            
            # 保存结果
            output_path = os.path.join(output_dir, f"{base_name}_{check_epoch}_denoised.png")
            denoised_img.save(output_path)
            
            # 计算PSNR和SSIM
            gt_np = np.array(gt_img)
            denoised_np = np.array(denoised_img)
            
            psnr = calculate_psnr(gt_np, denoised_np)
            ssim = calculate_ssim(gt_np, denoised_np)
            
            print(f"\nResults for {check_epoch}:")
            print(f"PSNR: {psnr:.2f} dB")
            print(f"SSIM: {ssim:.4f}")
            print(f"Saved as: {output_path}")
            
        except FileNotFoundError as e:
            print(f"\nError with {check_epoch}: {e}")
            continue
        except Exception as e:
            print(f"\nUnexpected error with {check_epoch}: {e}")
            continue

if __name__ == "__main__":
    main()
