import torch
from PIL import Image
from torchvision import transforms

from src.metrics.lpips import LPIPS
import torch.nn as nn

dev = 'cuda'
to_tensor_transform = transforms.Compose([transforms.ToTensor()])
mse_loss = nn.MSELoss()

def calculate_l2_difference(image1, image2, device = 'cuda'):
    if isinstance(image1, Image.Image):
        image1 = to_tensor_transform(image1).to(device)
    if isinstance(image2, Image.Image):
        image2 = to_tensor_transform(image2).to(device)

    mse = mse_loss(image1, image2).item()
    return mse

def calculate_psnr(image1, image2, device = 'cuda'):
    max_value = 1.0
    if isinstance(image1, Image.Image):
        image1 = to_tensor_transform(image1).to(device)
    if isinstance(image2, Image.Image):
        image2 = to_tensor_transform(image2).to(device)
    
    mse = mse_loss(image1, image2)
    psnr = 10 * torch.log10(max_value**2 / mse).item()
    return psnr


loss_fn = LPIPS(net_type='vgg').to(dev).eval()

def calculate_lpips(image1, image2, device = 'cuda'):
    if isinstance(image1, Image.Image):
        image1 = to_tensor_transform(image1).to(device)
    if isinstance(image2, Image.Image):
        image2 = to_tensor_transform(image2).to(device)
    
    loss = loss_fn(image1, image2).item()
    return loss

def calculate_metrics(image1, image2, device = 'cuda', size=(512, 512)):
    if isinstance(image1, Image.Image):
        image1 = image1.resize(size)
        image1 = to_tensor_transform(image1).to(device)
    if isinstance(image2, Image.Image):
        image2 = image2.resize(size)
        image2 = to_tensor_transform(image2).to(device)
        
    l2 = calculate_l2_difference(image1, image2, device)
    psnr = calculate_psnr(image1, image2, device)
    lpips = calculate_lpips(image1, image2, device)
    return {"l2": l2, "psnr": psnr, "lpips": lpips}

def get_empty_metrics():
    return {"l2": 0, "psnr": 0, "lpips": 0}

def print_results(results):
    print(f"Reconstruction Metrics: L2: {results['l2']},\t PSNR: {results['psnr']},\t LPIPS: {results['lpips']}")