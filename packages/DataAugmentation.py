import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

def mixup_data(x : torch.Tensor, 
               y : torch.Tensor,
               alpha : float = 0.4) -> tuple:
    '''Compute mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x : torch.Tensor,
                y : torch.Tensor, 
                alpha : float = 1.0) -> tuple:

    lam = np.random.beta(alpha, alpha) if(alpha > 0) else 1.0

    batch_size, C, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)
    # Randomly generate cutting areas
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # If the crop size is invalid, return the original data directly
    if cut_w <= 0 or cut_h <= 0:
        return x, y, y, 1.0
    
    # Randomly select the crop center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # If the crop area is invalid (width or height is 0), return the original data
    if (bbx2 - bbx1) <= 0 or (bby2 - bby1) <= 0:
        return x, y, y, 1.0
    
    # Using CutMix: Replace the specified area in each image with the corresponding area in the randomly selected image
    x_new = x.clone()
    x_new[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Recalculate lam to reflect the actual mixture ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x_new, y_a, y_b, lam
    
def generate_pseudo_labels(model : nn.Module,
                           unlabeled_loader : DataLoader, 
                           threshold : float = 0.9 ,
                           device : torch.device = "cuda") -> tuple:
    model.eval()
    pseudo_images = []
    pseudo_labels = []
    
    #Create pseudo images and labels
    with torch.inference_mode():
        for img, _ in unlabeled_loader:
            img = img.to(device)
            preds = model(img)
            probs, preds = torch.max(torch.softmax(preds, dim = 1), dim = 1)
            for i in range(img.size(0)):
                if probs[i] < threshold:
                    pseudo_images.append(img[i].cpu())
                    pseudo_labels.append(preds[i].cpu())
                    
    return torch.cat(pseudo_images), torch.cat(pseudo_labels)