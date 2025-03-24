import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from torch.utils.data import DataLoader
from torchvision import transforms

def visualize_data(data : torch.Tensor,
                  preds : torch.Tensor,
                  labels : torch.Tensor, 
                  class_names : list, 
                  num_samples : int =16):
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))
    for i in range(num_samples):
        axes[i].imshow(data[i].permute(1, 2, 0))
        axes[i].set_title(class_names[labels[i]],
                          color = 'green' if preds[i] == labels[i] else 'red',
                          fontsize = 8, fontweight = 'bold')
        axes[i].axis('off')
    plt.show()

def predict_a_batch_data(model : torch.nn, 
                         data_loader : DataLoader, 
                         device : torch.device = "cuda", 
                         transform : transforms = None):
    model.eval()
    with torch.inference_mode():
        for data, label in data_loader:
            data, label = data.to(device), label.to(device)
            preds = model(data)
            data = transform(data.cpu()) if(transform is not None) else data.cpu()
            return data, preds.softmax(dim = 1).argmax(dim = 1),label

def cal_dataset_distributed(root : str = r'dataset/food-101/meta/test.json'):
    with open(root) as f:
        data_info = json.loads(f.read())
        for key,val in data_info.items():
            print(f"{key} : {len(val)}")


    
def save_model(model : torch.nn, 
               name : str, 
               path : str = r"models/"):
    torch.save(model.state_dict(), f = f"{path}/{name}.pth")

def load_model(model : torch.nn, 
               name : str, 
               path : str = r"models/"):
    model_state_dict = torch.load(f = f"{path}{name}.pth")
    model.load_state_dict(model_state_dict)


def main():
    cal_dataset_distributed()

if __name__ == "__main__":
    main()
