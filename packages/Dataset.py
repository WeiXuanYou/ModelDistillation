import torch
import torchvision
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Food101

class Dataset():
    def __init__(self, IMG_SIZE : tuple = (224,224)):   
        #Create a training transform
        self.train_transform = transforms.Compose([ #transforms.RandomResizedCrop(224), # Randomly crop the image to 224x224 and randomly scale it
                                                #  transforms.Resize(256),
                                                #  transforms.CenterCrop(224),
                                                transforms.Resize(IMG_SIZE),
                                                transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally
                                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color dithering
                                                transforms.RandAugment(),  # Advanced Random Enhancement
                                                transforms.ToImage(),
                                                transforms.ToDtype(torch.float32,scale = True),
                                                transforms.Normalize(mean =[0.485,0.456,0.406], #A mean of [0.485,0.485,0.485] (across each color channel)
                                                std = [0.229,0.224,0.225]) #A standard deviation of [0.229,0.224,0.225] (across each color channel)
                                                ])
        #Create a test transform
        self.test_transform = transforms.Compose([ #transforms.Resize(256),
                                                #  transforms.CenterCrop(224),
                                                transforms.Resize(IMG_SIZE),
                                                transforms.ToImage(),
                                                transforms.ToDtype(torch.float32,scale = True),
                                                transforms.Normalize(mean =[0.485,0.456,0.406], #A mean of [0.485,0.485,0.485] (across each color channel)
                                                std = [0.229,0.224,0.225]) #A standard deviation of [0.229,0.224,0.225] (across each color channel)
                                                ])
        #Inverse normalization operation
        self.inv_normalize = transforms.Normalize(
                                                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                                std=[1/0.229, 1/0.224, 1/0.225])
        # Create training and test datasets
        self.train_dataset = Food101(r"dataset",download = True, transform = self.train_transform,split = "train")
        self.test_dataset = Food101(r"dataset",download = True, transform = self.test_transform,split = "test")
 
    def get_dataloader(self, train_dataset : torchvision.datasets = None,
                       test_dataset : torchvision.datasets = None,
                       batch_size : int = 12,
                       num_workers : int = 4) -> tuple:
        #check whether to use the default dataset
        if(not (train_dataset or test_dataset)):
            train_dataset = self.train_dataset
            test_dataset = self.test_dataset

        #Create train and test dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size,
                                    shuffle = True,num_workers = num_workers, pin_memory = True)
        
        test_dataloader  = DataLoader(test_dataset, batch_size = batch_size,
                                    shuffle = False,num_workers = num_workers, pin_memory = True)

        return train_dataloader, test_dataloader

def main():
    dataset_manager = Dataset()
    train_loader, test_loader = dataset_manager.get_dataloader(num_workers = 4)

    #Check dataloader
    print(next(iter(train_loader))[0].shape)
    print(next(iter(test_loader))[0].shape)


if __name__ == "__main__":
    main()