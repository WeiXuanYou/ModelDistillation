from torch import nn
from torchvision.models import regnet_y_16gf
import torchvision
from torchinfo import summary
import torch.nn.functional as  F

#Define the teacher model
class TeacherModel(nn.Module):
    def __init__(self, num_classes : int = 101 ):
        super().__init__()
        #Create a pretrained model
        self.model_weight = torchvision.models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
        self.model = regnet_y_16gf(weights = self.model_weight)
        
        #Freeze the feature extraction layers
        for params in self.model.stem.parameters():
            params.requires_grad = False
        for params in self.model.trunk_output.parameters():
            params.requires_grad = False

        
        self.stem = self.model.stem
        self.trunk_output = self.model.trunk_output
        self.avg_pool= nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.proj_head = nn.Linear(3024, 64)
        self.fc = nn.Linear(64, num_classes)
        #teacher_weights  = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1#torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
        # teacher_model = vit_b_16(weights = teacher_weights)
       
    def forward(self, x):
        x = self.stem(x)
        x = self.trunk_output(x)
        x_gap = self.avg_pool(x)
        x_gap = self.flatten(x_gap)
        features = self.proj_head(x_gap)
        features = F.normalize(features, dim = 1)
        out = self.fc(features)
        return out, features
    
class ConvBlock(nn.Module):

    def __init__(self, in_channels : int, 
                 out_channels : int,
                 kernel_size : int = 3, 
                 stride :  int = 1, 
                 padding : int = 1, 
                 pool : bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class StudentModel(nn.Module):
    def __init__(self, num_classes : int = 101):
        super().__init__()
        
        self.stem = nn.Sequential(
            ConvBlock(3, 32, kernel_size = 3, stride = 1, padding = 1, pool = True),  # 224 -> 112
            ConvBlock(32, 64, kernel_size = 3, stride = 1, padding = 1, pool = True)  # 112 -> 56
        )
        
        #Intermediate feature introduction layer: increase the depth and number of channels to further introduce features
        self.features = nn.Sequential(
            ConvBlock(64, 128, kernel_size = 3, stride = 1, padding = 1, pool = True),  # 56 -> 28
            ConvBlock(128, 256, kernel_size = 3, stride = 1, padding = 1, pool = True),  # 28 -> 14
            ConvBlock(256, 256, kernel_size = 3, stride = 1, padding = 1, pool = True)   # 14 -> 7
        )
        
        # Global average pooling layer aggregates feature maps into (batch, channels, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        #Projection head: The aggregated special deposit balance is used to determine the bond issuance amount so that the middle layer peers of the bond issuer
        self.proj_head = nn.Linear(256, 64)
        
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x: [batch, 3, 224, 224]
        x = self.stem(x)          
        x = self.features(x)       
        x_gap = self.gap(x)        
        x_gap = x_gap.view(x_gap.size(0), -1)  
        feature = F.normalize(self.proj_head(x_gap), dim=1)  
        out = self.fc(feature)        
        return out, feature         
    
def main():
    import torch
    tensor = torch.rand(1,3,224,224)
    teacher_model = TeacherModel()
    stu_model = StudentModel()
    print(stu_model(tensor)[1].shape)
    print(teacher_model(tensor)[1].shape)
    #check model 
    summary(teacher_model, input_size = (12, 3, 224, 224),
            col_names=["input_size","output_size","num_params","trainable"],
            col_width = 20,
            row_settings = ['var_names'])
    
    summary(stu_model, input_size = (12, 3, 224, 224),
            col_names=["input_size","output_size","num_params","trainable"],
            col_width = 20,
            row_settings = ['var_names'])

if __name__ == "__main__":
    main()