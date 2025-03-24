import torch
from torch import nn
import torch.nn.functional as  F

class LabelSmoothingCrossEntropy(nn.Module):
    """Using CrossEntropyLoss of Label Smoothing """
    def __init__(self, smoothing : float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, student_pred : torch.Tensor,
                y_true : torch.Tensor):
        
        num_classes = student_pred.size(1)  # number of class
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (num_classes - 1)

        # One-hot encoding conversion
        y_true_one_hot = torch.zeros_like(student_pred).scatter_(1, y_true.unsqueeze(1), 1)
        
        #Smoothing
        y_true_smooth = y_true_one_hot * confidence + smooth_value

        #Calculate CrossEntropy
        log_probs = nn.LogSoftmax(dim = 1)(student_pred)
        loss = (-y_true_smooth * log_probs).sum(dim=1).mean()
        return loss
    
#Define the Contrastive Representation Distillation CRD Loss function
class CRDLoss(nn.Module):
    def __init__(self, temperature : float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj_head = nn.Linear(3024, 64)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, student_features : torch.Tensor,
                teacher_features : torch.Tensor):
        
    
        #Calculate the similarity matrix
        logits = torch.mm(student_features, teacher_features.t()) / self.temperature
        # Correct matching index: Assume that the i-th sample in the batch should match the i-th
        labels = torch.arange(logits.shape[0]).to("cuda")

        loss = self.loss_fn(logits, labels)
         
        return loss
    
#Define the distillation loss function
class DistillationLoss(nn.Module):
    def __init__(self, alpha : float = 0.5, 
                 temperature : int = 5):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_pred : torch.Tensor,
                teacher_pred : torch.Tensor,
                stu_loss : torch. Tensor,
                epsilon : float = 1e-7):
        #Define the student loss
        # student_loss = nn.CrossEntropyLoss()(student_pred, y_true)
        #stu_loss = self.stu_loss_fn(student_pred, y_true)
        
        #Define the distillation loss
        teacher_soft = nn.Softmax(dim = 1)(teacher_pred / self.temperature)
        student_soft = nn.Softmax(dim = 1)(student_pred / self.temperature)
        
        #Restric the value to the range [0,1]
        teacher_soft = torch.clamp(teacher_soft, min = epsilon, max = 1.0)
        student_soft = torch.clamp(student_soft, min = epsilon, max = 1.0)
       
        distillation_loss = nn.KLDivLoss(reduction="batchmean")(torch.log(student_soft),
                                                                teacher_soft) * (self.temperature ** 2)
        
        return self.alpha * stu_loss + (1 - self.alpha)  * distillation_loss


def main():
    label_smooth_loss = LabelSmoothingCrossEntropy()
    crd_loss = CRDLoss()
    ditill_loss = DistillationLoss()


if __name__ == "__main__":
    main()