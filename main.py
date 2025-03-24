from torchmetrics import Accuracy
from packages import *
import torch
from torch import nn


device = "cuda" if torch.cuda.is_available() else "cpu"
#Initialize the superparameters
NUM_CLASSES = 101
IMG_SIZE = (224,224)
BATCH_SIZE = 12
# VALIDATION_SPLIT = 0.2
EPOCHS = 5
lr = 1e-4 
K = 3
SEED = 42

#Define Distillation superparameters
ALPHA = 0.7  #The balance parameter between the real label loss and the distillation loss.
TEMPERATURE = 3 #The temperature parameter for the softmax function.
acc_fn = Accuracy(task="MULTICLASS",num_classes = NUM_CLASSES).to(device)



def main():
    #Initialize datset
    dataset = Dataset(IMG_SIZE)
    train_dataloader, test_dataloader = dataset.get_dataloader(batch_size = 12,num_workers = 4)
    
    #Teacher model section
    teacher_model = TeacherModel().to(device)
    teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr = lr,
                                          amsgrad = True)
    teacher_loss_fn = nn.CrossEntropyLoss().to(device)
    teacher_model_trainer = TrainModel(train_dataloader, test_dataloader, 
                                       dataset.train_dataset, teacher_model,
                                       teacher_optimizer, teacher_loss_fn, acc_fn )
    #teacher_model_trainer.k_fold_train("food101_teacher_model",5)
    utils.load_model(teacher_model, name = "food101_teacher_model")
  
    #Student model section
    stu_model = StudentModel().to(device)
    stu_optimizer = torch.optim.Adam(stu_model.parameters(), lr = lr * 2 , 
                                     amsgrad = True , )# weight_decay = 1e-4)
    stu_loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stu_optimizer, T_max = 200)
    stu_model_trainer = TrainModel(train_dataloader, test_dataloader, 
                                   dataset.train_dataset, stu_model,
                                   stu_optimizer, stu_loss_fn, acc_fn,
                                   scheduler = scheduler, teacher_model = teacher_model
                                  )
    stu_model_trainer.k_fold_train("food101_stu_model", EPOCHS)
    # utils.load_model(stu_model,name = "food101_stu_model")
    
    

if __name__ == "__main__":
    main()
