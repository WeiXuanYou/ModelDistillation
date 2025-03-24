from torch.utils.data import DataLoader, Subset
from .Loss import DistillationLoss, CRDLoss
from sklearn.model_selection import KFold
from torchmetrics import Accuracy
from torch import nn
from tqdm import tqdm
import numpy as np
from . import utils
import torch
import torchvision
from .DataAugmentation import cutmix_data, mixup_data

class TrainModel():
    def __init__(self, train_loader : DataLoader,
                 test_loader : DataLoader, 
                 dataset : torchvision.datasets, 
                 model : nn.Module, 
                 optimizer : torch.optim, 
                 loss_fn : nn.Module, 
                 acc_fn : Accuracy,
                 valid_loader : DataLoader = None, 
                 seed : int = 42,
                 teacher_model : nn.Module = None, 
                 save_model : bool = True, 
                 save_best_checkpoint : bool = True,
                 scheduler : torch.optim = None, 
                 device : torch.device = 'cuda'):
        #Initialize device
        self.device = device
        
        #Initializ dataset and dataloaders
        self.dataset = dataset
        self.train_dataloader = train_loader
        self.valid_dataloader = valid_loader
        self.test_dataloader = test_loader
        
        #Initialize the student and the teacher model
        self.model = model
        self.teacher_model = teacher_model

        #Initialize optimizer and loss function
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.distillation_loss_fn = DistillationLoss().to(device)
        self.crd_loss_fn = CRDLoss().to(device)

        #Initialize accuracy function and scheduler
        self.acc_fn = acc_fn
        self.scheduler = scheduler
        
        #Initialize training parameters
        self.seed = seed
        self.save_model = save_model
        self.save_best_checkpoint = save_best_checkpoint
        
    def train_step(self, p : float = 0) -> tuple:
        self.model.train()
        batch_loss, batch_acc = 0, 0
        label_a, label_b = None,None
        
        for data, label in tqdm(self.train_dataloader):
            
            data, label = data.to(self.device), label.to(self.device)
            
            #Randomly choose to use Mixup or CutMix
            if np.random.rand() < p:
                data, label_a, label_b, lam = mixup_data(data, label)
            else:
                if np.random.rand() < p:
                    data, label_a, label_b, lam = cutmix_data(data, label)
            
            preds, stu_feature = self.model(data)
            
            #Check whether to use the MixUp or CutMix data augmentation methods
            if( label_a != None and label_b != None):
                loss = lam * self.loss_fn(preds, label_a) + \
                           (1-lam) * self.loss_fn(preds, label_b)
            else:
                loss = self.loss_fn(preds, label)
            
            #If the student model is being trained, then use the distillation loss and CRD loss
            if(self.teacher_model):
                self.teacher_model.eval()
                # print(self.teacher_model.device)
                teacher_preds, teacher_feature = self.teacher_model(data)

                loss = self.distillation_loss_fn(preds, teacher_preds, loss)
            
                loss_crd = self.crd_loss_fn(stu_feature, teacher_feature)
                loss = loss + 0.1 * loss_crd

            acc = self.acc_fn(preds.softmax(dim = 1).argmax(dim = 1), label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss += loss.item()
            batch_acc += acc.item()
            
        batch_loss /= len(self.train_dataloader)
        batch_acc /= len(self.train_dataloader)
        return batch_loss, batch_acc
    
    def test_step(self) -> tuple:
        if self.teacher_model: self.teacher_model.eval()
        self.model.eval()

        #Initializ batch loss and accuracy
        batch_loss, batch_acc = 0, 0
        
        #Test the model's loss and accuracy
        with torch.inference_mode():
            for data, label in tqdm(self.valid_dataloader):
                data, label = data.to(self.device), label.to(self.device)
                preds, _ = self.model(data)

                loss = self.loss_fn(preds, label)
                # self.teacher_model.eval()
                # teacher_preds = self.teacher_model(data)
                # loss = self.distillation_loss_fn(label, preds, teacher_preds, loss)

                acc = self.acc_fn(preds.softmax(dim = 1).argmax(dim = 1), label)

                batch_loss += loss.item()
                batch_acc += acc.item()

            batch_loss /= len(self.valid_dataloader)
            batch_acc /= len(self.valid_dataloader)

        return batch_loss, batch_acc
        
    def train(self, name : str,
              epochs : int = 5, 
              save_alpha : float = 0.6) -> tuple:
        
        #Set the manual seeds
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        #Define the training and testing losses and accuracies metrics
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        best_acc = 0

        for epoch in range(epochs):
            print("Start Training...")
            train_loss, train_acc = self.train_step()
            print("Start Testing...")
            test_loss, test_acc = self.test_step()

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {train_loss:.2f} | Acc: {train_acc:.2f} | " + 
                  f"Test Loss : {test_loss:.2f} | Test Acc : {test_acc:.2f}")
            
            if(self.scheduler): self.scheduler.step()
            test_acc,test_loss = self.evalate(self.test_dataloader)
            
            #If the accuracy exceeds the current best, store
            if(test_acc * save_alpha + train_acc * (1-save_alpha) > best_acc and self.save_best_checkpoint):
                best_acc = test_acc * save_alpha + train_acc * (1-save_alpha)
                utils.save_model(self.model, name = "best")

        if self.save_model:
            utils.save_model(self.model, name)
        
        return train_losses, test_losses, train_accs, test_accs 

    def k_fold_train(self, name : str, 
                     epochs : int, 
                     save_alpha : float = 0.6, 
                     K : int = 3, 
                     batch_size : int = 12) -> tuple:
        
        kf = KFold(n_splits = K, shuffle = True, random_state = 42)
        #Set the manual seeds
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        #Define the training and testing losses and accuracies metrics
        train_losses, valid_losses = [], []
        train_accs, valid_accs = [], []
        best_acc = 0
        #k-fold training
        for epoch in range(epochs):
            train_loss, valid_loss, train_acc, valid_acc  = (0, 0, 0, 0)
            for fold,(train_idx,val_idx) in enumerate(kf.split(self.dataset)):
                
                print(f"Fold : {fold+1}/{K}  Fold Cross Validating...")
                #Create subset of train and validation dataset
                train_subset = Subset(self.dataset, train_idx)
                val_subset = Subset(self.dataset, val_idx)
                
                #Create DataLoader
                self.train_dataloader = DataLoader(train_subset, batch_size = batch_size,
                                               shuffle = True,num_workers = 8, pin_memory = True)
                self.valid_dataloader = DataLoader(val_subset, batch_size = batch_size,
                                                   shuffle = False, num_workers = 8, pin_memory = True)
                
                
                print("Start Training...")
                k_train_loss, k_train_acc = self.train_step()
                print("Start Testing...")
                k_valid_loss, k_valid_acc = self.test_step()
           
                train_loss += k_train_loss
                valid_loss += k_valid_loss
                train_acc += k_train_acc
                valid_acc += k_valid_acc

                print(f"Epoch {epoch + 1}/{epochs} | Loss: {k_train_loss:.2f} | Acc: {k_train_acc:.2f} | " + 
                f"Test Loss : {k_valid_loss:.2f} | Test Acc : {k_valid_acc:.2f}\n\n")
            
            test_acc,test_loss = self.evalate(self.test_dataloader)
            
            #If the accuracy exceeds the current best, store
            if(test_acc * save_alpha + train_acc * (1-save_alpha) > best_acc and self.save_best_checkpoint):
                best_acc = test_acc * save_alpha + train_acc * (1-save_alpha)
                utils.save_model(self.model, name = "best")
            
            if(self.scheduler): self.scheduler.step()

            train_loss /= K
            valid_loss /= K
            train_acc /= K
            valid_acc /= K
            print(f"Epoch {epoch + 1}/{epochs} | Total Average K Train Loss: {train_loss:.2f} | Total Average K Train Acc: {train_acc:.2f} \n" + 
                f"Total Average K Test Loss: {valid_loss:.2f} | Total Average K Test Acc: {valid_acc:.2f}\n\n")
            
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
            
        if self.save_model:
            utils.save_model(self.model, name)

        return train_losses, valid_losses, train_accs, valid_accs 

    def evalate(self, test_dataloader : DataLoader, 
                device : torch.device = "cuda") -> tuple:

        self.model.eval()
        
        #Evaulate the model's loss and accuracy
        with torch.inference_mode():
            loss, acc = (0,0)
            print("Evaluating..")
            for x,y in tqdm(test_dataloader):
                x,y = x.to(device),y.to(device)
                y_pred, _ = self.model(x)
                batch_loss = self.loss_fn(y_pred,y)
                batch_acc = self.acc_fn(y_pred.argmax(dim = 1),y)

                loss += batch_loss.item()
                acc += batch_acc.item()
              
            print(f"Evaluation | Acc : {acc / len(test_dataloader):.2f}, Loss {loss / len(test_dataloader):.2f} ")
        return acc / len(test_dataloader), loss / len(test_dataloader)

def main():
    pass

if __name__ == "__main__":
    main()