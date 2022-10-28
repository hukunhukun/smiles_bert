import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm

from .optim_schedule import ScheduledOptim

from model.smiles_BERT import BERT,BERTLM


class smiles_BertTrainer:


    def __init__(self,bert:BERT,vocab_size: int,train_dataloader:DataLoader, test_dataloader: DataLoader = None,
    lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, log_freq: int = 10):
        
        # Setup cuda device for BERT training
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and with_cuda) else "cpu")
        self.bert = bert
        self.model = BERTLM(bert, vocab_size).to(self.device)

        self.vocab_size = vocab_size
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader


        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))



    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"

               
        avg_loss = 0.0
        total_correct = 0
        num_not_pad = 0

        i = 0
        if train:
            for x,label in data_loader:
                x = x.to(self.device)
                label = label.to(self.device)
                output = self.model.forward(x)
                pred = output.transpose(1, 2)
                loss = self.criterion(pred, label)

                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
                avg_loss += loss.item()


                if i % self.log_freq == 0:
                    print("iter%d_%s, avg_loss=" % (i, str_code), avg_loss /(i+1)) 
                i = i+1

        else:
            with torch.no_grad():
                for x,label in data_loader:
                    x = x.to(self.device)
                    label = label.to(self.device)
                    output = self.model.forward(x)
                    pred = output.transpose(1, 2)
                    loss = self.criterion(pred, label)


                    avg_loss += loss.item()
                    pred = pred.argmax(dim=1)
                    num = torch.ne(label,0).sum().item()
                    for k in range(len(label)):
                        for j in range(len(label[k])):
                            if label[k][j]==0: label[k][j] = self.vocab_size + 1
                    correct = torch.eq(pred,label).sum().float().item()
                    if i % self.log_freq == 0:
                        print("iter%d_%s, accu =" % (i, str_code), correct /(num)) 
                        print("iter%d_%s, avg_loss=" % (i, str_code), avg_loss /(i+1))
                    total_correct = total_correct + correct
                    num_not_pad = num_not_pad + num
               
                    i = i+1
    
            print("EP%d_%s, avg_accu=" % (epoch, str_code), total_correct / num_not_pad)
        
        
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_loader))


    
    def save(self, epoch, file_path="./output/"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        
        # output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.state_dict(),file_path + 'BERTLM' + ".ep%d" % epoch + ".pt")
        torch.save(self.bert.state_dict(), file_path + 'BERT' + ".ep%d" % epoch + ".pt")
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, file_path)
       