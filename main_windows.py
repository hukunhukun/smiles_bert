from torch.utils.data import DataLoader,random_split
import torch
from model.smiles_BERT import BERT
from dataset.dataset import SMILESBERTdataset
from train.pretrain import smiles_BertTrainer








def train(seq_len,batch_size,hidden,layers,attn_heads,epochs,output_path):



    data_path = "./dataset/250k_rndm_zinc_drugs_clean.smi"

    print("Loading Train Dataset")
    smiles_dataset = SMILESBERTdataset(data_path,seq_len)

    train_dataset,eval_dataset,test_dataset=random_split(smiles_dataset,[round(0.8*len(smiles_dataset)),len(smiles_dataset)- round(0.8*len(smiles_dataset)) -round(0.1*len(smiles_dataset)),round(0.1*len(smiles_dataset))],generator=torch.Generator().manual_seed(42)) 


    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("vocab_size",smiles_dataset.vocab_size)

    print("Building BERT model")
    bert = BERT(smiles_dataset.vocab_size,hidden=hidden, n_layers=layers, attn_heads=attn_heads)


    print("Creating BERT Trainer")
    trainer = smiles_BertTrainer(bert,smiles_dataset.vocab_size , train_dataloader=train_data_loader, test_dataloader=test_data_loader)

    print("Training Start")
    epochs = epochs
    for epoch in range(epochs):
        trainer.train(epoch)
        
    trainer.save(epoch,output_path)

    print("Testing Start")
    trainer.test(1)

if __name__ == "__main__":
    train(81,256,512,6,8,10,"./output/")

# python3 main.py -d "./dataset/250k_rndm_zinc_drugs_clean.smi" -o "output/" -s 81 -b 128 --with_cuda True