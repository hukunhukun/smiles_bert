from torch.utils.data import DataLoader

from model.smiles_BERT import BERT
from dataset.dataset import SMILESBERTdataset
from train.pretrain import smiles_BertTrainer


def train():

    data_path = "./dataset/250k_rndm_zinc_drugs_clean.smi"

    print("Loading Train Dataset")
    train_dataset = SMILESBERTdataset(data_path,81)

    batch_size = 128

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)


    print("Building BERT model")
    bert = BERT(train_dataset.vocab_size,hidden=512, n_layers=6, attn_heads=8)

    print("Creating BERT Trainer")
    trainer = smiles_BertTrainer(bert,train_dataset.vocab_size , train_dataloader=train_data_loader, test_dataloader=None,
                          )

    print("Training Start")
    epochs = 30
    for epoch in range(epochs):
        trainer.train(epoch)
        
    trainer.save(epoch)

train()