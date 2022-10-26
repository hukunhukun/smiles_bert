from torch.utils.data import DataLoader

from model.smiles_BERT import BERT
from dataset.dataset import SMILESBERTdataset
from train.pretrain import smiles_BertTrainer

import argparse






def train():

    parser = argparse.ArgumentParser()


    parser.add_argument("-d", "--data_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int, default=512, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=6, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=81, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=128, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")


    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=100, help="printing loss every n iter: setting n")
 
   


    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")


    args = parser.parse_args()

    # data_path = "./dataset/250k_rndm_zinc_drugs_clean.smi"

    print("Loading Train Dataset")
    train_dataset = SMILESBERTdataset(args.data_path,args.seq_len)

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    print("vocab_size",train_dataset.vocab_size)

    print("Building BERT model")
    bert = BERT(train_dataset.vocab_size,hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)


    print("Creating BERT Trainer")
    trainer = smiles_BertTrainer(bert,train_dataset.vocab_size , train_dataloader=train_data_loader, test_dataloader=None,
                        lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, log_freq=args.log_freq )

    print("Training Start")
    epochs = args.epochs
    for epoch in range(epochs):
        trainer.train(epoch)
        
    trainer.save(epoch,args.output_path)

train()

# python3 main.py -d "./dataset/250k_rndm_zinc_drugs_clean.smi" -o "output/bert.model" -s 81 -b 128 --with_cuda True