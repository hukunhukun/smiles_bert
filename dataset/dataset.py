

"""
# construct an dictionary based on zinc dataset
# convert smiles to sequence according to dict
# mask part of smiles for pre-training
@copyright 
"""

import csv
import torch
from torch.utils.data import Dataset
from random import randrange,random,shuffle,randint

def load_data(text_dir):
    """load data(smiles) line to line
    input: dataset dir
    output: 
    """
    sen_space = []
    f = open(text_dir, 'r')
    reader = csv.reader(f)
    for row in reader:
        sen_space.append(row)
    f.close()
    
    word1 = []
    processed = []
    # convert list to one dimension
    for i in range(len(sen_space)):
        word1=sen_space[i]
        processed.append(word1[0])
    
    all_smile=[]

    #end="\n"
    element_table=["C","N","B","O","P","S","F","Cl","Br","I","(",")","=","#"]

    # 将[c@@H]这种字符串组合视为一个元素
    for i in range(len(processed)):
        word_space=processed[i]
        word=[]
        j=0
        while j<len(word_space):
            word_space1=[]
            if word_space[j]=="[":
                word_space1.append(word_space[j])
                j=j+1
                while word_space[j]!="]":
                    word_space1.append(word_space[j])
                    j=j+1
                word_space1.append(word_space[j])
                word_space2=''.join(word_space1)
                word.append(word_space2)
                j=j+1
                
            else:
                word_space1.append(word_space[j])
                if j+1<len(word_space):
                    word_space1.append(word_space[j+1])
                    word_space2=''.join(word_space1)
                else:
                    word_space1.insert(0,word_space[j-1])
                    word_space2=''.join(word_space1)

                if word_space2 not in element_table:
                    word.append(word_space[j])
                    j=j+1
                else:#两个字符在element中。只能是Cl,Br
                    word.append(word_space2)
                    j=j+2

        all_smile.append(list(word))
    vocab=[]
    for i in range(len(all_smile)):
        for j in range(len(all_smile[i])):
            if all_smile[i][j] not in vocab:
                vocab.append(all_smile[i][j])
    return vocab, all_smile 




class SMILESBERTdataset(Dataset):
    def __init__(self,smiles_path,seq_len):
        self.vocab,self.all_smiles = load_data(smiles_path)
        self.seq_len = seq_len
        self.word2idx = {'[PAD]' : 0, '[GO]' : 1, '[MASK]' : 2}

        for i, w in enumerate(self.vocab):
            self.word2idx[w] = i + 3


        self.idx2word = {i: w for i, w in enumerate(self.word2idx)}
        self.vocab_size = len(self.word2idx)
  
 
        self.input,self.label = self.create_mask_data()



    def __len__(self):
        return len(self.input)

    
    def __getitem__(self, idx):
        return self.input[idx], self.label[idx]


    def create_mask_data(self):
        max_pred = 5                                # maximum char to be masked in one smile
        maxlen = self.seq_len
        batch = []
        token_list = list()
        for sentence in self.all_smiles:
            arr = [self.word2idx[s] for s in sentence]
            token_list.append(arr)
        
        for i in range(len(token_list)):
            tokens = token_list[i]
            input_ids = [self.word2idx['[GO]']] + tokens
            label = [0 for _ in range(len(input_ids))]
            
        
            #%15 tokens are substitute in one smile
            n_pred =  min(max_pred, max(1, int(len(input_ids) * 0.15)))   # n_pred means the num of char that are substitute 
            
            cand_masked_pos = [i for i, token in enumerate(input_ids) 
                            if token != self.word2idx['[GO]']]               
            shuffle(cand_masked_pos)                                        # random
            
            for pos in cand_masked_pos[:n_pred]:
                label[pos] = input_ids[pos]

                # 80% are relly masked with ["mask"]
                if random() < 0.8:
                    input_ids[pos] = self.word2idx['[MASK]']
                elif random() > 0.9:
                    index = randint(4,self.vocab_size-1)
                    input_ids[pos] = index
                    
            n_pad = maxlen - len(input_ids)
            input_ids.extend([0] * n_pad)
            label.extend([0]* n_pad)

            batch.append([input_ids,label])

        input_ids,label= zip(*batch)
        input_ids,label= torch.LongTensor(input_ids), torch.LongTensor(label)

        return input_ids,label






"""a test for Dataset"""
if __name__ == "__main__":
    MyData = SMILESBERTdataset("./dataset/test.smi",81)
    print(MyData.word2idx)