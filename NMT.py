# -*- coding: utf-8 -*-
"""NMT.ipynb

"""

# access to drive docs
from google.colab import drive
drive.mount('/content/drive')

"""Translate English docs to Chinese by using seq2seq model"""

#Import Libraries
import torch
import sys
from torch import nn, optim
import random
from torch.nn.functional import softmax
import re
import os

from torchtext.legacy import data
# from torchtext.data import Iterator, BucketIterator
from torchtext.legacy.data import Iterator, BucketIterator
import torch.nn.functional as F
from collections import defaultdict
import string
import dill

from tqdm import tqdm

"""## 1.Load Data

Data:6 Docs:
1. Train Set: train.seg.en.txt train.seg.zh.txt （11743）
2. Validation Set: dev.seg.en.txt dev.seg.zh.txt （2936）
3. Test Set: test.seg.en.txt test.seg.zh.txt （5194）

Use torchtext to load the data, mainly using the following components：

1. Field: Preprocessing Data (Word Tokenization, Lower and Upper Case, Start element or end element, Padding, Dictionary)

2. Dataset: Load Data

3. Iterator: Iterate Data

"""

class Dataloader:
    def __init__(self, batch_size, device, eval=False):
        raw_data = self.read_data("/content/drive/MyDrive/Colab Notebooks/python/data/", test=eval)
        ## Training Mode
        if not eval:
            train_data, dev_data = raw_data
            ##Get Data
            self.id_field = data.Field(sequential=False, use_vocab=False)
            self.en_field = data.Field(init_token='<sos>', eos_token='<eos>', lower=True, include_lengths=True)
            self.zh_field = data.Field(init_token='<sos>', eos_token='<eos>', lower=True)
            self.fields = [("id", self.id_field), ("en", self.en_field), ("zh", self.zh_field)]

            ##Build Dataset
            train_dataset = data.Dataset([data.Example.fromlist([idx, item[0], item[1]], self.fields) for idx, item in enumerate(train_data)], self.fields)
            dev_dataset =  data.Dataset([data.Example.fromlist([idx, item[0], item[1]], self.fields) for idx, item in enumerate(dev_data)], self.fields)
            
            ##Build Iterator
            self.train_iterator= BucketIterator(train_dataset, batch_size=batch_size, device=device, sort_key=lambda x: len(x.en), sort_within_batch=True)
            self.dev_iterator= BucketIterator(dev_dataset, batch_size=batch_size, device=device, sort_key=lambda x: len(x.en), sort_within_batch=True)
            
            ##Build Dic
            self.en_field.build_vocab(train_dataset, min_freq=2)
            self.zh_field.build_vocab(train_dataset, min_freq=2)
            
            ##Store 
            dill.dump(self.en_field, open("/content/drive/MyDrive/Colab Notebooks/python/model/EN.Field", "wb"))
            dill.dump(self.zh_field, open("/content/drive/MyDrive/Colab Notebooks/python/model/ZH.Field", "wb"))

            print("en vocab size:", len(self.en_field.vocab.itos),"zh vocab size:", len(self.zh_field.vocab.itos))
        
        ## Test Mode  
        else:
            test_data = raw_data[-1]
            ##Load data
            self.id_field = data.Field(sequential=False, use_vocab=False)
            self.en_field = dill.load(open("/content/drive/MyDrive/Colab Notebooks/python/model/EN.Field", "rb"))
            self.zh_field = dill.load(open("/content/drive/MyDrive/Colab Notebooks/python/model/ZH.Field", "rb"))
            self.fields = [("id", self.id_field), ("en", self.en_field), ("zh", self.zh_field)]
            
            ##Build Test Set and Iterator
            test_data = data.Dataset([data.Example.fromlist([idx, item[0], item[1]], self.fields) for idx, item in enumerate(test_data)], self.fields)
            self.test_iterator= BucketIterator(test_data, batch_size=batch_size, device=device, train = False, sort_key=lambda x: len(x.en), sort_within_batch = True)   
        
    ##Read data from docs
    def read_data(self, path, test=True, lang1='en', lang2 = 'zh'):
        data = []
        types = ['test'] if test else ['train', 'dev']
        # print(types)
        for type in types:
            sub_data = []
            with open(f"{path}/{type}.seg.{lang1}.txt", encoding='utf-8') as f1, open(f"{path}/{type}.seg.{lang2}.txt", encoding='utf-8') as f2:
                for src, trg in zip(f1, f2):
                    if len(src) > MAX_LEN and len(trg) > MAX_LEN:
                        continue
                    sub_data.append((src.strip(), trg.strip()))
            data.append(sub_data)

        return data

"""## 2.Build Model
Using seq2seq model for NMT, please refer pytorch tutorials https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

You need to complete a seq2seq+attention basic model
Related thesis:
1. attention:https://arxiv.org/abs/1409.0473
2. copy:https://arxiv.org/abs/1603.06393
3. coverage:https://arxiv.org/abs/1601.04811

### 2.1 Encoder
To simplify the code, set the embedding_size and hidden_size to the same value. Using bidirectional GRU for RNN module。
"""

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.vocab_size = input_dim
        ##embedding
        self.embedding = nn.Embedding(input_dim, emb_dim)
        ##rnn
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        ##dropout
        self.dropout = nn.Dropout(dropout)
        ##linear
        self.fc = nn.Linear(enc_hid_dim*2, dec_hid_dim)
        
        
    def forward(self, src_info):
        src, src_len = src_info
       
        ## embedding(+dropout)
        embedded = self.dropout(self.embedding(src))
        ## rnn
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]),dim = 1)))
        
        return outputs, hidden

"""### 2.2 Attention Module
"""

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn = nn.Linear((enc_hid_dim*2)+ dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs, mask):
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #Repeat the hidden state of decoder by src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        ##attention score
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        energy = energy.permute(0, 2, 1)
        #energy = [batch size, dec hid dim, src sent len]
        #v = [dec hid dim]
        
        ##mask the padding, and calculate softmax
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attn = torch.bmm(v,energy).squeeze(1)
        attn = attn.masked_fill(mask, -1e6)
        output = F.softmax(attn, dim=1)
        return output

"""### 2.3 Decoder
"""

class Decoder(nn.Module):
    def __init__(self, output_dim,emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        ## Cont..
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2)+emb_dim, dec_hid_dim)
        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_, hidden, encoder_outputs, mask):
        # input = [batch_size]
        # hidden = [batch_size, dec_hid_dim]
        # encoder_outputs = [src_sent_len, batch_size, enc_hid_dim * 2]
        input_ = input_.unsqueeze(1)
        
        ##embedding
        embedded = self.dropout(self.embedding(input_))
        
        ##Calculate attention score
        attn = self.attention(hidden, encoder_outputs, mask)
        attn = attn.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        ##Based on the attention score, calculate the weighted of the context vector
        weight = torch.bmm(attn, encoder_outputs)
        weight = weight.permute(1, 0, 2)
        embedded = embedded.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weight), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weight = weight.squeeze(0)
        
        output = self.out(torch.cat((output, weight, embedded), dim = 1))
        
        #output = [bsz, output dim]
        
        return output, hidden.squeeze(0), attn.squeeze(0)

"""## 2.4 Seq2seq Model"""

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx

    def forward(self, src_info, trg = None):
                
        src, src_len = src_info
        batch_size = src.shape[1]
        max_len = trg.shape[0] if trg is not None else MAX_LEN
        trg_vocab_size = self.decoder.output_dim
        
        ##Store all the results output by decoder
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        attn_scores = []
        
        ## encoder
        encoder_outputs, hidden = self.encoder(src_info)

        ##Initialise the input of decoder as <sos>token
        input = trg[0, :] if trg is not None else src[0, :]
        #mask = [batch size, src len]
        mask = self.create_mask(src)

        
        ## decode process，Every step decode a token 
        for t in range(1, max_len):
            ## The return: output, hidden, atten_score
            output, hidden, atten_score = self.decoder(input, hidden, encoder_outputs,mask)
            outputs[t] = output
            input = output.argmax(1)
            attn_scores.append(atten_score)
            
        return outputs, torch.cat(attn_scores, dim = 1).to(self.device)

    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        return mask

"""## 3.Training 
Calculate BLEU Score：BLEU is a kind of score the evaluate how effecient the NMT model is. The tools of calculate BLEU Score is include in NLTK Package.
"""

## Calculate bleu
import jieba
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def calculate_bleu(hypothesis, targets, cut=True, verbose=False):
    bleu_scores = []
    for sent, trg in zip(hypothesis, targets):
        trg = trg.strip().lower().split()
        sent = sent.strip()
        if cut:
            trg = list(jieba.cut(''.join(trg).replace("-", "")))
            sent = list(jieba.cut(''.join(sent).replace("-", "")))
 
        bleu = sentence_bleu([trg], sent, weights=(0.5, 0.5, 0., 0.),smoothing_function = SmoothingFunction().method1)
        if verbose:
            print(f"src:{sent.strip()}\ntrg:{trg}\npredict:{sent}\n{bleu}\n")
        bleu_scores.append(bleu)         
    return sum(bleu_scores) / len(bleu_scores)

"""train_iter:
(train_iter will be called in train function)
"""

def train_iter(model, iterator, optimizer, criterion, clip, nl_field = None):
    model.train()
    optimizer.zero_grad()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(iterator)):
        src = batch.en
        trg = batch.zh
        
        ##Cont..
        pred, _ = model(src,trg)
        pred_dim = pred.shape[-1]
        trg = trg[1:].view(-1)
        pred = pred[1:].view(-1, pred_dim)
        
        loss = criterion(pred, trg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()         
        
    return epoch_loss / len(iterator)

"""evaluate_iter：
Evaluate the model with validation set
Greedy search: Select the word with the highest probability until eos detected or reach the maximum length of sentence. (Can also try beam search)
"""

def evaluate_iter(model, iterator, en_field, zh_field, criterion):
    model.eval()

    hypothesis, targets = [], []
    eval_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.en
            trg = batch.zh

            length = len(trg)
            ## Get the result of decoder, calculate the loss, and get the batch_size
            pred, _ = model(src,trg)
            
            src, src_len = batch.en
            batch_size = len(src[0])

            ##decode the sentence, by using greedy search
            for sent in range(batch_size):
                predicts = []  # Predicted output  
                
                for i in range(1,length): 
                ## Cont..
                    tmp = pred[i][sent]
                    tmp = tmp.argmax()
                    if tmp == zh_field.vocab.stoi['<eos>'] or tmp == zh_field.vocab.stoi['<pad>']:
                        break
                    predicts.append(zh_field.vocab.itos[tmp])
                hypothesis.append(' '.join(predicts))
                
                trg_index = [x.item() for x in trg[:, sent]]
                trg_index = trg_index.index(zh_field.vocab.stoi['<eos>']) # Remove <eos>token
                trg_str = ' '.join([zh_field.vocab.itos[x.item()] for x in trg[1: trg_index, sent]])
                targets.append(trg_str) # ground truth
                
            # Calculate bleu
            bleu = calculate_bleu(hypothesis, targets)

            pred_dim = pred.shape[-1]
            pred = pred[1:].view(-1, pred_dim)
            trg = trg[1:].view(-1)

            loss = criterion(pred, trg)
            eval_loss += loss.item()
  
    return bleu, eval_loss / len(iterator)

"""## Train Model"""

def train(dataloader, model, model_output_path):
    print('Start training...')
    
    ## optimizer, criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    best_loss = float('inf')
    best_bleu = float(0)
    for epoch in range(N_EPOCHS):
        
        ## Call train_iter
        train_loss = train_iter(model, dataloader.train_iterator, optimizer, criterion, CLIP)
        bleu, valid_loss = evaluate_iter(model, dataloader.dev_iterator, dataloader.en_field, dataloader.zh_field,criterion)

        ## Save the model for every 5 epoch
        if epoch%5 == 0:
            torch.save(model, f'model_{epoch}.pt')
            
        ## Calculate the loss, if the loss is lower than the best_lost, save the model as 'model_best.pt' and best_lost = loss
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_bleu = bleu
            torch.save(model, 'model_best.pt')
            
        print(f'Best BLEU: {best_bleu:.3f} | Best Loss:{best_loss:.3f} |  Epoch: {epoch:d} |  BLeu： {bleu:.3f} | Loss:{valid_loss}', flush=True)

torch.cuda.get_device_name(0)

## Parameters
MAX_LEN = 256 
TRAIN_BATCH_SIZE = 256
INFERENCE_BATCH_SIZE = 256
HID_DIM = 256
DROPOUT=0.1
N_EPOCHS = 35
CLIP = 1
# device = torch.cuda.get_device_name(0)
device = torch.device('cuda')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH='./model/'

#if out of memory
torch.cuda.empty_cache()

## Define every module
dataloader = Dataloader(TRAIN_BATCH_SIZE, device)
attn = Attention(HID_DIM, HID_DIM)
INPUT_DIM = len(dataloader.en_field.vocab)
OUTPUT_DIM = len(dataloader.zh_field.vocab)
encoder = Encoder(INPUT_DIM, HID_DIM, HID_DIM, HID_DIM, DROPOUT)
decoder = Decoder(OUTPUT_DIM, HID_DIM, HID_DIM, HID_DIM, DROPOUT, attn)
model = Seq2Seq(encoder, decoder, device, dataloader.zh_field.vocab.stoi['<pad>']).to(device)

print(f"Train dataset : {len(dataloader.train_iterator)}")
print(f"Validation dataset : {len(dataloader.dev_iterator)}")

## Training: BATCH_SIZE = 256, HID = 256, MAX_LEN = 256, dropout = 0.1, epoch = 35
train(dataloader, model, model_output_path= MODEL_PATH)

"""## Inference
"""

def inference(model, iterator, en_field, zh_field):
    model.eval()

    with torch.no_grad():
        predict_res = []
        for _, batch in enumerate(iterator):
            src = batch.en
            id, trg = batch.id, batch.zh
            length = len(trg)
            
            ## Result from decoder
            pred, _ = model(src,trg)

            src, src_len = batch.en
            batch_size = len(src[0])
            idx = 0
            for sent in range(batch_size):
                if en_field is not None:

                    eos_index = [x.item() for x in src[:, sent]]
                    
                    eos_index = eos_index.index(en_field.vocab.stoi['<eos>'])
                    src_str = ' '.join([en_field.vocab.itos[x.item()] for x in src[1: eos_index, sent]])
                    sent_id = id[sent]
                predicts = []
                grounds = []
                trg_index = [x.item() for x in trg[:, sent]]
                    
                trg_index = trg_index.index(zh_field.vocab.stoi['<eos>'])
                trg_str = ' '.join([zh_field.vocab.itos[x.item()] for x in trg[1: trg_index, sent]])
                for i in range(1,length): 
                ## Cont..
                    tmp = pred[i][sent]
                    tmp = tmp.argmax()

                    # Remove <eos>token
                    if tmp == zh_field.vocab.stoi['<eos>'] or tmp == zh_field.vocab.stoi['<pad>'] :
                        break               
                
                    predicts.append(zh_field.vocab.itos[tmp])
                grounds.append(trg_str)
                
                predict_res.append((int(sent_id), src_str, ' '.join(predicts), " ".join(grounds)))
                idx += length
    predict_res = [(item[1],item[2], item[3] ) for item in sorted(predict_res, key=lambda x: x[0])]

    bleu = calculate_bleu([i[1] for i in predict_res], [i[2] for i in predict_res])
    return bleu, predict_res

"""Test the best model in test set"""

dataloader = Dataloader(INFERENCE_BATCH_SIZE, device, eval=True)
model_init_path = f"model_best.pt"
# model = Seq2Seq(encoder, decoder, device, dataloader.zh_field.vocab.stoi['<pad>']).to(device)
# model.load_state_dict(torch.load(model_init_path))
model = torch.load(model_init_path)
bleu, predict_output = inference(model, dataloader.test_iterator, dataloader.en_field, dataloader.zh_field)
for item in predict_output:
    src, pred, trg = item
    print(f"src:{src}\ntrg:{trg}\npred:{pred}\n")
print(bleu)



"""## Other results"""

## BATCH_SIZE = 256, HID = 256, MAX_LEN = 256, dropout = 0.1, epoch = 20
train(dataloader, model, model_output_path= MODEL_PATH)

## BATCH_SIZE = 256, HID = 256, MAX_LEN = 256, dropout = 0.2, epoch = 20
train(dataloader, model, model_output_path= MODEL_PATH)

## BATCH_SIZE = 128, HID = 512, MAX_LEN = 256, dropout = 0.2, epoch = 20
train(dataloader, model, model_output_path= MODEL_PATH)

## BATCH_SIZE = 128, HID = 512 , MAX_LEN = 128, dropout = 0.2, epoch = 20
train(dataloader, model, model_output_path= MODEL_PATH)

## BATCH_SIZE = 64, HID = 256, MAX_LEN = 128, dropout = 0.2, epoch = 20 
train(dataloader, model, model_output_path= MODEL_PATH)

## BATCH_SIZE = 64, HID = 512, MAX_LEN = 128, dropout = 0.2, epoch = 20
train(dataloader, model, model_output_path= MODEL_PATH)

