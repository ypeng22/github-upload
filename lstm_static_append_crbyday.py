import torch
import torch.nn as nn
import numpy as np
from numpy import array
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.nn import functional as F
import random 
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Variables
random.seed(10)
n_features = len(trainxSD[0][0])
n_timesteps = len(trainxSD[0])
epoch = 5
batch_size = 32

class lstm(nn.Module):
    def __init__(self,n_features,seq_length):
        super(lstm, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 40 # number of hidden states
        self.n_layers = 2# number of LSTM layers (stacked)   
        self.l_lstm = torch.nn.LSTM(input_size = self.n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True, dropout = .3)
        self.W_s1 = nn.Linear(self.n_hidden, 40)
        self.W_s2 = nn.Linear(40, 1)
        self.fc_layer = nn.Linear(self.n_hidden, 20)
        self.label = nn.Linear(20, 7)
        self.drop = nn.Dropout(p=.15)
        self.sig = nn.Sigmoid()
        
    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).to(dev)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).to(dev)
        self.hidden = (hidden_state, cell_state)  
        
    def attention(self, lstm_out):
        #print("lstm shape: ", lstm_out.shape)
        attn_weight_matrix = self.W_s1(lstm_out)
        attn_weight_matrix = torch.tanh(attn_weight_matrix)
        attn_weight_matrix = self.W_s2(attn_weight_matrix)
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)
        #print("attn matrix shape: ", attn_weight_matrix.shape)
        return attn_weight_matrix
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.drop(x)
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        #lstm_out = lstm_out.permute(1, 0, 2)
        attn_weight_matrix = self.attention(lstm_out)
        hidden_matrix = torch.bmm(attn_weight_matrix, lstm_out)
        #print('hidden matrix shape: ', hidden_matrix.shape);
		# Let's now concatenate the hidden_matrix and connect it to the fully connected layer.
        #print("hidden matrix post: ", hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]).shape)
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
        #print("fc_out shape: ", fc_out.shape)
        logits = self.label(fc_out)
        #print("logits shape: ", logits.shape)
        return logits
    
random.seed(10)    

dev = torch.device("cuda:0")
mv_net = lstm(n_features, n_timesteps)
mv_net.to(dev)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-3, weight_decay=.0001)


mv_net.train()
for t in range(epoch):
    for b in range(0,len(trainxSD),batch_size):
        inpt = trainxSD[b:b+batch_size,:,:]
        target = trainday[b:b+batch_size]           
        x_batch = torch.tensor(inpt,dtype=torch.float32).to(dev)   
        y_batch = torch.tensor(target,dtype=torch.float32).to(dev)    
        mv_net.init_hidden(x_batch.size(0))
        output = mv_net(x_batch)         
        loss = criterion(output, y_batch) #.view(-1)
        loss.backward()
        optimizer.step()        
        optimizer.zero_grad()
        
        if (b % batch_size * 10) == 0:
            torch.cuda.empty_cache()
    print('step : ' , t , 'loss : ' , loss.item())

#Eval
guess = torch.tensor([]).to(dev)
for b in range(0, len(testxSD), batch_size):
    inpt = testxSD[b:b+batch_size,:,:]
    x_batch = torch.tensor(inpt,dtype=torch.float32).to(dev)   
    mv_net.init_hidden(x_batch.size(0))
    output = mv_net(x_batch)
    #print(output.shape)
    guess = torch.cat((guess, output.detach()))

#preop_cr is at column 63
guess1 = np.zeros((len(testxSD)))
for i in range(len(guess1)):
    baseline = testxSD[i, 0, 63]
    min_thresh = min(baseline + .3, baseline * 1.5)
    if guess[i, 0] > min_thresh or guess[i, 1] > min_thresh:
        guess1[i] = 1
    min_thresh = baseline * 1.5
    if guess[i, 2] > min_thresh or guess[i, 3] > min_thresh or guess[i, 4] > min_thresh or guess[i, 5] > min_thresh or guess[i, 6] > min_thresh:
        guess1[i] = 1
auc = roc_auc_score(testlabel, guess1)
print(auc)

