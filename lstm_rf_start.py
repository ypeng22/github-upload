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
n_features = len(trainxSD1[0][0])
n_timesteps = len(trainxSD1[0])
epoch = 5
batch_size = 32

class lstm(nn.Module):
    def __init__(self,n_features,seq_length):
        super(lstm, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 48 # number of hidden states
        self.n_layers = 2# number of LSTM layers (stacked)   
        self.l_lstm = torch.nn.LSTM(input_size = self.n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True, dropout = .3)
        self.W_s1 = nn.Linear(self.n_hidden, 20)
        self.W_s2 = nn.Linear(20, 1)
        self.fc_layer = nn.Linear(self.n_hidden, 20)
        self.label = nn.Linear(20, 2)
        self.drop = nn.Dropout(p=.15)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)
        
    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).to(dev)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).to(dev)
        self.hidden = (hidden_state, cell_state)
        #print(type(self.hidden))
        
    def attention(self, lstm_out):
        #print("lstm shape: ", lstm_out.shape)
        attn_weight_matrix = self.W_s1(lstm_out)
        attn_weight_matrix = torch.tanh(attn_weight_matrix)
        attn_weight_matrix = self.W_s2(attn_weight_matrix)
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)
        #print("attn matrix shape: ", attn_weight_matrix.shape)
        return attn_weight_matrix
        
    def forward(self, x, stat):
        batch_size, seq_len, _ = x.size()
        x = self.drop(x)
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        attn_weight_matrix = self.attention(lstm_out)
        hidden_matrix = torch.bmm(attn_weight_matrix, lstm_out)
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
        fc_out = self.label(fc_out)
        fc_out = fc_out + stat
        return fc_out

#initialize LSTM and RF
dev = torch.device("cuda:0")
mv_net = lstm(n_features, n_timesteps)
mv_net.to(dev)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mv_net.parameters(), lr=.01, weight_decay=.0001)
model = RandomForestClassifier(criterion='entropy', n_estimators=500, bootstrap=True, max_features='sqrt', class_weight = "balanced_subsample", n_jobs=-1, max_depth = 15)


mv_net.train()
for t in range(epoch):
    #train RF on first half
    model.fit(trainSD1, trainlabel1)
    #use RF to guess second half
    for b in range(0,len(trainxSD2),batch_size):
        time_data = torch.tensor(trainxSD2[b:b+batch_size,:,:], dtype=torch.float32).to(dev)
        labels = torch.tensor(trainlabel2[b:b+batch_size], dtype=torch.long).to(dev)    
        static_data = trainSD2[b:b+batch_size, :]
        rf_starts = torch.tensor(model.predict_proba(static_data), dtype=torch.float32).to(dev)
        mv_net.init_hidden(time_data.size(0))
        output = mv_net(time_data, rf_starts)         
        loss = criterion(output, labels) #.view(-1)
        loss.backward()
        optimizer.step()        
        optimizer.zero_grad()
        
        if (b % batch_size * 10) == 0:
            torch.cuda.empty_cache()
            
    #train RF on second half
    model.fit(trainSD2, trainlabel2)
    for b in range(0,len(trainxSD1),batch_size):
        time_data = torch.tensor(trainxSD1[b:b+batch_size,:,:], dtype=torch.float32).to(dev)
        labels = torch.tensor(trainlabel1[b:b+batch_size], dtype=torch.long).to(dev)    
        static_data = trainSD1[b:b+batch_size, :]
        rf_starts = torch.tensor(model.predict_proba(static_data), dtype=torch.float32).to(dev)
        mv_net.init_hidden(time_data.size(0))
        output = mv_net(time_data, rf_starts)         
        loss = criterion(output, labels) #.view(-1)
        loss.backward()
        optimizer.step()        
        optimizer.zero_grad()
        
        if (b % batch_size * 10) == 0:
            torch.cuda.empty_cache()
    print('step : ' , t , 'loss : ' , loss.item())

#Eval
guess = torch.tensor([]).to(dev)
for b in range(0, len(testxSD), batch_size):
    time_data = torch.tensor(testxSD[b:b+batch_size,:,:], dtype=torch.float32).to(dev)
    static_data = testSD[b:b+batch_size, :]
    rf_starts = torch.tensor(model.predict_proba(static_data), dtype=torch.float32).to(dev)
    mv_net.init_hidden(time_data.size(0))
    output = mv_net(time_data, rf_starts)
    guess = torch.cat((guess, output.detach()))
guess = guess[:,[1]].cpu()
guess = np.array(guess).squeeze()
auc = roc_auc_score(testlabel, guess)
print(auc)
#compare how it does against just RF
#auc = roc_auc_score(testlabel, model.predict_proba(testSD)[:,1])
