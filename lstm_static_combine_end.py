import torch
import torch.nn as nn
import numpy as np
from numpy import array
from numpy import hstack
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.nn import functional as F
import random 
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Variables
random.seed(10)
n_features = 24
n_timesteps = 539
epoch = 10
batch_size = 32
n_static = 65

class lstm(nn.Module):
    def __init__(self,n_features,seq_length):
        super(lstm, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 10 # number of hidden states
        self.n_layers = 2# number of LSTM layers (stacked)   
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True, dropout = .3)
        self.W_s1 = nn.Linear(self.n_hidden, 40)
        self.W_s2 = nn.Linear(40, 1)
        self.fc_layer = nn.Linear(self.n_hidden, 20)
        self.label = nn.Linear(20, 2)
        self.drop = nn.Dropout(p=.15)
        self.sig = nn.Sigmoid()
        self.static1 = nn.Linear(n_static, 30)
        self.static2 = nn.Linear(30, 2)
        
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
        
    def forward(self, x, static):
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
		# logits.size() = (batch_size, output_size)
        #print("logits shape: ", logits.shape)
        #print("logits: ", logits)
        #print("static: ", self.static2(self.static1(static)))
        logits = (self.static2(self.static1(static)) + logits) / 2
        return logits

#load
#traindata = np.load('traindata.npy')
#trainSD = np.load('trainSD.npy')
#trainlabel = np.load('trainlabel.npy')
#testdata = np.load('testdata.npy')
#testSD = np.load('testSD.npy')
#testlabel = np.load('testlabel.npy')
#shuffle
#indicies = np.arange(len(traindata))
#random.shuffle(indicies)
#traindata = traindata[indicies]
#trainSD = trainSD[indicies]
#trainlabel = trainlabel[indicies]

dev = torch.device("cuda:0")
mv_net = lstm(n_features,n_timesteps)
mv_net.to(dev)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-4, weight_decay=.0001)


mv_net.train()
for t in range(epoch):
    for b in range(0,len(traindata),batch_size):
        inpt = traindata[b:b+batch_size,:,:]
        target = trainlabel[b:b+batch_size]           
        x_batch = torch.tensor(inpt,dtype=torch.float32).to(dev)   
        y_batch = torch.tensor(target,dtype=torch.long).to(dev)   
        SD_batch = torch.tensor(trainSD[b:b+batch_size,:], dtype=torch.float32).to(dev)   
        mv_net.init_hidden(x_batch.size(0))
        output = mv_net(x_batch, SD_batch)         
        loss = criterion(output, y_batch) #.view(-1)
        loss.backward()
        optimizer.step()        
        optimizer.zero_grad()
        
        if (b % batch_size * 10) == 0:
            torch.cuda.empty_cache()
    print('step : ' , t , 'loss : ' , loss.item())

#Eval
guess = torch.tensor([]).to(dev)
for b in range(0, len(testdata), batch_size):
    inpt = testdata[b:b+batch_size,:,:]
    x_batch = torch.tensor(inpt,dtype=torch.float32).to(dev)   
    SD_batch = torch.tensor(testSD[b:b+batch_size,:],dtype=torch.float32).to(dev)   
    mv_net.init_hidden(x_batch.size(0))
    output = mv_net(x_batch, SD_batch)
    #print(output.shape)
    guess = torch.cat((guess, output.detach()))
guess = guess[:,[1]].cpu()
guess = np.array(guess).squeeze()
model = RandomForestClassifier(n_estimators=500, bootstrap=True, max_features='sqrt', class_weight = "balanced", n_jobs=-1, max_depth = 3)
model.fit(trainSD, trainlabel)
predict = model.predict_proba(testSD)[:,1]
auc = roc_auc_score(testlabel, (guess+predict)/2)
print(auc)
#tn, fp, fn, tp = confusion_matrix(testlabel, guess).ravel()
