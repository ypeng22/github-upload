import torch
import torch.nn as nn
import numpy as np
from numpy import array
from numpy import hstack
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.nn import functional as F
import random 

n_features = 24
n_timesteps = 538
epoch = 3
batch_size = 10
#X = np.load('X2.npy')
#y = np.load('y2.npy')
r = random.sample(range(len(X)), k = int(.8 * len(X)))
y = y * 10;

class lstm(nn.Module):
    def __init__(self,n_features,seq_length):
        super(lstm, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 60 # number of hidden states
        self.n_layers = 2# number of LSTM layers (stacked)   
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True, bidirectional = True, 
                                 dropout = .2)
        self.W_s1 = nn.Linear(self.n_hidden*2, 60)
        self.W_s2 = nn.Linear(60, 1)
        self.fc_layer = nn.Linear(self.n_hidden*2, 10)
        self.label = nn.Linear(10, 1)
        self.drop = nn.Dropout(p=.15)
        
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(2*self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(2*self.n_layers,batch_size,self.n_hidden)
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
        #print("x is: ", type(x), " shape is: ", x.shape)
        x = self.drop(torch.Tensor(x))
        #print("x is: ", type(x), " shape is: ", x.shape)
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        #lstm_out = lstm_out.permute(1, 0, 2)
        #output = lstm_out.permute(1, 0, 2)
		# output.size() = (batch_size, num_seq, 2*hidden_size)
		# h_n.size() = (1, batch_size, hidden_size)
		# c_n.size() = (1, batch_size, hidden_size)
        attn_weight_matrix = self.attention(lstm_out)
		# attn_weight_matrix.size() = (batch_size, r, num_seq)
		# output.size() = (batch_size, num_seq, 2*hidden_size)
        hidden_matrix = torch.bmm(attn_weight_matrix, lstm_out)
		# hidden_matrix.size() = (batch_size, r, 2*hidden_size)
		# Let's now concatenate the hidden_matrix and connect it to the fully connected layer.
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
        #print("fc_out shape: ", fc_out.shape)
        logits = self.label(fc_out)
		# logits.size() = (batch_size, output_size)
        #print("logits shape: ", logits.shape)
        return logits

mv_net = lstm(n_features,n_timesteps)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-2, weight_decay=.01)

traindata = X[r] #960
trainlabel = y[r]
testdata = X[np.delete(np.array(range(len(X))), r)] 
testlabel = y[np.delete(np.array(range(len(X))), r)]
mv_net.train()
for t in range(epoch):
    for b in range(0,len(traindata),batch_size):
        inpt = traindata[b:b+batch_size,:,:]
        target = trainlabel[b:b+batch_size]    
        
        x_batch = torch.tensor(inpt,dtype=torch.float32)    
        y_batch = torch.tensor(target,dtype=torch.float32)
    
        mv_net.init_hidden(x_batch.size(0))
        output = mv_net(x_batch)
        #if target == [1]:
            #loss = criterion(output.view(-1), y_batch)
       # else:
        loss = criterion(output.view(-1), y_batch) #.view(-1)
        loss.backward()
        optimizer.step()        
        optimizer.zero_grad() 
    print('step : ' , t , 'loss : ' , loss.item())


guess = torch.tensor([])
for b in range(0, len(testdata), batch_size):
    inpt = testdata[b:b+batch_size,:,:]
    x_batch = torch.tensor(inpt,dtype=torch.float32)
    mv_net.init_hidden(x_batch.size(0))
    output = mv_net(x_batch)
    print(output.shape)
    guess = torch.cat((guess, output.detach().view(-1)))
auc = roc_auc_score(testlabel, guess)
#tn, fp, fn, tp = confusion_matrix(testlabel, guess).ravel()
