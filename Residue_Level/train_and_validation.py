
# !pip install imbalanced-learn
from imblearn.over_sampling import ADASYN
import torch
from torch import nn
import torch.nn.functional as F
#load data
import pickle
import gzip
import numpy as np

import torch
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn import Embedding, LSTM,Conv1d,Conv2d,MaxPool2d
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.optim as optim 

import torch.nn.functional as F


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

import sklearn
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
def report(y_t,y_p):
    acc=accuracy_score(y_t,y_p)
    pre=precision_score(y_t,y_p)
    rec=recall_score(y_t,y_p)
    spec=recall_score(y_t, y_p, pos_label=0)
    mt=matthews_corrcoef(y_t, y_p)
    print("Acc: "+str(acc))
    print("Precision: "+str(pre))
    print("Recll: "+str(rec))
    print("Specificity: "+str(spec))
    print("MCC score: "+str(mt))
    res=[acc,pre,rec,spec,mt]
    return res

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_chanels, **kwargs)
        self.bn = nn.BatchNorm1d(out_chanels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
class InceptionBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_1x1,
        red_1x1,
        out_3x3,
        out_5x5,
        out_pool,
    ):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1) # 
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_1x1, kernel_size=1, padding=0),
            ConvBlock(red_1x1, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_1x1, kernel_size=1),
            ConvBlock(red_1x1, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )
    
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        #print((self.branch1(x)).shape)
        return (torch.cat([branch(x) for branch in branches], 1))

class Baseline_1(nn.Module):
    def __init__(
        self, 

       
    ):
        super(Baseline_1, self).__init__()
        self.inception_1=InceptionBlock(1024,128,128,128,128,128)

        self.inception_2=InceptionBlock(512,64,64,64,64,64)

        self.inception_3=InceptionBlock(256,32,32,32,32,32)
        self.linear_1=nn.Linear(128,32)
        self.linear_2=nn.Linear(32,2)
        self.dropout = nn.Dropout(p=0.2)






    
    def forward(self, x):
        x=torch.transpose(x,2,1)
        x=self.inception_1(x)
        x=self.dropout(x)
        x=self.inception_2(x)
        x=self.dropout(x)
        x=self.inception_3(x)
        x=self.dropout(x)
        x=torch.mean(x,dim=2)
        #x = x.reshape(x.shape[0], -1)
        x=self.linear_1(x)
        x=F.relu(self.linear_2(x))
        
        return x


class MinimalDataset(Dataset):
    def __init__(self, data,y) -> None:
        super().__init__()
        self.data = data
        self.y=y

    def __getitem__(self, index):
        sample={'data':self.data[index],'y':self.y[index]}
        #return [self.data[index],self.y[index]]
        return sample

    def __len__(self):
        return len(self.data)

train_x_1=pickle.load(gzip.open('../dataset/LLM_features_pdna224.gz', "rb"))
train_y_1=pickle.load(gzip.open('../dataset/pdna224_label.pkl.gz',"rb"))

# train_x_2=pickle.load(gzip.open('D:/Toki_1805030/residue_level_task/dataset/pdna316_protbert.pkl.gz', "rb"))
# train_y_2=pickle.load(gzip.open('D:/Toki_1805030/residue_level_task/dataset/pdna316_label.pkl.gz',"rb"))

# train_x_3=pickle.load(gzip.open('D:/Toki_1805030/residue_level_task/dataset/pdna543_protbert.pkl.gz', "rb"))
# train_y_3=pickle.load(gzip.open('D:/Toki_1805030/residue_level_task/dataset/pdna543_label.pkl.gz',"rb"))

# test_x=pickle.load(gzip.open('D:/Toki_1805030/residue_level_task/dataset/pdnatest_protbert.pkl.gz', "rb"))
# test_y=pickle.load(gzip.open('D:/Toki_1805030/residue_level_task/dataset/pdnatest_label.pkl.gz',"rb"))

train_x_2=pickle.load(gzip.open('../dataset/LLM_features_pdna316.gz', "rb"))
train_y_2=pickle.load(gzip.open('../dataset/pdna316_label.pkl.gz',"rb"))

train_x_3=pickle.load(gzip.open('../dataset/LLM_features_pdna543.pkl.gz', "rb"))
train_y_3=pickle.load(gzip.open('../dataset/pdna543_label.pkl.gz',"rb"))

test_x=pickle.load(gzip.open('../dataset/LLM_features_pdna_test.gz', "rb"))
test_y=pickle.load(gzip.open('../dataset/pdnatest_label.pkl.gz',"rb"))

# Preprocessing data

train_y_temp=[]
for item in train_y_2:
    temp=[]
    for i in item:
        if(i!=","):
            temp.append(i)
    train_y_temp.append(temp[:-1])

train_y_2=train_y_temp

test_data_x=test_x["ProtBert_features"]

for i in range(len(test_data_x)):
  test_data_x[i]=(torch.from_numpy(test_data_x[i].reshape((test_data_x[i].shape[1],test_data_x[i].shape[2]))))
for i in range(len(test_y)):
    test_y[i]=[int(x) for x in test_y[i]]
    test_y[i]=np.array(test_y[i])

all_seq_test=[]
for item in test_data_x:
    for i in item:
        i=i.reshape(i.shape[0])
        all_seq_test.append(i)

all_label_test=[]
for item in test_y:
    for i in item:
        all_label_test.append(i)
        
train_data_x_1=train_x_1["ProtBert_features"]
train_data_x_2=train_x_2["ProtBert_features"]
train_data_x_3=train_x_3["ProtBert_features"]


for i in range(len(train_y_1)):
    train_y_1[i]=[int(x) for x in train_y_1[i]]
    train_y_1[i]=np.array(train_y_1[i])

for i in range(len(train_y_2)):
    train_y_2[i]=[int(x) for x in train_y_2[i]]
    train_y_2[i]=np.array(train_y_2[i])

for i in range(len(train_y_3)):
    train_y_3[i]=[int(x) for x in train_y_3[i]]
    train_y_3[i]=np.array(train_y_3[i])

for i in range(len(train_data_x_1)):
  train_data_x_1[i]=(torch.from_numpy(train_data_x_1[i].reshape((train_data_x_1[i].shape[1],train_data_x_1[i].shape[2]))))

for i in range(len(train_data_x_2)):
  train_data_x_2[i]=(torch.from_numpy(train_data_x_2[i].reshape((train_data_x_2[i].shape[1],train_data_x_2[i].shape[2]))))

for i in range(len(train_data_x_3)):
  train_data_x_3[i]=(torch.from_numpy(train_data_x_3[i].reshape((train_data_x_3[i].shape[1],train_data_x_3[i].shape[2]))))

all_seq_1=[]
for item in train_data_x_1:
    for i in item:
        i=i.reshape(i.shape[0])
        all_seq_1.append(i)

all_label_1=[]
for item in train_y_1:
    for i in item:
        all_label_1.append(i)


all_seq_2=[]
for item in train_data_x_2:
    for i in item:
        i=i.reshape(i.shape[0])
        all_seq_2.append(i)

all_label_2=[]
for item in train_y_2:
    for i in item:
        all_label_2.append(i)



all_seq_3=[]
for item in train_data_x_3:
    for i in item:
        i=i.reshape(i.shape[0])
        all_seq_3.append(i)

all_label_3=[]
for item in train_y_3:
    for i in item:
        all_label_3.append(i)



# 10 Fold Cross validation

from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
seq_1_index_train=[]
seq_1_index_valid=[]
for train_index, test_index in kf.split(all_seq_1):
    seq_1_index_train.append(train_index)
    seq_1_index_valid.append(test_index)
    
seq_2_index_train=[]
seq_2_index_valid=[]
for train_index, test_index in kf.split(all_seq_2):
    seq_2_index_train.append(train_index)
    seq_2_index_valid.append(test_index)

seq_3_index_train=[]
seq_3_index_valid=[]
for train_index, test_index in kf.split(all_seq_3):
    seq_3_index_train.append(train_index)
    seq_3_index_valid.append(test_index)




#TRAINING



dataset_1_res=[]
dataset_2_res=[]
dataset_3_res=[]
for fold in range(10):
    model=Baseline_1()
    loss_function=nn.CrossEntropyLoss()
    opt=optim.Adam(model.parameters(),lr=0.001)
    print(fold)
    train_x_1,valid_x_1,train_y_1,valid_y_1=[all_seq_1[i] for i in seq_1_index_train[fold]],[all_seq_1[i] for i in seq_1_index_valid[fold]],[all_label_1[i] for i in seq_1_index_train[fold]],[all_label_1[i] for i in seq_1_index_valid[fold]]
    train_x_2,valid_x_2,train_y_2,valid_y_2=[all_seq_2[i] for i in seq_2_index_train[fold]],[all_seq_2[i] for i in seq_2_index_valid[fold]],[all_label_2[i] for i in seq_2_index_train[fold]],[all_label_2[i] for i in seq_2_index_valid[fold]]
    train_x_3,valid_x_3,train_y_3,valid_y_3=[all_seq_3[i] for i in seq_3_index_train[fold]],[all_seq_3[i] for i in seq_3_index_valid[fold]],[all_label_3[i] for i in seq_3_index_train[fold]],[all_label_3[i] for i in seq_3_index_valid[fold]]
    
    
    
    
    for i in range(len(valid_x_1)):
      valid_x_1[i]=valid_x_1[i].reshape(1,valid_x_1[i].shape[0])

    for i in range(len(valid_x_2)):
      valid_x_2[i]=valid_x_2[i].reshape(1,valid_x_2[i].shape[0])

    for i in range(len(valid_x_3)):
      valid_x_3[i]=valid_x_3[i].reshape(1,valid_x_3[i].shape[0])
    
    
    all_seq_torch_1=torch.rand((len(train_x_1),1024))
    for i in range(len(train_x_1)):
        all_seq_torch_1[i]=train_x_1[i]
    all_seq_np_1=np.array(all_seq_torch_1)
    all_label_np_1=np.array(train_y_1)


    all_seq_torch_2=torch.rand((len(train_x_2),1024))
    for i in range(len(train_x_2)):
        all_seq_torch_2[i]=train_x_2[i]
    all_seq_np_2=np.array(all_seq_torch_2)
    all_label_np_2=np.array(train_y_2)



    all_seq_torch_3=torch.rand((len(train_x_3),1024))
    for i in range(len(train_x_3)):
        all_seq_torch_3[i]=train_x_3[i]
    all_seq_np_3=np.array(all_seq_torch_3)
    all_label_np_3=np.array(train_y_3)
    
    all_seq_np=np.concatenate((all_seq_np_1,all_seq_np_2,all_seq_np_3))
    all_label_np=np.concatenate((all_label_np_1,all_label_np_2,all_label_np_3))
    
    #from imblearn.over_sampling import ADASYN
    ada = ADASYN()
    X_resampled, y_resampled = ada.fit_resample(all_seq_np, all_label_np)
    #X_resampled=all_seq_np
    #y_resampled=all_label_np

    X=torch.from_numpy(X_resampled.reshape(X_resampled.shape[0],1,X_resampled.shape[1]))
    Y=y_resampled

    train_dataset = MinimalDataset(X,Y)
   

    data_loader_train = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=lambda x: x)







    best_mcc=0
    step=0
    model.train()
    for epoch in range(100): # 40,
      step+=1
      loss_total=0  
      for i, batch in enumerate(data_loader_train):
          #print(i)

          lens=[]
          data=[]
          labels=[]

          for items in batch:
            data.append(items['data'])
            labels.append(items['y'])
            lens.append(items['data'].shape[0])
          padded_seq_batch_train = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
          #packed_seq_batch_train = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch_train, lengths=lens, batch_first=True).to(device)
          scores=model(padded_seq_batch_train)
          labels=torch.Tensor(labels).to(torch.int64)

          loss=loss_function(scores,labels) # calculate loss
          loss_total+=loss



          #backward
          opt.zero_grad() #In PyTorch, for every mini-batch during the training phase, we need to explicitly set the gradients to zero
                              #before starting to do backpropragation (i.e., updation of Weights and biases) because PyTorch accumulates the gradients on 
                              #subsequent backward passes. This is convenient while training RNNs. So, the default action has been set to accumulate (i.e. sum)
                              # the gradients on every loss.backward() call
          loss.backward() 
          opt.step()
      print(loss_total) 
    
    #model.load_state_dict(torch.load('/content/drive/MyDrive/Bayezid sir/residue_level/model/best_model.pth'))

    model.eval()
    
    
    #224
    valid_dataset_1 = MinimalDataset(valid_x_1,valid_y_1)
    #valid_dataset = MinimalDataset(valid_x_,valid_y_)
    #test_dataset = MinimalDataset(test_data_x,test_y)

    data_loader_valid = DataLoader(valid_dataset_1, batch_size=512, shuffle=False, collate_fn=lambda x: x)
    num_correct=0
    num_samples=0
    all_true=[]
    all_pred=[]
    model.eval()
    with torch.no_grad():
         for i, batch in enumerate(data_loader_valid):     
            lens=[]
            data=[]
            labels=[]



            for items in batch:

              data.append(items['data'])
              labels.append(items['y'])
              lens.append(items['data'].shape[0])
            padded_seq_batch_train = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
            #packed_seq_batch_train = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch_train, lengths=lens, batch_first=True).to(device)
            scores=model(padded_seq_batch_train)
            labels=torch.Tensor(labels).to(torch.int64)

            _,predictions=scores.max(1) # get the max index of each ROW
            all_true.append(labels)
            all_pred.append(predictions)

            num_correct+=(predictions==labels).sum() # here in each iteration of the loop, a batch of size 64 will be passed here. In each batch ,
                                                  #we have 64 data. We sum all these 64 data predictions ( 1 or 0)
            num_samples+=predictions.size(0)

         acc=num_correct/num_samples

         print(acc) 


    all_true_concat=[]
    all_pred_concat=[]
    for item in all_true:
        for i in item:
            all_true_concat.append(i.item())
    for item in all_pred:
        for i in item:
            all_pred_concat.append(i.item())

    res=report(all_true_concat,all_pred_concat)
    dataset_1_res.append(all_pred_concat)
    
    
    
    valid_dataset_1 = MinimalDataset(valid_x_2,valid_y_2)
    #valid_dataset = MinimalDataset(valid_x_,valid_y_)
    #test_dataset = MinimalDataset(test_data_x,test_y)

    data_loader_valid = DataLoader(valid_dataset_1, batch_size=512, shuffle=False, collate_fn=lambda x: x)
    num_correct=0
    num_samples=0
    all_true=[]
    all_pred=[]
    model.eval()
    with torch.no_grad():
         for i, batch in enumerate(data_loader_valid):     
            lens=[]
            data=[]
            labels=[]



            for items in batch:

              data.append(items['data'])
              labels.append(items['y'])
              lens.append(items['data'].shape[0])
            padded_seq_batch_train = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
            #packed_seq_batch_train = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch_train, lengths=lens, batch_first=True).to(device)
            scores=model(padded_seq_batch_train)
            labels=torch.Tensor(labels).to(torch.int64)

            _,predictions=scores.max(1) # get the max index of each ROW
            all_true.append(labels)
            all_pred.append(predictions)

            num_correct+=(predictions==labels).sum() # here in each iteration of the loop, a batch of size 64 will be passed here. In each batch ,
                                                  #we have 64 data. We sum all these 64 data predictions ( 1 or 0)
            num_samples+=predictions.size(0)

         acc=num_correct/num_samples

         print(acc) 


    all_true_concat=[]
    all_pred_concat=[]
    for item in all_true:
        for i in item:
            all_true_concat.append(i.item())
    for item in all_pred:
        for i in item:
            all_pred_concat.append(i.item())

    res=report(all_true_concat,all_pred_concat)
    dataset_2_res.append(all_pred_concat)


    valid_dataset_1 = MinimalDataset(valid_x_3,valid_y_3)
    #valid_dataset = MinimalDataset(valid_x_,valid_y_)
    #test_dataset = MinimalDataset(test_data_x,test_y)

    data_loader_valid = DataLoader(valid_dataset_1, batch_size=512, shuffle=False, collate_fn=lambda x: x)
    num_correct=0
    num_samples=0
    all_true=[]
    all_pred=[]
    model.eval()
    with torch.no_grad():
         for i, batch in enumerate(data_loader_valid):     
            lens=[]
            data=[]
            labels=[]



            for items in batch:

              data.append(items['data'])
              labels.append(items['y'])
              lens.append(items['data'].shape[0])
            padded_seq_batch_train = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
            #packed_seq_batch_train = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch_train, lengths=lens, batch_first=True).to(device)
            scores=model(padded_seq_batch_train)
            labels=torch.Tensor(labels).to(torch.int64)

            _,predictions=scores.max(1) # get the max index of each ROW
            all_true.append(labels)
            all_pred.append(predictions)

            num_correct+=(predictions==labels).sum() # here in each iteration of the loop, a batch of size 64 will be passed here. In each batch ,
                                                  #we have 64 data. We sum all these 64 data predictions ( 1 or 0)
            num_samples+=predictions.size(0)

         acc=num_correct/num_samples

         print(acc) 


    all_true_concat=[]
    all_pred_concat=[]
    for item in all_true:
        for i in item:
            all_true_concat.append(i.item())
    for item in all_pred:
        for i in item:
            all_pred_concat.append(i.item())

    res=report(all_true_concat,all_pred_concat)
    dataset_3_res.append(all_pred_concat)
   

#checking the metrices for validation

np.array(dataset_1_res[0]).sum()/np.array(valid_y_1).sum()
report(valid_y_1,dataset_1_res[0])
#try others too






#Saving the model

# torch.save(model.state_dict(), '.../model/best_model.pth')










 
