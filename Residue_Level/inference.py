
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



   
#test
test_x=pickle.load(gzip.open('../dataset/LLM_features_pdna_test.gz', "rb"))
test_y=pickle.load(gzip.open('../dataset/pdnatest_label.pkl.gz',"rb"))
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
        
all_seq_torch_test=torch.rand((len(all_seq_test),1,1024))
for i in range(len(all_seq_test)):
    all_seq_torch_test[i]=all_seq_test[i].reshape(1,all_seq_test[i].shape[0])
all_label_np_test=np.array(all_label_test)
x_test=all_seq_torch_test
y_test=all_label_np_test

train_dataset = MinimalDataset(x_test,y_test)


data_loader_test = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=lambda x: x)


model=Baseline_1()
model.load_state_dict(torch.load('../dataset/residue_level_model.pth'))
model.eval()




def check_mcc(model):
    num_correct=0
    num_samples=0
    all_true=[]
    all_pred=[]
    model.eval()
    with torch.no_grad():
         for i, batch in enumerate(data_loader_test):     
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
    return res

check_mcc(model)
          


