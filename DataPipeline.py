#!/usr/bin/env python
# coding: utf-8

# In[1]:


import wfdb
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset


# In[45]:


class Preprocess:

    def __init__(self, path_to_dataset, eval=False):
        patients = []
        data = []
        labels = []

        #val_data? #test data?
        #val_data = np.array([])
        #test_data = np.array([])
        
        for dir in os.listdir(path1):
            p = os.path.join(path1,dir)
            if os.path.isdir(p):
                patients.append(p)

        for p in patients:
            for file in os.listdir(p):
                file_pathname = os.path.join(p,file)
                dat_pathname = file_pathname.replace('hea','dat')
                break
            signal = wfdb.io.rdsamp(dat_pathname.replace('.dat',''))
            #data = np.concatenate((data,np.array(signal[0])), axis = 0)
            data.append(np.array(signal[0]))
            
            if(not(eval)):
                labels.append(self.find_label(signal[1])) 
            mapped_labels = map_labels(labels)
            break
        self.data = np.array(data)
        if(not(eval)):
            self.labels = mapped_labels

    def get_data(self):
        return self.data, self.labels
      	 

    def find_label(self, md):
        metadata_keys = list(md.keys())
        metadata_keys_lower = [k.lower() for k in metadata_keys]
        if 'comments' in metadata_keys_lower:
            comment_index = metadata_keys.index('comments')
            metadata_comments = md[metadata_keys[comment_index]]
            metadata_comments_lower = [k.lower() for k in metadata_comments]
            label = ''
            for i, stri in enumerate(metadata_comments_lower):
                if 'reason for admission:' in stri:
                    label = stri.split(':')[1].strip()
            if label =='':
                print('reason for admission not in metadata')
                exit(1)
            return label

        else:
            print("comments not in metadata")
            exit(1)

    


# In[3]:


class EcgDataset(Dataset):
    
    def __init__(self, data,labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]



   


# In[55]:


def labels_nametonum():

    raw_labels = ['Myocardial infarction',
        'Cardiomyopathy/Heart failure',
        'Bundle branch block',
        'Dysrhythmia',	
        'Myocardial hypertrophy',
        'Valvular heart disease',	
        'Myocarditis',	
        'Miscellaneous',
        'Healthy controls']

    raw_labels_lower = [l.lower() for l in raw_labels]

    raw_labels = dict(zip(raw_labels_lower, list(range(len(raw_labels_lower)))))

    return raw_labels

def map_labels(pre_labels):

    labels = []
    label_mapping = labels_nametonum()
    print(label_mapping)
    for name in pre_labels:
        if name in label_mapping:
            labels.append(label_mapping[name])
        else:
            print("label %s not in mapping", name)
            exit(1)
        
    return np.array(labels)
            
    


# In[4]:


if __name__ == "main":

    path1 = "/home/aushim/Desktop/ptb-diagnostic-ecg-database-1.0.0"
    batch_size = 128
    procdata = Preprocess(path1)
    #np.save(procdata, procdata.npy )

    #train data
    dataset_train = EcgDataset(procdata.data, procdata.labels)
    train_args = dict(shuffle = True, batch_size = batch_size, num_workers=0)
    train_dataloader = DataLoader(dataset_train)

    print("data preprocessed")

