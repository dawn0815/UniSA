import pickle
import csv
import os 
import pandas as pd
import numpy as np


data_path='/root/data/ruiren/datasets/amazon_data_pretrain.pkl'
data_new=[]
with open(data_path, 'rb') as fp:
    data = pickle.load(fp)
a_text=''
cont=0

for d in data:
    if cont<20000000:
        if str(d['label'])=='5.0':
            if d['text']!=a_text:
                a_text=d['text']
                data_new.append(d)
                cont+=1
    else:
        break
del data
print(len(data_new))
with open('/root/data/ruiren/datasets/amazon_all_pretrain_5.pkl', 'wb') as handle:
    pickle.dump(data_new, handle,protocol=pickle.HIGHEST_PROTOCOL)