import json
import os
import pickle
import copy
import math
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils import convert_float_to_fp8,convert_to_classes
from src.utils import TaskType

"""
The dataset return format: a dictionary
{
    'task_type': ...        # TaskType
    'image_features': ...   # list[ndarray], optional
    'audio_features': ...   # list[ndarray], optional
    'text': ...            # str, optional
    'labels': ...           # str, optional
    ‘speaker’:
    'context':
    other task specific items...
}
"""

def get_sub_name(file_name, rank, world_size):
    dir_name, base_name = os.path.split(file_name) # './xxx', 'yyy.zzz'
    base_name, suf_name = base_name.split('.') # 'yyy', 'zzz'
    sub_file_name = os.path.join(dir_name, base_name + '_div_{}_sub_{}'.format(world_size, rank) + '.' + suf_name)
    return sub_file_name

class PretrainDataset(Dataset):
    def __init__(self, data_dir,use_text=True,use_image=True,use_audio=True):
        self._data_dir = data_dir 
            
        self._dataset = pickle.load(open(data_dir,'rb'))
        self._use_text = use_text
        self._use_image = use_image
        self._use_audio = use_audio
    def __getitem__(self, index):
        output={}
        if self._use_image:
            if self._dataset[index]['image_features']!='noimg':
                output['image_features']=self._dataset[index]['image_features'] 
        if self._use_audio:
            if self._dataset[index]['audio_features']!='noaudio':
                output['audio_features']=self._dataset[index]['audio_features'] 
        output['text']=self._dataset[index]['text'] if self._use_text else 'notext'
        output['labels']=self._dataset[index]['s_label']
        output['task_type']=self._dataset[index]['task_type']
        output['context']=self._dataset[index]['context']
        output['speaker']=self._dataset[index]['speaker']
        output['data_id']=self._dataset[index]['data_id']     
        output['index']=self._dataset[index]['index'] if 'index' in self._dataset[index] else -1
        output['concat_text']=self._dataset[index]['concat_text'] if 'concat_text' in self._dataset[index] else 'text'
        return output
    def __len__(self):
        return len(self._dataset)
class PretrainDataset2(Dataset):
    def __init__(self, data_dir,data_id,use_text=True,use_image=True,use_audio=True):
        self._data_dir = data_dir 
        self._data_id = data_id     
        self._dataset = pickle.load(open(data_dir,'rb'))
        self._use_text = use_text
        self._use_image = use_image
        self._use_audio = use_audio
        
    def __getitem__(self, index):
        output={}
        if self._use_image:
            output['image_features']=self._dataset[index]['image_features'] 
        if self._use_audio:
            output['audio_features']=self._dataset[index]['audio_features'] 
        output['text']=self._dataset[index]['text'] if self._use_text else 'notext'
        if self._data_id in ['mosi','mosei']:    
            output['labels']=convert_to_classes(self._dataset[index]['label'])
        else:
            output['labels']=self._dataset[index]['label']
        output['task_type']=self._dataset[index]['task_type']
        output['context']=self._dataset[index]['context']
        output['speaker']=self._dataset[index]['speaker']
        output['data_id']=self._data_id      
        output['index']=self._dataset[index]['index'] if 'index' in self._dataset[index] else -1
        
        return output
    def __len__(self):
        return len(self._dataset)
class COMMENTALLDataset(Dataset):
    def __init__(self, data_dir, rank, data_id='amazon', world_size=32):
        data_dir = get_sub_name(data_dir, rank, world_size)
        self._data_dir = data_dir
        
        self._dataset = pickle.load(open(data_dir,'rb'))
        self._data_id=data_id
    def __getitem__(self, index):
        output={}
        output['text']=self._dataset[index]['text']
        output['labels']=self._dataset[index]['label']
        output['task_type']=self._dataset[index]['task_type']
        output['speaker']=self._dataset[index]['speaker']
        output['context']=self._dataset[index]['context']
        output['data_id']=self._data_id

        return output
    def __len__(self):
        return len(self._dataset)
class ABSADataset(Dataset):
    def __init__(self, data_dir,data_id):
        self._data_dir = data_dir      
        self._dataset = pickle.load(open(data_dir,'rb'))
        self._data_id=data_id
    def __getitem__(self, index):
        output={}
        output['text']=self._dataset[index]['text']
        
        output['task_type']=self._dataset[index]['task_type']
        output['speaker']=self._dataset[index]['speaker']
        output['context']=self._dataset[index]['context']
        output['data_id']=self._data_id
        term=self._dataset[index]['term'].split('<sep>')[:-1]
        label=self._dataset[index]['label'].split('<sep>')[:-1]
        labels=""
        for j in range(len(term)):
            labels +=term[j]+';'+label[j]+'<sep>'
        output['labels']=labels        
        return output
    def __len__(self):
        return len(self._dataset)
class COMMENTDataset(Dataset):
    def __init__(self, data_dir,data_id):
        self._data_dir = data_dir      
        self._dataset = pickle.load(open(data_dir,'rb'))
        self._data_id=data_id
    def __getitem__(self, index):
        output={}
        output['text']=self._dataset[index]['text']
        output['labels']=self._dataset[index]['label']
        output['task_type']=self._dataset[index]['task_type']
        output['speaker']=self._dataset[index]['speaker']
        output['context']=self._dataset[index]['context']
        output['data_id']=self._data_id
                        
        return output
    def __len__(self):
        return len(self._dataset)
class ERCDataset(Dataset):
    def __init__(self, data_dir,data_id,use_text=True,use_image=True,use_audio=True):
        self._data_dir = data_dir      
        self._dataset = pickle.load(open(data_dir,'rb'))
        self._data_id=data_id
        self._use_text = use_text
        self._use_image = use_image
        self._use_audio = use_audio
    def __getitem__(self, index):
        output={}
        if self._use_image:
            output['image_features']=self._dataset[index]['image_features'] 
        if self._use_audio:
            output['audio_features']=self._dataset[index]['audio_features'] 
        output['text']=self._dataset[index]['text'] if self._use_text else 'notext'
        output['labels']=self._dataset[index]['label']
        output['task_type']=self._dataset[index]['task_type']
        output['speaker']=self._dataset[index]['speaker']
        output['context']=self._dataset[index]['context']
        output['index']=self._dataset[index]['index'] if 'index' in self._dataset[index] else None
        output['data_id']=self._data_id
                        
        return output
    def __len__(self):
        return len(self._dataset)
class SARCDataset(Dataset):
    def __init__(self, data_dir,data_id,use_text=True,use_image=False,use_audio=False):
        self._data_dir = data_dir      
        self._dataset = pickle.load(open(data_dir,'rb'))
        self._data_id=data_id
        self._use_text = use_text
        self._use_image = use_image
        self._use_audio = use_audio
    def __getitem__(self, index):
        output={}
        if self._use_image:
            output['image_features']=self._dataset[index]['image_features'] 
        if self._use_audio:
            output['audio_features']=self._dataset[index]['audio_features'] 
        output['text']=self._dataset[index]['text'] if self._use_text else 'notext'
        output['labels']=self._dataset[index]['label']
        output['task_type']=self._dataset[index]['task_type']
        output['speaker']=self._dataset[index]['speaker']
        output['context']=self._dataset[index]['context']
        output['data_id']=self._data_id
        
                        
        return output
    def __len__(self):
        return len(self._dataset)


class EMOJIDataset(Dataset):
    def __init__(self, data_dir,  data_id):
        
        self._data_dir = data_dir      
        self._split = split
        self._dataset = pickle.load(open(data_dir,'rb'))
        self._data_id=data_id
    def __getitem__(self, index):
        output={}
        output['text']=self._dataset[index]['text']
        output['labels']=self._dataset[index]['label']
        output['task_type']=self._dataset[index]['task_type']
        output['speaker']=self._dataset[index]['speaker']
        output['context']=self._dataset[index]['context']
        output['data_id']=self._data_id
                        
        return output

    def __len__(self):
        return len(self._dataset)

class MSADataset(Dataset):
    def __init__(self, data_dir,data_id, use_text=True,use_image=True,use_audio=True):
        self._use_text = use_text
        self._use_image = use_image
        self._use_audio = use_audio
        self._data_dir = data_dir      
        self._dataset = pickle.load(open(data_dir,'rb'))
        self._data_id=data_id
    def __getitem__(self, index):
        output={}
        if self._use_image:
            output['image_features']=self._dataset[index]['image_features'] 
        if self._use_audio:
            output['audio_features']=self._dataset[index]['audio_features'] 
          
        output['text']=self._dataset[index]['text'] if self._use_text else 'notext'
        output['labels']=convert_float_to_fp8(float(self._dataset[index]['label']))
        output['task_type']=self._dataset[index]['task_type']
        output['speaker']=self._dataset[index]['speaker']
        output['context']=self._dataset[index]['context']
        output['data_id']=self._data_id
                        
        return output

    def __len__(self):
        return len(self._dataset)




