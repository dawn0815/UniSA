import warnings

import numpy as np
import torch
from src.utils import TaskType
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Collator:
    """
    The collator for all types of dataset.
    Remember to add the corresponding collation code after adding a new type of task.
    """

    def __init__(
            self,
            tokenizer,
            has_label=True,
            mlm_enabled=False,
            mlm_probability=0.0,
            lm_max_len=360,
            context_max_len=480, 
            max_img_num=32,
            max_aud_num=157
    ):
        """
        :param tokenizer: ConditionTokenizer
        """
        self._tokenizer = tokenizer
        self._has_label = has_label
        self._lm_max_len = lm_max_len
        self._context_max_len=context_max_len
        self._max_img_num = max_img_num
        self._max_aud_num = max_aud_num

        if mlm_enabled and not has_label:
            raise ValueError('mlm_enabled can not be true while has_label is false. MLM need labels.')

    def context_split(self, c_num, c_index, c_list):
        if c_num == 1:
            return "nocontext"
        else:
            a = ""
            start = max(0, c_index - 5)  # start of context
            end = min(c_num, c_index + 6)  # end of context
            for i in range(start, end):
                if i==c_index:
                    a+='<sep>'
                else:
                    a += c_list[i]
        return a
    def _clip_text(self, text,length):
        tokenized = self._tokenizer.get_base_tokenizer()(text, add_special_tokens=False)
        return self._tokenizer.get_base_tokenizer().decode(tokenized['input_ids'][:length])
    
    def __call__(self, batch): 
        batch = [entry for entry in batch if entry is not None]
        #print(len(batch))

        image_features = [torch.from_numpy(x['image_features'][:self._max_img_num]) if 'image_features' in x else torch.empty(0) for x in batch]
        img_num = [len(x) for x in image_features]
        
        audio_features = [torch.from_numpy(x['audio_features'][:self._max_aud_num]) if 'audio_features' in x else torch.empty(0) for x in batch]
        audio_num = [len(x) for x in audio_features]
        task_type = [x['task_type'] for x in batch]
        data_id = [x['data_id'] for x in batch]
        speaker_token = [x['speaker'] for x in batch]
        text_token = [self._clip_text(x['text'],self._lm_max_len) for x in batch]
        context_index = [x['index'] if 'index' in x else -1 for x in batch]
        context_list=[x['context'] for x in batch]
        context_num = [len(x['context']) if x['context']!='nocontext' else 0 for x in batch]
        context=[]
        for i, value in enumerate(context_index):
            if value!=-1:
                context.append(self.context_split(context_num[i],value,context_list[i]))
            else:
                context.append("nocontext")
        
        labels = [x['labels'] for x in batch]
        
        encoded_conditions = self._tokenizer.encode_condition(
            data_id=data_id,
            speaker_token=speaker_token,
            img_num=img_num,
            audio_num=audio_num,
            text_token=text_token,
            context_num=context_num,
            task_type=task_type,
            context=context
        )
        input_ids = encoded_conditions['input_ids']
        
        
        output = {
            'input_ids': input_ids,
            'image_features': image_features,
            'audio_features': audio_features,
            'raw_labels':labels,
            'attention_mask': encoded_conditions['attention_mask'],
            'task_type': task_type,
            'data_id':data_id,
            'context_num':context_num
        }
        if self._has_label:
            encoded_labels = self._tokenizer.encode_label(task_type=task_type,raw_labels=labels,data_id=data_id)
            labels = encoded_labels['labels']
            decoder_input_ids = encoded_labels['decoder_input_ids']
            

            labels[(labels == self._tokenizer.pad_token_id) |
                   (labels == self._tokenizer.begin_img_id) |
                   (labels == self._tokenizer.end_img_id) |
                   (labels == self._tokenizer.begin_audio_id) |
                   (labels == self._tokenizer.end_audio_id) |
                   (labels == self._tokenizer.audio_feat_id) |
                   (labels == self._tokenizer.begin_context_id) |
                   (labels == self._tokenizer.end_context_id) |
                   (labels == self._tokenizer.speaker_a_id) | (labels == self._tokenizer.speaker_b_id) | (labels == self._tokenizer.speaker_c_id) |
                   (labels == self._tokenizer.speaker_d_id) | (labels == self._tokenizer.speaker_e_id) | (labels == self._tokenizer.speaker_f_id) |
                   (labels == self._tokenizer.img_feat_id)] = -100
        
            #print(labels.shape) 
            output['labels'] = labels  
            output['decoder_input_ids'] = decoder_input_ids
            output['decoder_attention_mask'] = encoded_labels['decoder_attention_mask']   
        
        return output


class PretrainCollator:
    def __init__(
            self,
            tokenizer,
            has_label=True,
            mlm_enabled=True,
            is_pretrain_stage1=True,
            mlm_probability=0.3,
            lm_max_len=360,
            max_img_num=32,
            max_aud_num=157
    ):
        """
        :param tokenizer: ConditionTokenizer
        """
        self._tokenizer = tokenizer
        self._has_label = has_label
        self._mlm_enabled = mlm_enabled
        self._lm_max_len = lm_max_len
        self._max_img_num = max_img_num
        self._max_aud_num = max_aud_num
        self._mlm_probability=mlm_probability
        self._is_pretrain_stage1=is_pretrain_stage1
        if mlm_enabled and not has_label:
            raise ValueError('mlm_enabled can not be true while has_label is false. MLM need labels.')

    def context_split(self,c_num,c_index,c_list):
        a=""
        if c_num==1:
            return "nocontext"
        else:
            if c_index>1 and c_num-c_index>2:
                a+=c_list[c_index-2]+c_list[c_index-1]+c_list[c_index+1]+c_list[c_index+2]
            elif c_index>1 and c_num-c_index==2:
                a+=c_list[c_index-2]+c_list[c_index-1]+c_list[c_index+1]
            elif c_index>1 and c_num-c_index<2:
                a+=c_list[c_index-2]+c_list[c_index-1]
            elif c_index==1 and c_num-c_index>2:
                a+=c_list[c_index-1]+c_list[c_index+1]+c_list[c_index+2]
            elif c_index==1 and c_num-c_index==2:
                a+=c_list[c_index-1]+c_list[c_index+1]
            elif c_index==1 and c_num-c_index<2:
                a+=c_list[c_index-1]
            elif c_index==0 and c_num-c_index>2:
                a+=c_list[c_index+1]+c_list[c_index+2]
            elif c_index==0 and c_num-c_index==2:
                a+=c_list[c_index+1]
        return a
    def _clip_text(self, text,length):
        tokenized = self._tokenizer.get_base_tokenizer()(text, add_special_tokens=False)
        return self._tokenizer.get_base_tokenizer().decode(tokenized['input_ids'][:length])
    
    def __call__(self, batch): 
        batch = [entry for entry in batch if entry is not None]
        #print(len(batch))

        image_features = [torch.from_numpy(x['image_features'][:self._max_img_num]) if 'image_features' in x else torch.empty(0) for x in batch]
        img_num = [len(x) for x in image_features]
        
        audio_features = [torch.from_numpy(x['audio_features'][:self._max_aud_num]) if 'audio_features' in x else torch.empty(0) for x in batch]
        audio_num = [len(x) for x in audio_features]
        task_type = [x['task_type'] for x in batch]
        data_id = [x['data_id'] for x in batch]
        speaker_token = [x['speaker'] for x in batch]
        context_index = [x['index'] if 'index' in x else -1 for x in batch]
        mlm = [self._clip_text(x['text'],self._lm_max_len) for x in batch]
        context_list=[x['context'] for x in batch]
        context_num = [len(x['context']) if x['context']!='nocontext' else 0 for x in batch]
        concat_text = [x['concat_text'] if 'concat_text' in x else 'notext' for x in batch]

        context=[]
        for i, value in enumerate(context_index):
            if value!=-1:
                context.append(self.context_split(context_num[i],value,context_list[i]))
            else:
                context.append("nocontext")
        if self._is_pretrain_stage1:
            labels = [x['labels'] for x in batch]
        else:
            labels = [x['pseudo_labels'] for x in batch]
        
        encoded_conditions = self._tokenizer.pre_encode_condition(
            data_id=data_id,
            speaker_token=speaker_token,
            img_num=img_num,
            audio_num=audio_num,
            mlm=mlm,
            context_num=context_num,
            context=context,
            task_type=task_type,
            concat_text=concat_text,
            is_pretrain_stage1=self._is_pretrain_stage1
        )
        
        input_ids = encoded_conditions['input_ids']
        
        
        if self._mlm_enabled:
            input_ids = self._mask_tokens(inputs=input_ids, input_mask=encoded_conditions['mlm_mask'])
            input_ids = self._mask_audio_and_image(inputs=input_ids, input_mask=encoded_conditions['mam_mask'])
        
        output = {
            'input_ids': input_ids,
            'image_features': image_features,
            'audio_features': audio_features,
            'raw_labels':labels,
            'attention_mask': encoded_conditions['attention_mask'],
            # 'context_token':context_token,
            'context_num':context_num,
            'task_type': task_type,
            'data_id':data_id
        }
        if self._has_label:
            if self._is_pretrain_stage1:
                encoded_labels = self._tokenizer.pre_encode_label(mlm=mlm,context=context,raw_labels=labels)
            else:
                encoded_labels = self._tokenizer.pre_encode_label2(mlm=mlm,context=context,raw_labels=labels)
            labels = encoded_labels['labels']
            labels[(labels == self._tokenizer.sep_token_id) | 
                    (labels == self._tokenizer.begin_audio_id) |
                   (labels == self._tokenizer.end_audio_id) |
                   (labels == self._tokenizer.begin_text_id) |
                   (labels == self._tokenizer.end_text_id) |
                   (labels == self._tokenizer.begin_context_id) |
                   (labels == self._tokenizer.end_context_id) |
                   (labels == self._tokenizer.pad_token_id) ] = -100

            #print(labels.shape) 
            output['labels'] = labels  
            output['decoder_input_ids'] = encoded_labels['decoder_input_ids']
            output['decoder_attention_mask'] = encoded_labels['decoder_attention_mask']   
        # if self._has_label:
        #     output['raw_labels'] = [x['labels'] for x in batch]
        return output
    def _mask_tokens(self, inputs, input_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

        :param inputs: torch.LongTensor, batch data
        :param input_mask: torch.Tensor, mask for the batch, False for the position with 0% probability to be masked
        """

        labels = inputs.clone()
        tokenizer = self._tokenizer.get_base_tokenizer()

        # We sample a few tokens in each sequence for masked-LM training
        probability_matrix = torch.full(labels.shape, self._mlm_probability, dtype=torch.float)
        special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                               for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if tokenizer.pad_token is not None:
            padding_mask = labels.eq(tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced & input_mask] = tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random & input_mask] = random_words[indices_random & input_mask]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs
    def _mask_audio_and_image(self, inputs, input_mask):
        """
        Prepare masked tokens inputs/labels for masked audio/image modeling: 80% MASK, 20% original.

        :param inputs: torch.LongTensor, batch data
        :param input_mask: torch.Tensor, mask for the batch, False for the position with 0% probability to be masked
        """

        labels = inputs.clone()
        tokenizer = self._tokenizer.get_base_tokenizer()

        # We sample a few tokens in each sequence for masked-LM training
        probability_matrix = torch.full(labels.shape, self._mlm_probability, dtype=torch.float)
        special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                               for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if tokenizer.pad_token is not None:
            padding_mask = labels.eq(tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced & input_mask] = tokenizer.mask_token_id

        return inputs