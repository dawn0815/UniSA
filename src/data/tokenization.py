import torch
from transformers import BartTokenizer
from src.utils import TaskType


mosi="-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0"
meld="joy,neutral,fear,surprise,disgust,sad,angry"
iemocap="neutral,angry,frustrate,sad,happy,excited"
emoji="happy,sad,fear,angry,love,neutral,surprise"
emory="joy,sad,fear,peaceful,mad,neutral,powerful"
daily="happy,sad,fear,angry,disgust,neutral,surprise"
sst="-1.0,1.0"
emowoz="neutral,dissatisfied,apology,abusive,happy,sad,satisfied"
amazon="1.0,2.0,3.0,4.0,5.0"
sarc="sarcasm,normal"
absa14="positive,negative,neutral,conflict"
data_ans = {
    'mosi': mosi,
    'mosei': mosi,
    'iemocap':iemocap,
    'meld': meld,
    'emoji':emoji,
    'emory':emory,
    'daily':daily,
    'sst':sst,
    'imdb':sst,
    'amazon':amazon,
    'emowoz':emowoz,
    'sarc':sarc,
    'absa14':absa14,
    'absa16':absa14
}
def context_split(c_num,c_index,c_list):
    a=""
    if c_num==1:
        return " "
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

class ConditionTokenizer:
    """
    tokenizer for image features, event and task type
    this is NOT inherent from transformers Tokenizer
    """

    def __init__(
            self,
            pretrained_model_name='facebook/bart-base',
            
            #sep
            sep_token="[sep]",
            
            #task
            msa="<msa>",
            erc="<erc>",
            emoji="<emoji>",
            comment="<comment>",
            absa="<absa>",
            
            #data and ans
            begin_data="<data>",
            end_data="</data>",
            begin_ans="<ans>",
            end_ans="</ans>",
            #data id
            mosi="<mosi>",
            meld="<meld>",
            iemocap="<iemocap>",
            emowoz="<emowoz>",
            daily="<daily>",
            emory="<emory>",
            amazon="<amazon>",
            sarc="<sarc>",
            sst="<sst>",
            imdb="<imdb>",
            absa14="<absa14>",
            #dialogue specific
            speaker_a="<a>",
            speaker_b="<b>",
            speaker_c="<c>",
            speaker_d="<d>",
            speaker_e="<e>",
            speaker_f="<f>",
            speaker_g="<g>",
            speaker_h="<h>",
            speaker_i="<i>",
            begin_context="<context>",
            end_context="</context>",
            context="<context_token>",
            
            #modality
            begin_img="<img>",
            img_feat='<img_feat>',
            end_img="</img>",
            begin_audio="<audio>",
            audio_feat='<audio_feat>',
            end_audio="</audio>",
            begin_text="<text>",
            end_text="</text>",
            cls_token="<cls>",
            token1="<token1>",
            token2="<token2>",
            token3="<token3>"
        
    ):
        self._base_tokenizer = BartTokenizer.from_pretrained(
            pretrained_model_name,
        )

        self.additional_special_tokens = [
            sep_token,
            msa,
            erc,
            emoji,
            comment,
            absa,

            begin_ans,
            end_ans,
            begin_data,
            end_data,

            mosi,
            iemocap,
            meld,
            emowoz,
            emory,
            daily,
            amazon,
            sst,
            imdb,
            sarc,
            absa14,

            speaker_a,
            speaker_b,
            speaker_c,
            speaker_d,
            speaker_e,
            speaker_f,
            speaker_g,
            speaker_h,
            speaker_i,
            context,
            begin_context,
            end_context,

            begin_img,
            end_img,
            begin_audio,
            end_audio,
            begin_text,
            end_text,
            img_feat,
            audio_feat,
            cls_token,
            token1,
            token2,
            token3
        ]

        self._base_tokenizer.add_special_tokens(
            {'additional_special_tokens': self.additional_special_tokens}
        )
        self.sep_token=sep_token
        self.erc=erc
        self.msa=msa
        self.emoji=emoji
        self.comment=comment
        self.absa=absa
        self.begin_ans=begin_ans
        self.end_ans=end_ans
        self.begin_data=begin_data
        self.end_data=end_data
        self.mosi=mosi
        self.meld=meld
        self.iemocap=iemocap
        self.emowoz=emowoz
        self.emory=emory
        self.daily=daily
        self.sst=sst
        self.imdb=imdb
        self.amazon=amazon
        self.sarc=sarc
        self.absa14=absa14
        self.speaker_a=speaker_a
        self.speaker_b=speaker_b
        self.speaker_c=speaker_c
        self.speaker_d=speaker_d
        self.speaker_e=speaker_e
        self.speaker_f=speaker_f
        self.speaker_g=speaker_g
        self.speaker_h=speaker_h
        self.speaker_i=speaker_i 
        self.context = context     
        self.begin_context = begin_context
        self.end_context = end_context
        self.img_feat = img_feat
        self.audio_feat = audio_feat    
        self.begin_img = begin_img
        self.end_img= end_img
        self.begin_audio= begin_audio
        self.end_audio= end_audio
        self.begin_text = begin_text
        self.end_text= end_text
        self.cls_token = cls_token

        self.sep_token_id=self.convert_tokens_to_ids(sep_token)
        self.erc_id=self.convert_tokens_to_ids(erc)
        self.msa_id=self.convert_tokens_to_ids(msa)
        self.emoji_id=self.convert_tokens_to_ids(emoji)
        self.comment_id=self.convert_tokens_to_ids(comment)
        self.absa_id=self.convert_tokens_to_ids(absa)
        self.absa14_id=self.convert_tokens_to_ids(absa14)

        self.begin_ans_id=self.convert_tokens_to_ids(begin_ans)
        self.end_ans_id=self.convert_tokens_to_ids(end_ans)
        self.begin_data_id=self.convert_tokens_to_ids(begin_data)
        self.end_data_id=self.convert_tokens_to_ids(end_data)
        self.mosi_id=self.convert_tokens_to_ids(mosi)
        self.meld_id=self.convert_tokens_to_ids(meld)
        self.iemocap_id=self.convert_tokens_to_ids(iemocap)
        self.emory_id=self.convert_tokens_to_ids(emory)
        self.daily_id=self.convert_tokens_to_ids(daily)
        self.emowoz_id=self.convert_tokens_to_ids(emowoz)
        self.sst_id=self.convert_tokens_to_ids(sst)
        self.imdb_id=self.convert_tokens_to_ids(imdb)
        self.sarc_id=self.convert_tokens_to_ids(sarc)
        self.amazon_id=self.convert_tokens_to_ids(amazon)
        
        self.speaker_a_id=self.convert_tokens_to_ids(speaker_a)
        self.speaker_b_id=self.convert_tokens_to_ids(speaker_b)
        self.speaker_c_id=self.convert_tokens_to_ids(speaker_c)
        self.speaker_d_id=self.convert_tokens_to_ids(speaker_d)
        self.speaker_e_id=self.convert_tokens_to_ids(speaker_e)
        self.speaker_f_id=self.convert_tokens_to_ids(speaker_f)
        self.speaker_g_id=self.convert_tokens_to_ids(speaker_g)
        self.speaker_h_id=self.convert_tokens_to_ids(speaker_h)
        self.speaker_i_id=self.convert_tokens_to_ids(speaker_i)
        self.context_id=self.convert_tokens_to_ids(context)
        self.begin_context_id=self.convert_tokens_to_ids(begin_context)
        self.end_context_id=self.convert_tokens_to_ids(end_context)
        self.img_feat_id = self.convert_tokens_to_ids(img_feat)
        self.audio_feat_id = self.convert_tokens_to_ids(audio_feat)
        self.begin_img_id = self.convert_tokens_to_ids(begin_img)
        self.end_img_id = self.convert_tokens_to_ids(end_img)
        self.begin_audio_id = self.convert_tokens_to_ids(begin_audio)
        self.end_audio_id = self.convert_tokens_to_ids(end_audio)
        self.begin_text_id = self.convert_tokens_to_ids(begin_text)
        self.end_text_id = self.convert_tokens_to_ids(end_text)
        self.cls_token_id = self.convert_tokens_to_ids(cls_token)

        self.vocab_size = self._base_tokenizer.vocab_size
        self.bos_token = self._base_tokenizer.bos_token
        self.bos_token_id = self._base_tokenizer.bos_token_id
        self.eos_token = self._base_tokenizer.eos_token
        self.eos_token_id = self._base_tokenizer.eos_token_id
        self.pad_token = self._base_tokenizer.pad_token
        self.pad_token_id = self._base_tokenizer.pad_token_id
        self.unk_token = self._base_tokenizer.unk_token
        self.unk_token_id = self._base_tokenizer.unk_token_id
        self.mask_token_id=self._base_tokenizer.mask_token_id
    def encode(self, *args, **kwargs):
        return self._base_tokenizer(*args, **kwargs)
    
    
    def pre_encode_condition(self, task_type=None, data_id=None,speaker_token=None,img_num=None,audio_num=None, mlm=None,concat_text=None,context_num=None,context=None,is_pretrain_stage1=True):
        text = []
        if not isinstance(task_type, list):
            task_type = [task_type]
        
        for value in task_type:
            if value == TaskType.ERC:
                text.append(self.erc)
            elif value == TaskType.MSA:
                text.append(self.msa)
            elif value == TaskType.EMOJI:
                text.append(self.emoji)
            elif value == 'comment':
                text.append(self.comment)
            elif value == 'absa':
                text.append(self.absa)
            else:
                raise ValueError('Unexpected task type "{}"'.format(value))
        
        #build the data id
        if not isinstance(data_id, list):
            data_id = [data_id]
       
        for index,value in enumerate(data_id):
            if value == 'mosi' :
                text[index]+=self.begin_data+self.mosi+self.end_data
            elif value == 'mosei' :
                text[index]+=self.begin_data+self.mosi+self.end_data
            elif value == 'iemocap':
                text[index]+=self.begin_data+self.iemocap+self.end_data
            elif value == 'meld':
                text[index]+=self.begin_data+self.meld+self.end_data
            elif value == 'emoji':
                text[index]+=self.begin_data+self.emoji+self.end_data
            elif value == 'emory':
                text[index]+=self.begin_data+self.emory+self.end_data
            elif value == 'daily':
                text[index]+=self.begin_data+self.daily+self.end_data
            elif value == 'sst':
                text[index]+=self.begin_data+self.sst+self.end_data
            elif value == 'imdb':
                text[index]+=self.begin_data+self.imdb+self.end_data
            elif value == 'amazon':
                text[index]+=self.begin_data+self.amazon+self.end_data
            elif value == 'emowoz':
                text[index]+=self.begin_data+self.emowoz+self.end_data
            elif value == 'sarc':
                text[index]+=self.begin_data+self.sarc+self.end_data
            elif value == 'other':
                text[index]+=self.begin_data+'<other>'+self.end_data
            elif value in ['absa14','absa16']:
                text[index]+=self.begin_data+self.absa14+self.end_data
            else:
                raise ValueError('Unexpected data id "{}"'.format(value))
        
        #build speaker token
        if speaker_token is not None:
            if not isinstance(speaker_token, list):
                speaker_token = [speaker_token]
            for index,value in enumerate(speaker_token):
                if value == "speaker_1":
                    text[index]+=self.speaker_a
                elif value == "speaker_2":
                    text[index]+=self.speaker_b
                elif value == "speaker_3":
                    text[index]+=self.speaker_c
                elif value == "speaker_4":
                    text[index]+=self.speaker_d
                elif value == "speaker_5":
                    text[index]+=self.speaker_e
                elif value == "speaker_6":
                    text[index]+=self.speaker_f
                elif value == "speaker_7":
                    text[index]+=self.speaker_g
                elif value == "speaker_8":
                    text[index]+=self.speaker_h
                elif value == "speaker_9":
                    text[index]+=self.speaker_i
                else:
                    text[index]+=self.pad_token

        #build the audio and visual token
        if not isinstance(img_num, list):
            img_num = [img_num]
        for index in range(len(img_num)):
            if img_num[index]>0:
                text[index] += self.begin_img + self.img_feat * img_num[index] + self.end_img
            else:
                text[index] += self.begin_img + self.pad_token + self.end_img
        for index in range(len(audio_num)):
            if audio_num[index]>0:
                text[index] += self.begin_audio + self.audio_feat * audio_num[index] + self.end_audio
            else:
                text[index] += self.begin_audio + self.pad_token + self.end_audio
        # build the mlm token
        # <text> mlm </text>
        if mlm is not None:
            if not isinstance(mlm, list):
                mlm = [mlm]
            if not isinstance(concat_text, list):
                concat_text = [concat_text]
            if is_pretrain_stage1:
                for index, value in enumerate(mlm):
                    if concat_text[index]!='notext':
                        text[index] += self.begin_text + value + self.sep_token + concat_text[index]+ self.end_text
                    else:
                        text[index] += self.begin_text + value + self.end_text
            else:
                for index, value in enumerate(mlm):
                    text[index] += self.begin_text + value + self.end_text

        # build context
        # <context> context tokens </context>        
        if context_num is not None:
            if not isinstance(context_num, list):
                context_num = [context_num]
            if not isinstance(context, list):
                context = [context]
            for i, value in enumerate(context):
                if value!="nocontext":
                    text[i] +=self.begin_context +value+self.end_context
                else:
                    text[i] +=self.begin_context+self.pad_token+self.end_context
         
        # print(text[0])
        encoded = self.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=800,
            return_tensors='pt',
            padding=True
        )
        # build mlm mask and context mask
        if mlm is not None:
            mlm_mask = torch.zeros(encoded['input_ids'].size(), dtype=torch.bool)
            for index, value in enumerate(encoded['input_ids']):
                start = (value == self.begin_text_id).nonzero(as_tuple=True)[0]
                end = (value == self.end_text_id).nonzero(as_tuple=True)[0]
                mlm_mask[index, start + 1: end] = True
                c_start = (value == self.begin_context_id).nonzero(as_tuple=True)[0]
                c_end = (value == self.end_context_id).nonzero(as_tuple=True)[0]
                mlm_mask[index, c_start + 1: c_end] = True
            encoded['mlm_mask'] = mlm_mask
        
        mam_mask = torch.zeros(encoded['input_ids'].size(), dtype=torch.bool)
        for index, value in enumerate(encoded['input_ids']):
            start = (value == self.begin_img_id).nonzero(as_tuple=True)[0]
            end = (value == self.end_img_id).nonzero(as_tuple=True)[0]
            mam_mask[index, start + 1: end] = True
            a_start = (value == self.begin_audio_id).nonzero(as_tuple=True)[0]
            a_end = (value == self.end_audio_id).nonzero(as_tuple=True)[0]
            mam_mask[index, a_start + 1: a_end] = True
        encoded['mam_mask'] = mam_mask
        
               
        return encoded
    def encode_condition(self, task_type=None, data_id=None,speaker_token=None,img_num=None,audio_num=None, text_token=None,context_num=None,context=None):
        """
        tokenize text, image features and audio features
        the output format (after decoded back):
        <task_type> <data><data_id></data> <speaker> <audio><audio_token></audio> <img><img_token></img> <text>"text"</text> <context>"context"</context>
        """
        
        text = []
        # build task types      
        if not isinstance(task_type, list):
            task_type = [task_type]
       
        for value in task_type:
            if value == TaskType.ERC:
                text.append(self.erc)
            elif value == TaskType.MSA:
                text.append(self.msa)
            elif value == TaskType.EMOJI:
                text.append(self.emoji)
            elif value == 'comment':
                text.append(self.comment)
            elif value == 'absa':
                text.append(self.absa)
            else:
                raise ValueError('Unexpected task type "{}"'.format(value))
        
        #build the data id
        if not isinstance(data_id, list):
            data_id = [data_id]
       
        for index,value in enumerate(data_id):
            if value == 'mosi' :
                text[index]+=self.begin_data+self.mosi+self.end_data
            elif value == 'mosei' :
                text[index]+=self.begin_data+self.mosi+self.end_data
            elif value == 'iemocap':
                text[index]+=self.begin_data+self.iemocap+self.end_data
            elif value == 'meld':
                text[index]+=self.begin_data+self.meld+self.end_data
            elif value == 'emoji':
                text[index]+=self.begin_data+self.emoji+self.end_data
            elif value == 'emory':
                text[index]+=self.begin_data+self.emory+self.end_data
            elif value == 'daily':
                text[index]+=self.begin_data+self.daily+self.end_data
            elif value == 'sst':
                text[index]+=self.begin_data+self.sst+self.end_data
            elif value == 'imdb':
                text[index]+=self.begin_data+self.imdb+self.end_data
            elif value == 'amazon':
                text[index]+=self.begin_data+self.amazon+self.end_data
            elif value == 'emowoz':
                text[index]+=self.begin_data+self.emowoz+self.end_data
            elif value == 'sarc':
                text[index]+=self.begin_data+self.sarc+self.end_data
            elif value == 'other':
                text[index]+=self.begin_data+'<other>'+self.end_data
            elif value in ['absa14','absa16']:
                text[index]+=self.begin_data+self.absa14+self.end_data
            else:
                raise ValueError('Unexpected data id "{}"'.format(value))
        
        #build speaker token
        if speaker_token is not None:
            if not isinstance(speaker_token, list):
                speaker_token = [speaker_token]
            for index,value in enumerate(speaker_token):
                if value == "speaker_1":
                    text[index]+=self.speaker_a
                elif value == "speaker_2":
                    text[index]+=self.speaker_b
                elif value == "speaker_3":
                    text[index]+=self.speaker_c
                elif value == "speaker_4":
                    text[index]+=self.speaker_d
                elif value == "speaker_5":
                    text[index]+=self.speaker_e
                elif value == "speaker_6":
                    text[index]+=self.speaker_f
                elif value == "speaker_7":
                    text[index]+=self.speaker_g
                elif value == "speaker_8":
                    text[index]+=self.speaker_h
                elif value == "speaker_9":
                    text[index]+=self.speaker_i
                else:
                    text[index]+=self.pad_token
        
        #bulid the answer sets
        #<ans> ans sets </ans>
        for index,value in enumerate(data_id):
            text[index]+=self.begin_ans+data_ans[value]+self.end_ans
        
       #build the audio and visual token
        if not isinstance(img_num, list):
            img_num = [img_num]
        for index in range(len(img_num)):
            if img_num[index]>0:
                text[index] += self.begin_img + self.img_feat * img_num[index] + self.end_img
            else:
                text[index] += self.begin_img + self.pad_token + self.end_img
        for index in range(len(audio_num)):
            if audio_num[index]>0:
                text[index] += self.begin_audio + self.audio_feat * audio_num[index] + self.end_audio
            else:
                text[index] += self.begin_audio + self.pad_token + self.end_audio
        
            
        # build text
        # <text> text_token </text>
        if text_token is not None:
            if not isinstance(text_token, list):
                text_token = [text_token]
            for index, value in enumerate(text_token):
                if value != "notext":
                    text[index] += self.begin_text + str(value) + self.end_text
                else:
                    text[index] +=self.begin_text + self.pad_token + self.end_text

        # build context
        # <context> context tokens </context>        
        if context_num is not None:
            if not isinstance(context_num, list):
                context_num = [context_num]
            if not isinstance(context, list):
                context = [context]
            for i, value in enumerate(context):
                if value!="nocontext":
                    text[i] +=self.begin_context +value+self.end_context
                else:
                    text[i] +=self.begin_context+self.pad_token+self.end_context
            
        # print(text[0])
        encoded = self.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=800,
            return_tensors='pt',
            padding=True
        )
        
        return encoded
    
    
    def pre_encode_label(self,mlm,context,raw_labels):
        
        text=[]
        if not isinstance(raw_labels, list):
            raw_labels = [raw_labels]
        if not isinstance(mlm, list):
            mlm = [mlm]
        if not isinstance(context, list):
            context = [context]
        
        for i, value in enumerate(raw_labels):
            if context[i]!="nocontext":       
                text.append(self.bos_token+self.begin_text+mlm[i]+self.end_text+self.begin_context+context[i]+self.end_context+self.sep_token+str(value)+self.eos_token)
            else:
                text.append(self.bos_token+self.begin_text+mlm[i]+self.end_text+self.sep_token+str(value)+self.eos_token)

        encoded_label = self.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            return_tensors='pt',
            padding=True
        )
        
        input_ids = encoded_label['input_ids']
        attention_mask = encoded_label['attention_mask']

        output_shape = input_ids[:, 1:].shape
        labels = torch.empty(output_shape, dtype=torch.long)
        decoder_input_ids = torch.empty(output_shape, dtype=torch.long)
        decoder_attention_mask = torch.empty(output_shape, dtype=torch.long)

        # remove <s> from labels, remove </s> from decoder_input_ids
        # remove the element in attention_mask at the same position as </s> in decoder_input_ids
        for i in range(labels.size(0)):
            labels[i] = input_ids[i][1:]
            decoder_input_ids[i] = input_ids[i][:-1]
            decoder_attention_mask[i] = attention_mask[i][:-1]
        
        output = {
            'labels': labels,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask
        }
        
        
        return output
    def pre_encode_label2(self,mlm,context,raw_labels):
        
        text=[]
        if not isinstance(raw_labels, list):
            raw_labels = [raw_labels]
        if not isinstance(mlm, list):
            mlm = [mlm]
        if not isinstance(context, list):
            context = [context]
        
        for i, value in enumerate(raw_labels):
            if context[i]!="nocontext":       
                text.append(self.bos_token+self.begin_text+mlm[i]+self.end_text+self.begin_context+context[i]+self.end_context+self.sep_token+str(value[0])+self.sep_token+str(value[1])+self.sep_token+str(value[2])+self.eos_token)
            else:
                text.append(self.bos_token+self.begin_text+mlm[i]+self.end_text+self.sep_token+str(value[0])+self.sep_token+str(value[1])+self.sep_token+str(value[2])+self.eos_token)
        
        encoded_label = self.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            return_tensors='pt',
            padding=True
        )
        
        input_ids = encoded_label['input_ids']
        attention_mask = encoded_label['attention_mask']

        output_shape = input_ids[:, 1:].shape
        labels = torch.empty(output_shape, dtype=torch.long)
        decoder_input_ids = torch.empty(output_shape, dtype=torch.long)
        decoder_attention_mask = torch.empty(output_shape, dtype=torch.long)

        # remove <s> from labels, remove </s> from decoder_input_ids
        # remove the element in attention_mask at the same position as </s> in decoder_input_ids
        for i in range(labels.size(0)):
            labels[i] = input_ids[i][input_ids[i] != self.bos_token_id]
            decoder_input_ids[i] = input_ids[i][input_ids[i] != self.eos_token_id]
            decoder_attention_mask[i] = attention_mask[i][input_ids[i] != self.eos_token_id]
        
        output = {
            'labels': labels,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask
        }
        
        return output

    def encode_label(self,task_type,raw_labels,data_id):
        if not isinstance(raw_labels, list):
            raw_labels = [raw_labels]
        text=[]
        for i, value in enumerate(raw_labels):    
            text.append(self.bos_token+str(value)+self.eos_token)
       
        encoded_label = self.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            return_tensors='pt',
            padding=True
        )
        
        input_ids = encoded_label['input_ids']
        attention_mask = encoded_label['attention_mask']

        output_shape = input_ids[:, 1:].shape
        labels = torch.empty(output_shape, dtype=torch.long)
        decoder_input_ids = torch.empty(output_shape, dtype=torch.long)
        decoder_attention_mask = torch.empty(output_shape, dtype=torch.long)

        # remove <s> from labels, remove </s> from decoder_input_ids
        # remove the element in attention_mask at the same position as </s> in decoder_input_ids
        for i in range(labels.size(0)):
            labels[i] = input_ids[i][input_ids[i] != self.bos_token_id]
            decoder_input_ids[i] = input_ids[i][input_ids[i] != self.eos_token_id]
            decoder_attention_mask[i] = attention_mask[i][input_ids[i] != self.eos_token_id]
        
        output = {
            'labels': labels,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask
        }
        
        return output

    def decode(self, token_ids, skip_special_tokens=False):
        return self._base_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )

    def convert_tokens_to_ids(self, tokens):
        return self._base_tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        return self._base_tokenizer.convert_ids_to_tokens(ids)

    def get_base_tokenizer(self):
        return self._base_tokenizer

    def __len__(self):
        return len(self._base_tokenizer)
