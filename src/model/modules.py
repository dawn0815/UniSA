import random
import math

import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from transformers.modeling_bart import (
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    invert_mask,
    EncoderLayer,
    LayerNorm,
)

from src.model.config import MultiModalBartConfig

data_dict={
    'mosi': 1,
    'mosei': 2,
    'iemocap':3,
    'meld': 4,
    'emoji':5,
    'emory':6,
    'daily':7,
    'sst':8,
    'imdb':9,
    'amazon':10,
    'emowoz':11,
    'sarc':12,
    'absa14':13,
    'absa16':13
}
class ContextRNN(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, lengths):
        
        
        package = nn.utils.rnn.pack_padded_sequence(inputs, lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False)
        result, hn = super().forward(package)
        output, lens = nn.utils.rnn.pad_packed_sequence(result, batch_first=self.batch_first, total_length=inputs.shape[self.batch_first])
        return output, hn

class ImageEmbedding(nn.Module):
    def __init__(self, image_dim, final_dim):
        super(ImageEmbedding, self).__init__()
        self.linear = nn.Linear(image_dim, final_dim)

    def forward(self, image_features):
        img_len = list(map(len, image_features))
        non_empty_features = list(filter(lambda x: len(x) > 0, image_features))

        embedded = None
        if len(non_empty_features) > 0:
            img_tensor = torch.cat(non_empty_features, dim=0)
            img_tensor=img_tensor.to(torch.float32)
            embedded = self.linear(img_tensor).type(torch.float32)

        output = []
        index = 0
        for l in img_len:
            if l > 0:
                output.append(embedded[index: index + l])
            else:
                output.append(torch.empty(0))
            index += l
        return output

class AudioEmbedding(nn.Module):
    def __init__(self, audio_dim, final_dim):
        super(AudioEmbedding, self).__init__()
        self.linear = nn.Linear(audio_dim, final_dim)

    def forward(self, audio_features):
        aud_len = list(map(len, audio_features))
        non_empty_features = list(filter(lambda x: len(x) > 0, audio_features))

        embedded = None
        if len(non_empty_features) > 0:
            aud_tensor = torch.cat(non_empty_features, dim=0)
            aud_tensor=aud_tensor.to(torch.float32)
            embedded = self.linear(aud_tensor).type(torch.float32)

        output = []
        index = 0
        for l in aud_len:
            if l > 0:
                output.append(embedded[index: index + l])
            else:
                output.append(torch.empty(0))
            index += l
        return output


# This is copied from transformers.BartEncoder
# The modifications are:
# - added embed_images layer
# - added _embed_multi_modal function
# - added image_features in forward
class MultiModalBartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:EncoderLayer.

    Args:
        config: MultiModalBartConfig
    """

    def __init__(self, config: MultiModalBartConfig, embed_tokens,embed_data_ids):
        super().__init__()
        self.img_feat_id = config.img_feat_id
        self.aud_feat_id = config.aud_feat_id
        self.cls_token_id = config.cls_token_id
        self.begin_text_id=config.begin_text_id
        self.end_text_id=config.end_text_id
        self.context_id = config.context_id
        self.begin_audio_id=config.begin_audio_id
        self.begin_img_id=config.begin_img_id
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        self.indentity = nn.Identity()

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings
        self.a_encoder = AudioEmbedding(config.audio_feature_size,config.d_model)
        self.v_encoder = ImageEmbedding(config.image_feature_size,config.d_model)
        self.embed_tokens = embed_tokens
        self.embed_data_ids=embed_data_ids
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None
    def _context_rnn(self,context_token,context_num):
        contexts_emb=[]
        lstm = ContextRNN(input_size=768, hidden_size=768, batch_first=True, num_layers=1, bidirectional=True).cuda()
        non_empty_features = list(filter(lambda x: len(x) > 0, context_token))
        
        if len(non_empty_features) > 0:
            for i in range(len(non_empty_features)): 
                context=self.embed_tokens(non_empty_features[i].cuda())
                context=context.flatten(1)
                context=self.context_linear_1(context)
                context=self.context_dropout(context)
                context=self.context_linear_2(context)
                contexts_emb.append(context)
            contexts_emb=torch.stack(contexts_emb).cuda()
            c_num=[]
            for i in range(len(context_num)):
                if context_num[i]!=0:
                    c_num.append(context_num[i])
            out, _ = lstm(contexts_emb, torch.Tensor(c_num))
            output=[]
            for i in range(len(c_num)):
                output.append(self.context_linear_3(out[i]))
        contexts=[]
        j=0
        for i in range(len(context_num)):
            if context_num[i]>0:
                contexts.append(output[i-j][:context_num[i],:].type(torch.float32))
            else:
                contexts.append(torch.empty(0).type(torch.float32))
                j+=1
        return contexts
    def _embed_multi_modal(self, input_ids,audio_features,image_features):
        """embed textual,visual and audio inputs,then combine them into one embedding""" 
        embedded = self.embed_tokens(input_ids)
        mask = (input_ids == self.aud_feat_id) 
        embedded_audio = self.a_encoder(audio_features)
        
        if not embedded_audio[0].dtype == torch.float32:
            embedded = embedded.half()  
        for index, value in enumerate(embedded_audio):
            if len(value) > 0:
                a_start = (input_ids[index] == self.begin_audio_id).nonzero(as_tuple=True)[0]
                for idx,v in enumerate(input_ids[index]):
                    if v==self.aud_feat_id:
                        embedded[index][idx] = value[idx-a_start-1]
        
        mask = (input_ids == self.img_feat_id)   
        embedded_images = self.v_encoder(image_features) 
        if not embedded_images[0].dtype == torch.float32:
            embedded = embedded.half()
        for index, value in enumerate(embedded_images):
            if len(value) > 0:
                i_start = (input_ids[index] == self.begin_img_id).nonzero(as_tuple=True)[0]
                for idx,v in enumerate(input_ids[index]):
                    if v==self.img_feat_id:
                        embedded[index][idx] = value[idx-i_start-1]
        return embedded
    def _embed_data_ids(self, input_ids,data_id,embed_data_ids):
        """embed textual,visual and audio inputs,then combine them into one embedding""" 
        embedded_data=[]
        for i in range(len(data_id)):
            l=len(input_ids[i])
            a=torch.ones(l)*data_dict[data_id[i]]
            b=torch.zeros(l)
            
            left=int(torch.nonzero(input_ids[i]==self.begin_text_id).squeeze())
            right=int(torch.nonzero(input_ids[i]==self.end_text_id).squeeze())
            a_1=b[:left+1]
            a_2=a[left+1:right]
            a_3=b[right:]
            c=torch.cat([a_1,a_2,a_3],0)
            
            c=embed_data_ids(c.long().cuda())
            embedded_data.append(c)
        return torch.stack(embedded_data)

    def forward(
            self,
            input_ids,
            image_features,
            audio_features,
            data_id,
            context_num,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False
    ):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param output_attentions:
        :param output_hidden_states:
        :return: Tuple comprised of:
            - x (Tensor): the last encoder layer's output of
              shape (src_len, batch, embed_dim)
            - encoder_states (List[Tensor]): all intermediate
              hidden states of shape (src_len, batch, embed_dim).
              Only populated if output_hidden_states: is True.
            - all_attentions (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
        """
       
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)
        # contexts=self._context_rnn(context_token,context_num)
        inputs_embeds = self._embed_multi_modal(input_ids,audio_features,image_features) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        embed_data=self._embed_data_ids(input_ids,data_id,self.embed_data_ids)
        
        x = inputs_embeds + embed_pos + embed_data
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

            if output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)

        return x, encoder_states, all_attentions


