# Based on transformers.modeling_bart

from typing import Optional, Tuple
import re
import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_bart import (
    PretrainedBartModel,
    BartDecoder,
    BartClassificationHead,
    BartForConditionalGeneration,
    _make_linear_from_emb,
    _prepare_bart_decoder_inputs,
    _filter_out_falsey_values
)
from transformers import BartTokenizer
from src.model.config import MultiModalBartConfig
from src.model.mixins import GenerationMixin, FromPretrainedMixin
from src.model.modules import MultiModalBartEncoder

import re
r = re.compile("[^\d\.]")
task1=['amazon','emoji','sst','imdb']
task2=['mosi','mosei']
task3=['daily','emory','meld','iemocap','emowoz']
task4=['sarc']
task5=['absa14','absa16']
classes = {"positive": 0, "negative": 1, "neutral": 2}
def contrastive_loss(representations, raw_labels, temperature=0.05):
    """
    :param representations: FloatTensor 
    :param labels: List(str) 
    :param temperature
    :return: loss
    """
    distance = torch.exp(-torch.norm(representations.unsqueeze(1) - representations.unsqueeze(0), dim=-1)) / temperature
    labels = torch.LongTensor([classes[x] for x in raw_labels]).to(representations.device)
    mask = torch.cuda.BoolTensor(labels.unsqueeze(1) == labels.unsqueeze(0)).to(representations.device)
    
    return torch.sum(torch.sum(distance * mask, dim=-1) / torch.sum(distance, dim=-1))
def mean_pool(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min = 1e-9)
    mean_embeddings = sum_embeddings/sum_mask
    return mean_embeddings
# This is based on transformers.BartModel
# The modifications are:
# - BartConfig -> MultiModalBartConfig
# - BartEncoder -> MultiModalBartEncoder
# - added image_features in forward
class MultiModalBartModel(FromPretrainedMixin, PretrainedBartModel):
    def __init__(self, config: MultiModalBartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.embed_data_ids=nn.Embedding(15, config.d_model)
        self.encoder = MultiModalBartEncoder(config, self.shared,self.embed_data_ids)
        self.decoder = BartDecoder(config, self.shared)
        self.init_weights()

    def forward(
            self,
            input_ids,
            image_features,
            audio_features,
            is_pretrain_stage1,
            data_id,
            context_num=None,
            raw_labels=None,
            attention_mask=None,
            decoder_input_ids=None,
            encoder_outputs: Optional[Tuple] = None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None
        
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    image_features=image_features,
                    audio_features=audio_features,
                    data_id=data_id,
                    context_num=context_num,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
            
        assert isinstance(encoder_outputs, tuple)
        if is_pretrain_stage1:
            representations = mean_pool(encoder_outputs[0],attention_mask)
            c_loss = contrastive_loss(representations, raw_labels, temperature=0.05)
        else:
            c_loss=0
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=decoder_cached_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )

        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs: Tuple = _filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0], torch.Tensor)
        encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)
        return decoder_outputs + encoder_outputs + (c_loss,)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly

class BartGenerationHead(nn.Module):
    """Head for generation tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        out_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim,bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(inner_dim, out_dim,bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
# This is based on transformers.BartForConditionalGeneration
# The modifications are:
# - BartConfig -> MultiModalBartConfig
# - BartModel -> MultiModalBartModel
# - added image_features and audio_features in forward
# - changed loss computation in forward
# - added image_features and audio_features in prepare_inputs_for_generation
# - rewrite generate function

class MultiModalBartForPreTraining(FromPretrainedMixin, GenerationMixin, PretrainedBartModel):
    base_model_prefix = "model"
    #task 1: MLM
    #task 2: sentiment analysis
    def __init__(self, config: MultiModalBartConfig):
        super().__init__(config)
        self.cls_token_id = config.cls_token_id
        self.model = MultiModalBartModel(config)
        
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
          
    def forward(
            self,
            input_ids,
            image_features,
            audio_features,
            is_pretrain_stage1,
            data_id=None,
            context_num=None,
            labels=None,
            raw_labels=None,
            task_type=None,
            
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            **unused,
    ):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param encoder_outputs:
        :param decoder_input_ids:
        :param decoder_attention_mask:
        :param decoder_cached_states:
        :param labels: Labels for computing the language modeling loss.
            Indices should either be in [0, ..., config.vocab_size] or -100.
            Tokens with indices set to -100 are ignored (masked), the loss is only computed for the tokens
            with labels in [0, ..., config.vocab_size].
        :param mrm_labels: labels for computing masked region modeling loss. similar to labels
        :param use_cache:
        :param output_attentions:
        :param output_hidden_states:
        :param unused:
        :return: :obj:tuple(torch.FloatTensor) comprising various elements depending on the configuration and inputs:
            loss (optional, returned when labels is provided) dict of FloatTensor of shape (1,):
                {
                    'loss': total loss,
                    'lm_loss': Masked language modeling loss (if labels is given),
                    'mrm_loss': masked region modeling loss (if mrm_labels is given).
                }
            prediction_scores (:obj: torch.FloatTensor of shape :obj:(batch_size, sequence_length, config.vocab_size))
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            hidden_states (:obj:tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed
                or when config.output_hidden_states=True):
                Tuple of :obj:FloatTensor (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:(batch_size, sequence_length, hidden_size).

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed
                or when config.output_attentions=True):
                Tuple of :obj:FloatTensor (one for each layer) of shape
                :obj:(batch_size, num_heads, sequence_length, sequence_length).

                Attentions weights after the attention softmax, used to compute the weighted average
                in the self-attention heads.
        """
        
        if (labels is not None):
            use_cache = False
        
        outputs = self.model(
            input_ids,
            image_features,
            audio_features,
            is_pretrain_stage1,
            data_id=data_id,
            context_num=context_num,
            raw_labels=raw_labels,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        losses={}
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        outputs = (lm_logits,) + outputs[1:]
        lm_loss = 0
        c_loss = 0
        if labels is not None:
            labels = labels.clone()
            labels[labels == self.cls_token_id] = -100
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
            lm_loss *= self.config.lm_loss_factor
            losses['lm_loss'] = lm_loss
        if is_pretrain_stage1:
            c_loss = outputs[-1]
        losses['loss'] = lm_loss + c_loss
        outputs = (losses,) + outputs
        return outputs


# This is based on transformers.BartForConditionalGeneration
# The modifications are:
# - BartConfig -> MultiModalBartConfig
# - BartModel -> MultiModalBartModel
# - added image_features in forward
class MultiModalBartForConditionalGeneration(FromPretrainedMixin, GenerationMixin, PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, config: MultiModalBartConfig):
        super().__init__(config)
        self.cls_token_id = config.cls_token_id
        self.model = MultiModalBartModel(config)
        
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.task_head_1 = BartGenerationHead(
            config.d_model,
            config.d_model,
            self.model.shared.num_embeddings,
            0.1,
        )
        self._init_weights(self.task_head_1.dense)
        self._init_weights(self.task_head_1.out_proj)
        self.task_head_2 = BartGenerationHead(
            config.d_model,
            config.d_model,
            self.model.shared.num_embeddings,
            0.1,
        )
        self._init_weights(self.task_head_2.dense)
        self._init_weights(self.task_head_2.out_proj)
        self.task_head_3 = BartGenerationHead(
            config.d_model,
            config.d_model,
            self.model.shared.num_embeddings,
            0.1,
        )
        self._init_weights(self.task_head_3.dense)
        self._init_weights(self.task_head_3.out_proj)
        self.task_head_4 = BartGenerationHead(
            config.d_model,
            config.d_model,
            self.model.shared.num_embeddings,
            0.1,
        )
        self._init_weights(self.task_head_4.dense)
        self._init_weights(self.task_head_4.out_proj)
        self.task_head_5 = BartGenerationHead(
            config.d_model,
            config.d_model,
            self.model.shared.num_embeddings,
            0.1,
        )
        self._init_weights(self.task_head_5.dense)
        self._init_weights(self.task_head_5.out_proj)
        
    def forward(
            self,
            input_ids,
            image_features,
            audio_features,
            is_pretrain_stage1,
            data_id,
            context_num,
            
            labels=None,
            raw_labels=None,
            task_type=None,
            
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            **unused,
    ):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param encoder_outputs:
        :param decoder_input_ids:
        :param decoder_attention_mask:
        :param decoder_cached_states:
        :param labels: Labels for computing the language modeling loss.
            Indices should either be in [0, ..., config.vocab_size] or -100.
            Tokens with indices set to -100 are ignored (masked), the loss is only computed for the tokens
            with labels in [0, ..., config.vocab_size].
        :param mrm_labels: labels for computing masked region modeling loss. similar to labels
        :param use_cache:
        :param output_attentions:
        :param output_hidden_states:
        :param unused:
        :return: :obj:tuple(torch.FloatTensor) comprising various elements depending on the configuration and inputs:
            loss (optional, returned when labels is provided) dict of FloatTensor of shape (1,):
                {
                    'loss': total loss,
                    'lm_loss': Masked language modeling loss (if labels is given),
                    'mrm_loss': masked region modeling loss (if mrm_labels is given).
                }
            prediction_scores (:obj: torch.FloatTensor of shape :obj:(batch_size, sequence_length, config.vocab_size))
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            hidden_states (:obj:tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed
                or when config.output_hidden_states=True):
                Tuple of :obj:FloatTensor (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:(batch_size, sequence_length, hidden_size).

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed
                or when config.output_attentions=True):
                Tuple of :obj:FloatTensor (one for each layer) of shape
                :obj:(batch_size, num_heads, sequence_length, sequence_length).

                Attentions weights after the attention softmax, used to compute the weighted average
                in the self-attention heads.
        """
        
        if (labels is not None):
            use_cache = False
        
        outputs = self.model(
            input_ids,
            image_features,
            audio_features,
            data_id=data_id,
            context_num=context_num,
            is_pretrain_stage1=False,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        def erc_loss(logits,labels):
            labels=labels.clone()
            labels[labels == self.cls_token_id] = -100
            a=logits.view(-1, self.config.vocab_size)
            
            lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.reshape(-1))
            
            return lm_loss
        def vocab_limit(logits,v_list,tokenizer):
            vocab = tokenizer.get_vocab()
            v_indexes = [vocab[word] for word in v_list if word in vocab]
            v_indexes.sort()
            return logits
        losses={}
        
        loss_fct = nn.CrossEntropyLoss()
        lm_logits=[]
        data_id_gen=[]
        num_beams=3
        for i in range(len(data_id)):
            for k in range(num_beams):
                data_id_gen.append(data_id[i])
        if labels is not None:
            for i in range(len(data_id)):
                if data_id[i] in task1:
                    lm_logits.append(self.task_head_1(outputs[0][i]))
                elif data_id[i] in task2:
                    lm_logits.append(self.task_head_2(outputs[0][i]))
                elif data_id[i] in task3:
                    lm_logits.append(self.task_head_3(outputs[0][i]))
                elif data_id[i] in task4:
                    lm_logits.append(self.task_head_4(outputs[0][i]))
                elif data_id[i] in task5:
                    lm_logits.append(self.task_head_5(outputs[0][i]))
        else:
            for i in range(len(data_id_gen)):
                if data_id_gen[i] in task1:
                    lm_logits.append(self.task_head_1(outputs[0][i]))
                elif data_id_gen[i] in task2:
                    lm_logits.append(self.task_head_2(outputs[0][i]))
                elif data_id_gen[i] in task3:
                    lm_logits.append(self.task_head_3(outputs[0][i]))
                elif data_id_gen[i] in task4:
                    lm_logits.append(self.task_head_4(outputs[0][i]))
                elif data_id_gen[i] in task5:
                    lm_logits.append(self.task_head_5(outputs[0][i]))
        lm_logits=torch.stack(lm_logits).cuda()

        #lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        
        outputs = (lm_logits,) + outputs[1:] 
        absa14_loss,absa16_loss=0,0
        mosi_loss,mosei_loss=0,0
        iemocap_loss,meld_loss=0,0
        emowoz_loss,daily_loss,emory_loss=0,0,0
        sarc_loss,amazon_loss,imdb_loss,sst_loss,emoji_loss=0,0,0,0,0
        if labels is not None:
            for i in range(len(data_id)):
                if data_id[i]=='iemocap':
                    iemocap_loss+=erc_loss(lm_logits[i],labels[i])
                elif data_id[i]=='meld':       
                    meld_loss+=erc_loss(lm_logits[i],labels[i])
                elif data_id[i]=='daily':       
                    daily_loss+=erc_loss(lm_logits[i],labels[i])
                elif data_id[i]=='emory':        
                    emory_loss+=erc_loss(lm_logits[i],labels[i])
                elif data_id[i]=='emoji':       
                    emoji_loss+=erc_loss(lm_logits[i],labels[i])
                elif data_id[i]=='mosi':      
                    mosi_loss+=erc_loss(lm_logits[i],labels[i])
                elif data_id[i]=='mosei':       
                    mosei_loss+=erc_loss(lm_logits[i],labels[i])
                elif data_id[i]=='sst':       
                    sst_loss+=erc_loss(lm_logits[i],labels[i])
                elif data_id[i]=='imdb':       
                    imdb_loss+=erc_loss(lm_logits[i],labels[i])
                elif data_id[i]=='amazon':       
                    amazon_loss+=erc_loss(lm_logits[i],labels[i])
                elif data_id[i]=='emowoz':        
                    emowoz_loss+=erc_loss(lm_logits[i],labels[i])
                elif data_id[i]=='sarc':       
                    sarc_loss+=erc_loss(lm_logits[i],labels[i])
                elif data_id[i]=='absa14':       
                    absa14_loss+=erc_loss(lm_logits[i],labels[i])
                elif data_id[i]=='absa16':       
                    absa16_loss+=erc_loss(lm_logits[i],labels[i])
                elif data_id[i]=='other':       
                    amazon_loss+=erc_loss(lm_logits[i],labels[i])
        
    
            losses['meld_loss']=meld_loss
            losses['iemocap_loss']=iemocap_loss
            losses['mosi_loss']=mosi_loss
            losses['mosei_loss']=mosei_loss
            losses['emoji_loss']=emoji_loss
            losses['daily_loss']=daily_loss
            losses['emory_loss']=emory_loss
            losses['sst_loss']=sst_loss
            losses['imdb_loss']=imdb_loss
            losses['amazon_loss']=amazon_loss
            losses['emowoz_loss']=emowoz_loss
            losses['sarc_loss']=sarc_loss
            losses['loss1']=(amazon_loss+imdb_loss+sst_loss+emoji_loss)
            losses['loss2']=(mosi_loss+mosei_loss)
            losses['loss3']=(emowoz_loss+emory_loss+daily_loss+iemocap_loss+meld_loss)
            losses['loss4']=(sarc_loss)
            losses['loss5']=(absa14_loss+absa16_loss)

            loss=[iemocap_loss,meld_loss,mosi_loss,mosei_loss,emoji_loss,emory_loss,daily_loss,sst_loss,imdb_loss,amazon_loss,emowoz_loss,sarc_loss,absa14_loss,absa16_loss]
            total_loss=0
            for i in range(len(loss)):
                total_loss+=loss[i]
            losses['loss']=total_loss
            
            outputs = (losses,)+outputs
        return outputs




