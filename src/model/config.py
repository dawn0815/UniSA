from transformers import BartConfig


class MultiModalBartConfig(BartConfig):
    def __init__(
            self,
            activation_dropout=0.1,
            extra_pos_embeddings=2,
            activation_function="gelu",
            vocab_size=50320,
            #image.shape:(32x64)
            image_feature_size=64,
            #audio.shape:(157x64)
            audio_feature_size=64,
            image_dim=32*64,
            audio_dim=157*64,
            d_model=768,
            encoder_ffn_dim=3072,
            encoder_layers=6,
            encoder_attention_heads=12,
            decoder_ffn_dim=4096,
            decoder_layers=6,
            decoder_attention_heads=12,
            encoder_layerdrop=0,
            decoder_layerdrop=0,
            attention_dropout=0,
            dropout=0.3,
            max_position_embeddings=1024,
            init_std=0.02,
            classif_dropout=0,
            num_labels=1,
            is_encoder_decoder=True,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            cls_token_id=50306,
            begin_audio_id=50300,
            begin_img_id=50298,
            aud_feat_id=50305,
            img_feat_id=50304,
            begin_text_id=50302,
            end_text_id=50303,
            context_id=50309,
            normalize_before=False,
            add_final_layer_norm=False,
            scale_embedding=False,
            normalize_embedding=True,
            static_position_embeddings=False,
            add_bias_logits=False,
            decoder_start_token_id=0,
            partial_load=(),
            **common_kwargs
    ):
        super(MultiModalBartConfig, self).__init__(
            activation_dropout=activation_dropout,
            extra_pos_embeddings=extra_pos_embeddings,
            activation_function=activation_function,
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_layers=encoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_layers=decoder_layers,
            decoder_attention_heads=decoder_attention_heads,
            encoder_layerdrop=encoder_layerdrop,
            decoder_layerdrop=decoder_layerdrop,
            attention_dropout=attention_dropout,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
            init_std=init_std,
            classif_dropout=classif_dropout,
            num_labels=num_labels,
            is_encoder_decoder=is_encoder_decoder,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            normalize_before=normalize_before,
            add_final_layer_norm=add_final_layer_norm,
            scale_embedding=scale_embedding,
            normalize_embedding=normalize_embedding,
            static_position_embeddings=static_position_embeddings,
            add_bias_logits=add_bias_logits,
            decoder_start_token_id=decoder_start_token_id,
            **common_kwargs
        )

        self.image_feature_size = image_feature_size
        self.audio_feature_size = audio_feature_size
        self.img_feat_id = img_feat_id
        self.aud_feat_id = aud_feat_id
        self.cls_token_id = cls_token_id
        self.partial_load = partial_load
        self.begin_text_id=begin_text_id
        self.end_text_id=end_text_id
        self.context_id=context_id
        self.image_dim=image_dim
        self.audio_dim=audio_dim
        self.begin_audio_id=begin_audio_id
        self.begin_img_id=begin_img_id

