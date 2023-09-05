
python pretrain.py \
  --dataset concat_pretrain ./datasets/concat_pretrain.pkl \ 
  --dataset concat_val ./datasets/concat_val.pkl \ 
  --checkpoint_dir ./checkpoint_saved \ 
  --model_config ./config/pretrain_base.json \ 
  --checkpoint facebook/bart-base \ 
  --log_dir ./logs \ 
  --validate_loss --amp \ 
  --dropout 0.3
