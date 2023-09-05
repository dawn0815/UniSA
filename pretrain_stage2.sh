train_iemocap='./datasets/path_to_iemocap_train_dir'
train_meld='./datasets/path_to_meld_train_dir'
train_emowoz='./datasets/path_to_emowoz_train_dir'
train_daily='./datasets/path_to_daily_train_dir'
train_emory='./datasets/path_to_emory_train_dir'
train_mosi='./datasets/path_to_mosi_train_dir'
train_mosei='./datasets/path_to_mosei_train_dir'
train_amazon='./datasets/path_to_amazon_train_dir'

val_iemocap='./datasets/path_to_iemocap_val_dir'
val_meld='./datasets/path_to_meld_val_dir'
val_emowoz='./datasets/path_to_emowoz_val_dir'
val_daily='./datasets/path_to_daily_val_dir'
val_emory='./datasets/path_to_emory_val_dir'
val_mosi='./datasets/path_to_mosi_val_dir'
val_mosei='./datasets/path_to_mosei_val_dir'
val_amazon='./datasets/path_to_amazon_val_dir'

python pretrain2.py \
  --dataset iemocap_pretrain ${train_iemocap} \ 
  --dataset meld_pretrain ${train_meld} \ 
  --dataset emowoz_pretrain ${train_emowoz} \ 
  --dataset daily_pretrain ${train_daily} \ 
  --dataset emory_pretrain ${train_emory} \ 
  --dataset mosi_pretrain ${train_mosi} \ 
  --dataset mosei_pretrain ${train_mosei} \ 
  --dataset amazon_pretrain ${train_amazon} \ 
  --dataset iemocap_val ${val_iemocap} \ 
  --dataset meld_val ${val_meld} \ 
  --dataset emowoz_val ${val_emowoz} \ 
  --dataset daily_val ${val_daily} \ 
  --dataset emory_val ${val_emory} \ 
  --dataset mosi_val ${val_mosi} \ 
  --dataset mosei_val ${val_mosei} \ 
  --dataset amazon_val ${val_amazon} \  
  --checkpoint_dir ./checkpoint_saved \ 
  --model_config ./config/pretrain_base.json \ 
  --checkpoint ./checkpoint_saved/pretrain_stage2 \ 
  --log_dir ./logs \ 
  --validate_loss --amp \ 
  --dropout 0.3 --lr 5e-6 \ 
  --continue_training
