train_iemocap='./datasets/path_to_iemocap_train_dir'
train_meld='./datasets/path_to_meld_train_dir'
train_emowoz='./datasets/path_to_emowoz_train_dir'
train_daily='./datasets/path_to_daily_train_dir'
train_emory='./datasets/path_to_emory_train_dir'
train_mosi='./datasets/path_to_mosi_train_dir'
train_mosei='./datasets/path_to_mosei_train_dir'
train_sst='./datasets/path_to_sst_train_dir'
train_imdb='./datasets/path_to_imdb_train_dir'
train_absa14='./datasets/path_to_absa14_train_dir'
train_absa16='./datasets/path_to_absa16_train_dir'

val_iemocap='./datasets/path_to_iemocap_val_dir'
val_meld='./datasets/path_to_meld_val_dir'
val_emowoz='./datasets/path_to_emowoz_val_dir'
val_daily='./datasets/path_to_daily_val_dir'
val_emory='./datasets/path_to_emory_val_dir'
val_mosi='./datasets/path_to_mosi_val_dir'
val_mosei='./datasets/path_to_mosei_val_dir'
val_sst='./datasets/path_to_sst_val_dir'
val_imdb='./datasets/path_to_imdb_val_dir'
val_absa14='./datasets/path_to_absa14_val_dir'
val_absa16='./datasets/path_to_absa16_val_dir'

checkpoint_saved='./checkpoint_saved/pretrain_stage2'  #dir to the model weights (checkpoint)

python finetune.py \
  --dataset iemocap_pretrain ${train_iemocap} \ 
  --dataset meld_pretrain ${train_meld} \ 
  --dataset emowoz_pretrain ${train_emowoz} \ 
  --dataset daily_pretrain ${train_daily} \ 
  --dataset emory_pretrain ${train_emory} \ 
  --dataset mosi_pretrain ${train_mosi} \ 
  --dataset mosei_pretrain ${train_mosei} \ 
  --dataset sst_pretrain ${train_sst} \ 
  --dataset imdb_pretrain ${train_imdb} \ 
  --dataset absa14_pretrain ${train_absa14} \ 
  --dataset absa16_pretrain ${train_absa16} \ 
  --dataset iemocap_val ${val_iemocap} \ 
  --dataset meld_val ${val_meld} \ 
  --dataset emowoz_val ${val_emowoz} \ 
  --dataset daily_val ${val_daily} \ 
  --dataset emory_val ${val_emory} \ 
  --dataset mosi_val ${val_mosi} \ 
  --dataset mosei_val ${val_mosei} \ 
  --dataset sst_val ${val_sst} \ 
  --dataset imdb_val ${val_imdb} \ 
  --dataset absa14_val ${val_absa14} \ 
  --dataset absa16_val ${val_absa16} \   
  --checkpoint_dir ./checkpoint_saved \ 
  --model_config ./config/pretrain_base.json \ 
  --checkpoint ${checkpoint_saved} \  
  --log_dir ./logs \ 
  --validate_loss --amp \ 
  --dropout 0.1 --lr 5e-6 \ 
  --continue_training
