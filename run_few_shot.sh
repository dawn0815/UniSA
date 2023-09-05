train_emotion='./datasets/path_to_emotion_fewshot_train_dir'
test_emotion='./datasets/path_to_emotion_fewshot_test_dir'

python finetune.py \
  --dataset emotion_few_train ${train_emotion} \ 
  --dataset emotion_few_test ${test_emotion} \   
  --checkpoint_dir ./checkpoint_saved \ 
  --model_config ./config/pretrain_base.json \ 
  --checkpoint ./checkpoint_saved/finetune \ 
  --log_dir ./logs \ 
  --validate_loss --amp \ 
  --continue_training
