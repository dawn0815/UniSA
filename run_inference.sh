
test_iemocap='./datasets/path_to_iemocap_test_dir'
test_meld='./datasets/path_to_meld_test_dir'
test_emowoz='./datasets/path_to_emowoz_test_dir'
test_daily='./datasets/path_to_daily_test_dir'
test_emory='./datasets/path_to_emory_test_dir'
test_mosi='./datasets/path_to_mosi_test_dir'
test_mosei='./datasets/path_to_mosei_test_dir'
test_sst='./datasets/path_to_sst_test_dir'
test_imdb='./datasets/path_to_imdb_test_dir'
test_absa14='./datasets/path_to_absa14_test_dir'
test_absa16='./datasets/path_to_absa16_test_dir'

python inference.py \
  --dataset iemocap_test ${test_iemocap} \ 
  --dataset meld_test ${test_meld} \ 
  --dataset emowoz_test ${test_emowoz} \ 
  --dataset daily_test ${test_daily} \ 
  --dataset emory_test ${test_emory} \ 
  --dataset mosi_test ${test_mosi} \ 
  --dataset mosei_test ${test_mosei} \ 
  --dataset sst_test ${test_sst} \ 
  --dataset imdb_test ${test_imdb} \ 
  --dataset absa14_test ${test_absa14} \ 
  --dataset absa16_test ${test_absa16} \   
  --checkpoint_dir ./checkpoint_saved \ 
  --model_config ./config/pretrain_base.json \ 
  --checkpoint ./checkpoint_saved/finetune \ 
  --log_dir ./logs \ 
  --validate_loss --amp \ 
