from datetime import datetime
import re
from torch.cuda.amp import autocast
from early_stopping import EarlyStopping,Pre_EarlyStopping

import numpy as np

from src.utils import convert_fp8_to_float, golden,convert_aspect_label
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.metrics import accuracy_score,classification_report
early_stopping = EarlyStopping()
pre_early_stopping = Pre_EarlyStopping()
import re

emoji_true_list=['neutral','angry','sad','happy','love','fear','surprise']
meld_true_list=['joy','neutral','fear','surprise','disgust','sad','angry']
iemocap_true_list=['neutral','angry','frustrate','sad','happy','excited']
emory_true_list=['joy','neutral','fear','mad','peaceful','sad','powerful']
daily_true_list=['neutral','angry','sad','happy','disgust','fear','surprise']
sst_true_list=['-1.0','1.0']
amazon_true_list=['1.0','2.0','3.0','4.0','5.0']
sarc_true_list=['sarcasm','normal']
emowoz_true_list=['neutral','dissatisfied','sad','happy','apology','abusive','satisfied']
r = re.compile("[^\d\.]")


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False
def mae_calculate(a,b):
    return abs(a-b)
def erc_score(pre,true_labels_list):
    for k in true_labels_list:
        if re.match(k, pre):
            pre=k
    if pre not in true_labels_list:
        pre='neutral'
    return pre
    

def multiclass_acc(preds, truths):

    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
def eval_mosei_senti(msa_pre, msa_label, exclude_zero=False):
    
    msa_pre=np.array([float(e) for e in msa_pre])
    msa_label=np.array([float(e) for e in msa_label])
    non_zeros = np.array([i for i, e in enumerate(msa_label) if e != 0])
    test_preds_a7 = np.clip(msa_pre, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(msa_label, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(msa_pre, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(msa_label, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(msa_pre - msa_label))   # Average L1 distance between preds and truths
    corr = np.corrcoef(msa_pre, msa_label)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    # f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    tmp = msa_label[non_zeros] > 0
    binary_truth_non0 = np.array([int(ele) for ele in tmp])
    tmp = msa_pre[non_zeros] > 0
    binary_preds_non0 =  np.array([int(ele) for ele in tmp])
    f_score_non0 = f1_score(binary_truth_non0, binary_preds_non0, average='weighted')

    acc_2_non0 = accuracy_score(binary_truth_non0, binary_preds_non0)
    binary_truth_has0 = msa_label >= 0
    binary_preds_has0 = msa_pre >= 0
    acc_2 = accuracy_score(binary_truth_has0, binary_preds_has0)
    f_score = f1_score(binary_truth_has0, binary_preds_has0, average='weighted')
    
    return mae, corr, mult_a7, acc_2,acc_2_non0,f_score,f_score_non0
def erc_calculate(pre,label):
    if len(label)>0:
        _wf1=f1_score(label,pre, average='weighted')
        _acc=accuracy_score(label,pre)
    else:
        _wf1,_acc=0,0
    return _wf1,_acc
def absa_calculate(pre,label):
    if len(label)>0:
        _wf1=f1_score(label,pre, average='weighted')
        _acc=accuracy_score(label,pre)
    else:
        _wf1,_acc=0,0
    return _wf1,_acc
def daily_calculate(pre,label):
    if len(label)>0:
        _mf1=f1_score(label,pre, average='micro',labels=['angry','sad','happy','disgust','fear','surprise'])
        _wf1=f1_score(label,pre, average='weighted',labels=['angry','sad','happy','disgust','fear','surprise'])
    else:
        _mf1,_wf1=0,0
    return _mf1,_wf1
def msa_calculate(pre,label):    
    if len(label)>0:
        mae,corr,acc7,acc2,acc20,f1,f10=eval_mosei_senti(pre,label)
    else:
        mae,corr,acc7,acc2,acc20,f1,f10=0,0,0,0,0,0,0
    return mae,corr,acc7,acc2,acc20,f1,f10
def val(
        epoch,
        model,
        tokenizer,
        val_loader,
        device,
        args,
        is_pretrain_stage1,
        logger=None,
        log_interval=1,
        tb_writer=None
):
    total_step = len(val_loader)
    model.eval()
    total_loss = 0
    
    start_time = datetime.now()
    total_len=0

    _pre,_label,_loss=[],[],0
    for i, batch in enumerate(val_loader):
        with autocast(enabled=args.amp):
            outputs = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), batch['image_features'])),     
                audio_features=list(map(lambda x: x.to(device), batch['audio_features'])),
                is_pretrain_stage1=is_pretrain_stage1,
                context_num=batch['context_num'],
                attention_mask=batch['attention_mask'].to(device),             
                decoder_input_ids=batch['decoder_input_ids'].to(device) if 'decoder_input_ids' in batch else None,
                decoder_attention_mask=batch['decoder_attention_mask'].to(device) if 'decoder_attention_mask' in batch else None,
                labels=batch['labels'].to(device),
                raw_labels=batch['raw_labels'],
                task_type=batch['task_type'],
                data_id=batch['data_id']
            )
            loss = outputs[0]['loss']  
            data_len=len(batch['labels'])
        total_loss += loss.item()
        
        total_len+=data_len
        
        if logger is not None:
            logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, ETA: {}'.format(
                epoch + 1,
                args.epochs,
                i + 1,
                total_step,
                loss.item(),
                str((total_step - (i + 1)) / (i + 1) * (datetime.now() - start_time))
            ))
    
    total_loss/=total_len
    
    if logger is not None:
        logger.info('Val loss', pad=True)
        logger.info('Epoch: {}, val loss: {} '.format(epoch + 1,  total_loss))
        logger.line()

    if tb_writer is not None:
        tb_writer.add_scalars('loss/epoch', {'val': total_loss}, epoch + 1)
    
    pre_early_stopping(total_loss, model)
    if pre_early_stopping.early_stop:
        print("Early stopping")
        
        return True
    else:
        return False
def validate(
        epoch,
        model,
        tokenizer,
        val_loader,
        device,
        args,
        logger=None,
        log_interval=1,
        tb_writer=None
):
    total_step = len(val_loader)
    model.eval()
    total_loss = 0
    
    start_time = datetime.now()
    total_len=0

    sarc_pre,sarc_label,sarc_loss=[],[],0
    amazon_pre,amazon_label,amazon_loss=[],[],0
    emowoz_pre,emowoz_label,emowoz_loss=[],[],0
    sst_pre,sst_label,sst_loss=[],[],0
    imdb_pre,imdb_label,imdb_loss=[],[],0
    meld_pre,meld_label,meld_loss=[],[],0
    iemocap_pre,iemocap_label,iemocap_loss=[],[],0
    daily_pre,daily_label,daily_loss=[],[],0
    emory_pre,emory_label,emory_loss=[],[],0
    emoji_pre,emoji_label,emoji_loss=[],[],0
    mosi_pre,mosi_label,mosi_loss=[],[],0
    mosei_pre,mosei_label,mosei_loss=[],[],0
    absa14_pre,absa14_label=[],[]
    absa16_pre,absa16_label=[],[]
    
    for i, batch in enumerate(val_loader):
        with autocast(enabled=args.amp):
            outputs = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), batch['image_features'])),     
                audio_features=list(map(lambda x: x.to(device), batch['audio_features'])),
                is_pretrain_stage1=False,
                context_num=batch['context_num'],
                attention_mask=batch['attention_mask'].to(device),             
                decoder_input_ids=batch['decoder_input_ids'].to(device) if 'decoder_input_ids' in batch else None,
                decoder_attention_mask=batch['decoder_attention_mask'].to(device) if 'decoder_attention_mask' in batch else None,
                labels=batch['labels'].to(device),
                raw_labels=batch['raw_labels'],
                task_type=batch['task_type'],
                data_id=batch['data_id']
            )
            loss = outputs[0]['loss']  
            data_len=len(batch['labels'])
        meld_loss += outputs[0]['meld_loss']
        iemocap_loss += outputs[0]['iemocap_loss']
        mosi_loss += outputs[0]['mosi_loss']
        mosei_loss += outputs[0]['mosei_loss']
        emoji_loss += outputs[0]['emoji_loss']
        daily_loss += outputs[0]['daily_loss']
        emory_loss += outputs[0]['emory_loss']
        sst_loss += outputs[0]['sst_loss']
        imdb_loss += outputs[0]['imdb_loss']
        amazon_loss += outputs[0]['amazon_loss']
        emowoz_loss += outputs[0]['emowoz_loss']
        sarc_loss += outputs[0]['sarc_loss']
        total_loss += loss.item()
        
        total_len+=data_len
        
        if logger is not None:
            logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, ETA: {}'.format(
                epoch + 1,
                args.epochs,
                i + 1,
                total_step,
                loss.item(),
                str((total_step - (i + 1)) / (i + 1) * (datetime.now() - start_time))
            ))
        with autocast(enabled=args.amp):
            outputs = model.module.generate(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), batch['image_features'])),
                audio_features=list(map(lambda x: x.to(device), batch['audio_features'])),
                data_id=batch['data_id'],
                context_num=batch['context_num'],
                attention_mask=batch['attention_mask'].to(device),
                max_length=30,
                min_length=4,
                num_beams=args.num_beams,
                num_return_sequences=args.num_gen,
                top_p=args.top_p if hasattr(args, 'top_p') else 1.0,
                top_k=args.top_k if hasattr(args, 'top_k') else 10,
                early_stopping=False,
                do_sample=True
            )
        
        for j in range(len(batch['data_id'])):
            pre=tokenizer.decode(outputs[j], skip_special_tokens=True)
            if batch['data_id'][j] in ["mosi", "mosei"]:
                pre = convert_fp8_to_float(pre)
                batch['raw_labels'][j]=convert_fp8_to_float(batch['raw_labels'][j])
            elif batch['data_id'][j] in ["absa14", "absa16"]:
                pre,batch['raw_labels'][j]=convert_aspect_label(pre,batch['raw_labels'][j])
                eval(batch['data_id'][j] + "_pre").extend(pre)
                eval(batch['data_id'][j] + "_label").extend(batch['raw_labels'][j])
            else:
                pre = erc_score(pre, golden[batch['data_id'][j]])
                batch['raw_labels'][j]=str(batch['raw_labels'][j])
            if batch['data_id'][j] not in ["absa14", "absa16"]:
                eval(batch['data_id'][j] + "_pre").append(pre)
                eval(batch['data_id'][j] + "_label").append(batch['raw_labels'][j])
        if (i + 1) % log_interval == 0:
            logger.info('Generating, Step [{}/{}], ETA: {}'.format(
                i + 1,
                total_step,
                str((total_step - (i + 1)) / (i + 1) * (datetime.now() - start_time))
            ))
    
    total_loss/=total_len
    absa14_f1,absa14_acc=absa_calculate(absa14_pre,absa14_label)
    absa16_f1,absa16_acc=absa_calculate(absa16_pre,absa16_label)
    sarc_f1,sarc_acc=erc_calculate(sarc_pre,sarc_label)
    amazon_f1,amazon_acc=erc_calculate(amazon_pre,amazon_label)
    emowoz_f1,emowoz_acc=erc_calculate(emowoz_pre,emowoz_label)
    sst_f1,sst_acc=erc_calculate(sst_pre,sst_label)
    imdb_f1,imdb_acc=erc_calculate(imdb_pre,imdb_label)
    emoji_f1,emoji_acc=erc_calculate(emoji_pre,emoji_label)
    meld_f1,meld_acc=erc_calculate(meld_pre,meld_label)
    emory_f1,emory_acc=erc_calculate(emory_pre,emory_label)
    daily_mf1,daily_wf1=daily_calculate(daily_pre,daily_label)
    iemocap_f1,iemocap_acc=erc_calculate(iemocap_pre,iemocap_label)
    mosi_mae,mosi_corr,mosi_acc7,mosi_acc2,mosi_acc20,mosi_f1,mosi_f10=msa_calculate(mosi_pre,mosi_label)
    mosei_mae,mosei_corr,mosei_acc7,mosei_acc2,mosei_acc20,mosei_f1,mosei_f10=msa_calculate(mosei_pre,mosei_label)
    if logger is not None:
        logger.info('Val loss', pad=True)
        logger.info('Epoch: {}, val loss: {}, meld_acc: {},meld_f1: {}, mosei_mae: {},mosei_corr: {}, mosei_acc7: {},mosei_acc2: {},mosei_acc20: {}, mosei_f1: {},mosei_f10: {}, \
                                              iemocap_acc: {},iemocap_f1: {}, mosi_mae: {},mosi_corr: {}, mosi_acc7: {},mosi_acc2: {},mosi_acc20: {}, mosi_f1: {},mosi_f10: {}, \
                                              emoji_acc: {},emoji_f1: {},emory_acc: {},emory_f1: {},daily_mf1: {},daily_wf1: {},sst_acc: {},sst_f1: {},imdb_acc: {},imdb_f1: {} ,amazon_acc: {},amazon_f1: {},emowoz_acc: {},emowoz_f1: {},sarc_acc: {},sarc_f1: {},\
                                               absa14_acc: {},absa14_f1: {},absa16_acc: {},absa16_f1: {}  '.format(
                  epoch + 1,   total_loss,     meld_acc,     meld_f1,    mosei_mae,     mosei_corr,    mosei_acc7,    mosei_acc2,    mosei_acc20,     mosei_f1,    mosei_f10 ,
                                              iemocap_acc,    iemocap_f1,  mosi_mae,       mosi_corr,    mosi_acc7,     mosi_acc2,    mosi_acc20,      mosi_f1,  mosi_f10  ,
                                              emoji_acc, emoji_f1, emory_acc,emory_f1,daily_mf1,daily_wf1,sst_acc,sst_f1,imdb_acc,imdb_f1,amazon_acc,amazon_f1,emowoz_acc,emowoz_f1,sarc_acc,sarc_f1,
                                              absa14_acc,absa14_f1,absa16_acc,absa16_f1
        ))
        logger.line()

    if tb_writer is not None:
        tb_writer.add_scalars('loss/epoch', {'val': total_loss}, epoch + 1)
    
    acc=meld_acc+meld_f1+iemocap_acc+iemocap_f1+mosi_acc2+mosi_f1+mosei_acc2+mosei_f1+emory_acc+daily_mf1+emowoz_f1+sarc_f1+sst_f1+imdb_f1+absa14_f1
    
    early_stopping(acc, model)
    if early_stopping.early_stop:
        print("Early stopping")
        
        return True
    else:
        return False

