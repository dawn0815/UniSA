import re
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.metrics import accuracy_score
from datetime import datetime
from torch.cuda.amp import autocast
import numpy as np
r = re.compile("[^\d\.]")
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
sent="normal,irony"
emotion="sadness,anger,joy,optimism"
shot="normal,humor"
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
    'absa16':absa14,
    'emotion':emotion,
    'shot':shot
}
def daily_calculate(pre,label):
    if len(label)>0:
        _mf1=f1_score(label,pre, average='micro',labels=['angry','sad','happy','disgust','fear','surprise'])
        _wf1=f1_score(label,pre, average='weighted',labels=['angry','sad','happy','disgust','fear','surprise'])
    else:
        _mf1,_wf1=0,0
    return _mf1,_wf1
def erc_calculate(pre,label):
    if len(label)>0:
        _wf1=f1_score(label,pre, average='weighted')
        
        _acc=accuracy_score(label,pre)
    else:
        _wf1,_acc=0,0
    return _wf1,_acc
def shot_calculate(pre,label):
    if len(label)>0:
        _mf1=f1_score(label,pre, average='macro')
        _wf1=f1_score(label,pre, average='weighted')
    else:
        _mf1,_wf1=0,0
    return _mf1,_wf1
def msa_calculate(pre,label):    
    if len(label)>0:
        mae,corr,acc7,acc2,acc20,f1,f10=eval_mosei_senti(pre,label)
    else:
        mae,corr,acc7,acc2,acc20,f1,f10=0,0,0,0,0,0,0
    return mae,corr,acc7,acc2,acc20,f1,f10
def erc_score(pre,true_labels_list):
    for k in true_labels_list:
        if re.match(k, pre):
            pre=k
    if pre not in true_labels_list:
        pre='neutral'
    return pre
def dot_process(s):
    a=1
    if '-' in s:
        a=-1
    s=r.sub('', s)
    if is_number(s):
        s=float(s)
    else:
        s=0
    if s>3:
        s=3
    elif s<-3:
        s=-3
    return a*s
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
golden = {"meld": ['joy', 'neutral', 'fear', 'surprise', 'disgust', 'sad', 'angry'],
          "iemocap": ['neutral', 'angry', 'frustrate', 'sad', 'happy', 'excited'],
          "emory": ['joy', 'neutral', 'fear', 'mad', 'peaceful', 'sad', 'powerful'],
          "daily": ['neutral', 'angry', 'sad', 'happy', 'disgust', 'fear', 'surprise'],
          "sst": ['-1.0', '1.0'],
          "imdb": ['-1.0', '1.0'],
          "shot":['humor','normal'],
          "emowoz": ['neutral', 'dissatisfied', 'sad', 'happy', 'apology', 'abusive', 'satisfied']}