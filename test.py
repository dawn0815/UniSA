import torch

from torch.cuda.amp import autocast
import argparse
import json
import os
from datetime import datetime
from transformers import BartTokenizer
import torch
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from pcgrad import PCGradAMP
from src.data.collation import PretrainCollator
from src.data.dataset import (PretrainDataset2, COMMENTALLDataset)
from src.data.tokenization import ConditionTokenizer
from src.model.config import MultiModalBartConfig
from src.model.model import MultiModalBartForConditionalGeneration, MultiModalBartForPreTraining, MultiModalBartModel
from src.training import pretrain, fine_tune
from src.validation import validate
from src.generation import generate_text
from src.utils import (Logger, cleanup_process, load_training_data,
                       save_training_data, setup_process)
from new_sampler import BatchSchedulerSampler

classes = {"positive": 0, "negative": 1, "neutral": 2}


def mean_pool(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings


def contrastive_loss(representations, labels, temperature=0.05):
    """
    :param representations: FloatTensor 大小为(BS, 768)
    :param labels: List(str) 长度为BS，表示每个样本是positive, negative, neutral。
    :param temperature
    :return: loss
    """
    distance = torch.exp(-torch.norm(representations.unsqueeze(1) - representations.unsqueeze(0), dim=-1)) / temperature
    labels = torch.LongTensor([classes[x] for x in labels]).to(representations.device)
    mask = torch.BoolTensor(labels.unsqueeze(1) == labels.unsqueeze(0)).to(representations.device)
    return torch.mean(torch.sum(distance * mask, dim=-1) / torch.sum(distance, dim=-1))

labels={
        "erc":{"positive":['joy','surprise','happy','excited','peaceful','powerful','satisfied'],"negative":['fear','disgust','sad','angry','frustrate','mad','dissatisfied','apology','abusive'],"neutral":['neutral']},
        "msa":{"positive":['extremely positive','very positive','lightly positive'],"negative":['extremely negative','very negative','lightly negative'],"neutral":['neutral']},
        "comment":{"positive":['4.0','5.0'],"negative":['1.0','2.0'],"neutral":['3.0']}
    }
class PretrainDataset2(Dataset):
    def __init__(self, data_dir,data_id,use_text=True,use_image=True,use_audio=True):
        self._data_dir = data_dir 
        self._data_id = data_id     
        self._dataset = pickle.load(open(data_dir,'rb'))
        self._use_text = use_text
        self._use_image = use_image
        self._use_audio = use_audio
        
    def __getitem__(self, index):
        output={}
        if self._use_image:
            output['image_features']=self._dataset[index]['image_features'] 
        if self._use_audio:
            output['audio_features']=self._dataset[index]['audio_features'] 
        output['text']=self._dataset[index]['text'] if self._use_text else 'notext'
        if self._data_id in ['mosi','mosei']:    
            output['labels']=convert_to_classes(self._dataset[index]['label'])
        else:
            output['labels']=self._dataset[index]['label']
        output['task_type']=self._dataset[index]['task_type']
        output['context']=self._dataset[index]['context']
        output['speaker']=self._dataset[index]['speaker']
        output['data_id']=self._data_id      
        output['index']=self._dataset[index]['index'] if 'index' in self._dataset[index] else -1
        
        return output
    def __len__(self):
        return len(self._dataset)
def generate_pseudo_label(datasets, model, labels, args, device, collate_fn):
    """
    以某一个任务的label为标准，对所有label进行聚类。
    datasets: 现在输入的数据集仍然是一个list，只是要把按数据聚类改为按任务聚类，通过dataset._data_id可以知道data ID，从而确定任务类型
    model: 模型，会使用model.module.encoder()获取representations
    labels: 字典格式，形式如{"erc": {"positive": ["joy", "..."], "negative": ["sad", "..."], "neutral": ["neutral"]}}
            如果没有neutral，用[]表示。将所有预训练使用的数据集以及它们的label分好正/负/中性标签之后传入。
            labels的key顺序尽量和datasets相同。
    device, args, collate_fn: 必要参数
    """
    print(datasets[0]._dataset[0]['text'])
    for dataset in datasets:
        for i in range(len(dataset._dataset)):
            dataset._dataset[i]["pseudo_labels"] = []
    # TODO 1. 跑出所有样本的emotion representation。
    label_to_task = {}   # dict，key值为任务名字如erc，value值为list，大小等于对应任务下所有数据集的大小之和，表示每个样本的label
    label_to_dataset = {}   # dict，key值为数据集名字如meld，value值为list，大小等于对应数据集的大小，表示每个样本的label
    representations_to_task = {}
    representations_to_dataset = {}

    for dataset in datasets:
        data_id = dataset._data_id
        task = dataset._dataset[0]['task_type']
        if data_id in ['imdb', 'sst']:
            task = 'comment2'
        if task not in representations_to_task:
            representations_to_task[task] = []
        if task not in label_to_task:
            label_to_task[task] = []
        label_to_dataset[data_id] = []
        representations_to_dataset[data_id] = []
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        model.eval()
        print('task :', task)
        for i, batch in enumerate(data_loader):
            with torch.no_grad():
                output = model.module.encoder(input_ids=batch['input_ids'].to(device),
                                              image_features=list(map(lambda x: x.to(device), batch['image_features'])),
                                              audio_features=list(map(lambda x: x.to(device), batch['audio_features'])),
                                              context_num=batch['context_num'],
                                              data_id=batch['data_id'],
                                              attention_mask=batch['attention_mask'].to(device))
                representation = mean_pool(output[0], batch['attention_mask'].to(device))

                representations_to_task[task].append(representation)
                representations_to_dataset[data_id].append(representation)
                raw_labels = [str(x) for x in batch['raw_labels']]
                label_to_task[task].extend(raw_labels)
                label_to_dataset[data_id].extend(raw_labels)
        representations_to_dataset[data_id] = torch.cat(representations_to_dataset[data_id], dim=0)
    for task in labels:
        representations_to_task[task] = torch.cat(representations_to_task[task], dim=0)
        print(representations_to_task[task].shape)
    
    # representations[task]为(N, 768)大小的向量。N=数据集的大小。
    # TODO 2. 根据emotion representation和有标注的样本进行聚类。
    #         聚类过程为：首先求出有标注样本的中心点，然后求出无标注样本到所有中心点的距离，选择最近的类别加入。可以进行类别平衡。
    for task in labels:
        # TODO 2-1 每个task样本归类
        positive = labels[task]["positive"]
        negative = labels[task]["negative"]
        centroids = {x: [] for x in labels[task]["positive"] + labels[task]["negative"] + labels[task]["neutral"]}
        for vector, label in zip(representations_to_task[task], label_to_task[task]):
            centroids[label].append(vector)

        # TODO 2-2 求出每个任务、所有类别的聚类中心点。
        #          考虑到后面可能要进行类别平衡和赋权，可以顺便记下簇内平均距离。
        intra_distances = {}
        for label in centroids:
            centroids[label] = torch.cat(centroids[label], dim=0)
            centroid = torch.mean(centroids[label], dim=0)
            intra_distances[label] = torch.mean(torch.norm(centroids[label] - centroid, dim=-1))
            centroids[label] = centroid
        # intra_distances为dict，key值为label，value值为类内间距。
        # TODO 2-3 求出每个样本到聚类中心点的距离。
        for dataset in datasets:
            data_id = dataset._data_id
            task2 = dataset._dataset[0]['task_type']
            if data_id in ['imdb', 'sst']:
                task2 = 'comment2'
            if task2 == task:
                for j in range(len(dataset._dataset)):
                    dataset._dataset[j]["pseudo_labels"].append(label_to_dataset[data_id][j])
            else:
                distances = [{"positive": {}, "negative": {}} for _ in range(representations_to_dataset[data_id].shape[0])]
                for label2 in centroids:
                    flag_p = label2 in positive
                    flag_n = label2 in negative
                    distance = torch.norm(representations_to_dataset[data_id] - centroids[label2], dim=-1).tolist()  # (N,)
                    for j, item in enumerate(distance):
                        if flag_p:
                            distances[j]["positive"][label2] = item
                        elif flag_n:
                            distances[j]["negative"][label2] = item
                # TODO 2-4 根据距离和样本的正/负性，将样本归类到对应的类别。
                #          task: 当前有标注的数据集  task2: 被标注的样本所来自的数据集
                for j, (label, distance) in enumerate(zip(label_to_dataset[data_id], distances)):
                    # TODO 2-4-1 判断样本的极性。如果是neutral，则直接赋值neutral。否则给出具体类别。
                    if label in labels[task2]["neutral"]:
                        if task=="comment":
                            dataset._dataset[j]["pseudo_labels"].append("3.0")
                        else:
                            dataset._dataset[j]["pseudo_labels"].append("neutral")
                    elif label in labels[task2]["positive"]:
                        # TODO 2-4-2 列举出样本到positive的所有label的距离并从大到小排序，取距离最小的label作为样本的label。
                        candidates = sorted(distance["positive"].items(), key=lambda x: x[1])
                        dataset._dataset[j]["pseudo_labels"].append(candidates[0][0])
                    else:
                        candidates = sorted(distance["negative"].items(), key=lambda x: x[1])
                        dataset._dataset[j]["pseudo_labels"].append(candidates[0][0])
    
    return datasets
pretrain_dataset_list=[]

pretrain_dataset_list.append(PretrainDataset2(
        args.dataset['mosi_pretrain'],data_id="mosi",use_text=True,use_image=True,use_audio=True))

pretrain_dataset_list.append(PretrainDataset2(
        args.dataset['mosei_pretrain'],data_id="mosei",use_text=True,use_image=True,use_audio=True))
pretrain_dataset_list = generate_pseudo_label(pretrain_dataset_list, module, labels_dic, args, device, collate)
train_dataset = ConcatDataset(pretrain_dataset_list)

DATASET_NAMES = (
    'iemocap_pretrain', 'meld_pretrain', 'mosi_pretrain', 'mosei_pretrain', 'emoji_pretrain', 'emory_pretrain',
    'daily_pretrain', 'sst_pretrain', 'imdb_pretrain', 'amazon_pretrain', 'emowoz_pretrain', 'sarc_pretrain',
    'iemocap_val', 'meld_val', 'mosi_val', 'mosei_val', 'emoji_val', 'daily_val', 'emory_val', 'sst_val', 'imdb_val',
    'amazon_val', 'emowoz_val', 'sarc_val',
    'iemocap_test', 'meld_test', 'mosi_test', 'mosei_test', 'emoji_test', 'daily_test', 'emory_test', 'sst_test',
    'imdb_test', 'amazon_test', 'emowoz_test', 'sarc_test'
)


def main(rank, args):
    # ============ logging, initialization and directories ==============

    if not args.cpu:
        setup_process(rank, args.gpu_num, master_port=args.master_port)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_path = os.path.join(args.checkpoint_dir, timestamp)
    tb_writer = None
    log_dir = os.path.join(args.log_dir, timestamp)

    # make log dir and tensorboard writer if log_dir is specified
    if rank == 0 and args.log_dir is not None:
        os.makedirs(log_dir)
        tb_writer = SummaryWriter(log_dir=log_dir)

    logger = Logger(log_dir=os.path.join(log_dir, 'log.txt'), enabled=(rank == 0))

    # make checkpoint dir if not exist
    if rank == 0 and not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
        logger.info('Made checkpoint directory: "{}"'.format(checkpoint_path))

    logger.info('Initialed with {} GPU(s)'.format(args.gpu_num), pad=True)
    for k, v in vars(args).items():
        logger.info('{}: {}'.format(k, v))

    # =========================== model =============================

    logger.info('Loading model...')

    if args.cpu:
        device = 'cpu'
        map_location = device
    else:
        device = torch.device("cuda:{}".format(rank))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    tokenizer = ConditionTokenizer()

    if args.model_config is not None:
        bart_config = MultiModalBartConfig.from_dict(json.load(open(args.model_config)))
    else:
        bart_config = MultiModalBartConfig.from_pretrained(args.checkpoint)

    if args.dropout is not None:
        bart_config.dropout = args.dropout
    if args.attention_dropout is not None:
        bart_config.attention_dropout = args.attention_dropout
    if args.classif_dropout is not None:
        bart_config.classif_dropout = args.classif_dropout
    if args.activation_dropout is not None:
        bart_config.activation_dropout = args.activation_dropout

    if args.checkpoint:
        model = MultiModalBartModel.from_pretrained(
            args.checkpoint,
            config=bart_config,
            error_on_mismatch=False
        )
    else:
        model = MultiModalBartModel(bart_config)

    model.to(device)

    if not args.cpu:
        torch.cuda.set_device(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scaler = GradScaler() if args.amp else None
    # grad_optimizer = PCGradAMP(4, optimizer, scaler = scaler, reduction='sum', cpu_offload = False)
    epoch = 0
    if args.continue_training:
        epoch = load_training_data(
            args.checkpoint,
            optimizer=optimizer,
            scaler=scaler,
            map_location=map_location
        )['epoch'] + 1

    # =========================== data =============================

    collate_fn = PretrainCollator(
        tokenizer,
        has_label=True,
        mlm_enabled=False,
        mlm_probability=args.mlm_probability,
        lm_max_len=args.lm_max_len,
        max_img_num=args.max_img_num,
        max_aud_num=args.max_aud_num
    )

    pretrain_dataset_list=[]
    val_dataset_list=[]
    if 'mosi_pretrain' in args.dataset:
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['mosi_pretrain'],data_id="mosi",use_text=True,use_image=True,use_audio=True))
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['mosi_pretrain'],data_id="mosi",use_text=True,use_image=False,use_audio=False))
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['mosi_pretrain'],data_id="mosi",use_text=True,use_image=False,use_audio=True))
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['mosi_pretrain'],data_id="mosi",use_text=True,use_image=True,use_audio=False))
    if 'mosi_val' in args.dataset:    
        val_dataset_list.append(PretrainDataset2(
            args.dataset['mosi_val'],data_id="mosi",use_text=True,use_image=True,use_audio=True))
    
    if 'mosei_pretrain' in args.dataset:
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['mosei_pretrain'],data_id="mosei",use_text=True,use_image=True,use_audio=True))
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['mosei_pretrain'],data_id="mosei",use_text=True,use_image=False,use_audio=False))
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['mosei_pretrain'],data_id="mosei",use_text=True,use_image=False,use_audio=True))
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['mosei_pretrain'],data_id="mosei",use_text=True,use_image=True,use_audio=False))
    if 'mosei_val' in args.dataset:    
        val_dataset_list.append(PretrainDataset2(
            args.dataset['mosei_val'],data_id="mosei",use_text=True,use_image=True,use_audio=True))

    if 'iemocap_pretrain' in args.dataset:
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['iemocap_pretrain'],data_id="iemocap",use_text=True,use_image=True,use_audio=True))
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['iemocap_pretrain'],data_id="iemocap",use_text=True,use_image=False,use_audio=False))
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['iemocap_pretrain'],data_id="iemocap",use_text=True,use_image=False,use_audio=True))
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['iemocap_pretrain'],data_id="iemocap",use_text=True,use_image=True,use_audio=False))
    if 'iemocap_val' in args.dataset:    
        val_dataset_list.append(PretrainDataset2(
            args.dataset['iemocap_val'],data_id="iemocap",use_text=True,use_image=True,use_audio=True))

    if 'meld_pretrain' in args.dataset:
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['meld_pretrain'],data_id="meld",use_text=True,use_image=True,use_audio=True))
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['meld_pretrain'],data_id="meld",use_text=True,use_image=False,use_audio=False))
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['meld_pretrain'],data_id="meld",use_text=True,use_image=False,use_audio=True))
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['meld_pretrain'],data_id="meld",use_text=True,use_image=True,use_audio=False))
    if 'meld_val' in args.dataset:    
        val_dataset_list.append(PretrainDataset2(
            args.dataset['meld_val'],data_id="meld",use_text=True,use_image=True,use_audio=True))
        
    if 'emory_pretrain' in args.dataset:
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['emory_pretrain'],data_id="emory",use_text=True,use_image=False,use_audio=False))
    if 'emory_val' in args.dataset:    
        val_dataset_list.append(PretrainDataset2(
            args.dataset['emory_val'],data_id="emory",use_text=True,use_image=False,use_audio=False))
    
    if 'daily_pretrain' in args.dataset:
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['daily_pretrain'],data_id="daily",use_text=True,use_image=False,use_audio=False))
    if 'daily_val' in args.dataset:    
        val_dataset_list.append(PretrainDataset2(
            args.dataset['daily_val'],data_id="daily",use_text=True,use_image=False,use_audio=False))

    if 'emowoz_pretrain' in args.dataset:
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['emowoz_pretrain'],data_id="emowoz",use_text=True,use_image=False,use_audio=False))
    if 'emowoz_val' in args.dataset:    
        val_dataset_list.append(PretrainDataset2(
            args.dataset['emowoz_val'],data_id="emowoz",use_text=True,use_image=False,use_audio=False))

    if 'sst_pretrain' in args.dataset:
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['sst_pretrain'],data_id="sst",use_text=True,use_image=False,use_audio=False))
    if 'sst_val' in args.dataset:    
        val_dataset_list.append(PretrainDataset2(
            args.dataset['sst_val'],data_id="sst",use_text=True,use_image=False,use_audio=False))

    if 'imdb_pretrain' in args.dataset:
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['imdb_pretrain'],data_id="imdb",use_text=True,use_image=False,use_audio=False))
    if 'imdb_val' in args.dataset:    
        val_dataset_list.append(PretrainDataset2(
            args.dataset['imdb_val'],data_id="imdb",use_text=True,use_image=False,use_audio=False))

    if 'amazon_pretrain' in args.dataset:
        pretrain_dataset_list.append(PretrainDataset2(
            args.dataset['amazon_pretrain'],data_id="amazon",use_text=True,use_image=False,use_audio=False))
    if 'amazon_val' in args.dataset:    
        val_dataset_list.append(PretrainDataset2(
            args.dataset['amazon_val'],data_id="amazon",use_text=True,use_image=False,use_audio=False))
    
    
    datasets_new = generate_pseudo_label(pretrain_dataset_list, model, labels, args, device, collate_fn)
    # train_sampler = DistributedSampler(
    #     train_dataset,
    #     num_replicas=args.gpu_num,
    #     rank=rank,
    #     shuffle=True
    # )

    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     #sampler=BatchSchedulerSampler(dataset=train_dataset,batch_size=args.batch_size),
    #     sampler=train_sampler,
    #     collate_fn=collate_fn
    # )


#     val_dataset_list = []
#     #val dataset
#     if 'mosi_val' in args.dataset:
#         val_dataset_list.append(MSADataset(
#             args.dataset['mosi_val'],data_id='mosi',use_text=True,use_image=True,use_audio=True))
#     if 'mosei_val' in args.dataset:
#         val_dataset_list.append(MSADataset(
#             args.dataset['mosei_val'],data_id='mosei',use_text=True,use_image=True,use_audio=True))
#     if 'iemocap_val' in args.dataset:
#         val_dataset_list.append(ERCDataset(
#             args.dataset['iemocap_val'],data_id='iemocap',use_text=True,use_image=True,use_audio=True))
#     if 'meld_val' in args.dataset:
#         val_dataset_list.append(ERCDataset(
#             args.dataset['meld_val'],data_id='meld',use_text=True,use_image=True,use_audio=True))
#     if 'emory_val' in args.dataset:
#         val_dataset_list.append(ERCDataset(
#             args.dataset['emory_val'],data_id='emory',use_text=True,use_image=False,use_audio=False))
#     if 'emowoz_val' in args.dataset:
#         val_dataset_list.append(ERCDataset(
#             args.dataset['emowoz_val'],data_id='emowoz',use_text=True,use_image=False,use_audio=False))
#     if 'daily_val' in args.dataset:
#         val_dataset_list.append(ERCDataset(
#             args.dataset['daily_val'],data_id='daily',use_text=True,use_image=False,use_audio=False))
#     if 'emoji_val' in args.dataset:
#         val_dataset_list.append(EMOJIDataset(
#             args.dataset['emoji_val'],data_id='emoji'))
#     if 'sst_val' in args.dataset:
#         val_dataset_list.append(COMMENTDataset(
#             args.dataset['sst_val'],data_id='sst'))
#     if 'imdb_val' in args.dataset:
#         val_dataset_list.append(COMMENTDataset(
#             args.dataset['imdb_val'],data_id='imdb'))
#     if 'amazon_val' in args.dataset:
#         val_dataset_list.append(COMMENTDataset(
#             args.dataset['amazon_val'],data_id='amazon'))
#     if 'sarc_val' in args.dataset:
#         val_dataset_list.append(SARCDataset(
#             args.dataset['sarc_val'],data_id='sarc',use_text=True,use_image=False,use_audio=False))

#    ##test dataset
#     if 'mosi_test' in args.dataset:
#         val_dataset_list.append(MSADataset(
#             args.dataset['mosi_test'],data_id='mosi',use_text=True,use_image=True,use_audio=True))
#     if 'mosei_test' in args.dataset:
#         val_dataset_list.append(MSADataset(
#             args.dataset['mosei_test'],data_id='mosei',use_text=True,use_image=True,use_audio=True))
#     if 'iemocap_test' in args.dataset:
#         val_dataset_list.append(ERCDataset(
#             args.dataset['iemocap_test'],data_id='iemocap',use_text=True,use_image=True,use_audio=True))
#     if 'meld_test' in args.dataset:
#         val_dataset_list.append(ERCDataset(
#             args.dataset['meld_test'],data_id='meld',use_text=True,use_image=True,use_audio=True))
#     if 'emory_test' in args.dataset:
#         val_dataset_list.append(ERCDataset(
#             args.dataset['emory_test'],data_id='emory',use_text=True,use_image=False,use_audio=False))
#     if 'emowoz_test' in args.dataset:
#         val_dataset_list.append(ERCDataset(
#             args.dataset['emowoz_test'],data_id='emowoz',use_text=True,use_image=False,use_audio=False))
#     if 'daily_test' in args.dataset:
#         val_dataset_list.append(ERCDataset(
#             args.dataset['daily_test'],data_id='daily',use_text=True,use_image=False,use_audio=False))
#     if 'emoji_test' in args.dataset:
#         val_dataset_list.append(EMOJIDataset(
#             args.dataset['emoji_test'],data_id='emoji'))
#     if 'sst_test' in args.dataset:
#         val_dataset_list.append(COMMENTDataset(
#             args.dataset['sst_test'],data_id='sst'))
#     if 'imdb_test' in args.dataset:
#         val_dataset_list.append(COMMENTDataset(
#             args.dataset['imdb_test'],data_id='imdb'))
#     if 'amazon_test' in args.dataset:
#         val_dataset_list.append(COMMENTDataset(
#             args.dataset['amazon_test'],data_id='amazon'))
#     if 'sarc_test' in args.dataset:
#         val_dataset_list.append(SARCDataset(
#             args.dataset['sarc_test'],data_id='sarc',use_text=True,use_image=False,use_audio=False))


#     val_dataset=ConcatDataset(val_dataset_list)
#     val_loader = DataLoader(
#         dataset=val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         collate_fn=collate_fn
#     )


def parse_args():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument('--dataset', action='append', nargs=2, metavar=('DATASET_NAME', 'DATASET_PATH'), required=True,
                        help='append a dataset, one of "{}"'.format('", "'.join(DATASET_NAMES)))
    parser.add_argument('--checkpoint_dir', required=True, type=str,
                        help='where to save the checkpoint')

    # path
    parser.add_argument('--log_dir', default=None, type=str,
                        help='path to output log files, not output to file if not specified')
    parser.add_argument('--model_config', default=None, type=str,
                        help='path to load model config')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='name or path to load weights')

    # training and evaluation
    parser.add_argument('--no_mrm', dest='mrm_enabled', action='store_true', default=True,
                        help='do not use masked region modelling')
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of training epoch')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--num_gen', default=1, type=int,
                        help='number of generated sentence on validation')
    parser.add_argument('--num_beams', default=3, type=int,
                        help='level of beam search on validation')
    parser.add_argument('--continue_training', action='store_true',
                        help='continue training, load optimizer and epoch from checkpoint')
    parser.add_argument('--validate_loss', action='store_true',
                        help='compute the validation loss at the end of each epoch')
    parser.add_argument('--validate_score', action='store_true',
                        help='compute the validation score (BLEU, METEOR, etc.) at the end of each epoch')
    parser.add_argument('--max_img_num', type=int, default=32,
                        help='max number of image feature per data entry')
    parser.add_argument('--max_aud_num', type=int, default=157,
                        help='max number of audio feature per data entry')
    parser.add_argument('--lm_max_len', type=int, default=360,
                        help='max number of words for the language modeling per data entry')
    parser.add_argument('--context_max_len', type=int, default=480,
                        help='max number of words for the context per data entry')
    parser.add_argument('--mlm_probability', type=float, default=0.3,
                        help='mask probability for MLM')

    # dropout
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout rate for the transformer. This overwrites the model config')
    parser.add_argument('--classif_dropout', default=0, type=float,
                        help='dropout rate for the classification layers. This overwrites the model config')
    parser.add_argument('--attention_dropout', default=0, type=float,
                        help='dropout rate for the attention layers. This overwrites the model config')
    parser.add_argument('--activation_dropout', default=0.1, type=float,
                        help='dropout rate for the activation layers. This overwrites the model config')

    # hardware and performance
    parser.add_argument('--gpu_num', default=1, type=int,
                        help='number of GPUs in total')
    parser.add_argument('--cpu', default=False, action='store_false',
                        help='if only use cpu to run the model')
    parser.add_argument('--amp', action='store_true',
                        help='whether or not to use amp')
    parser.add_argument('--master_port', type=str, default='12777',
                        help='master port for DDP')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='#workers for data loader')

    args = parser.parse_args()

    if args.gpu_num != 1 and args.cpu:
        raise ValueError('--gpu_num are not allowed if --cpu is set to true')

    if args.checkpoint is None and args.model_config is None:
        raise ValueError('--model_config and --checkpoint cannot be empty at the same time')

    # check repeated dataset names
    names = [k for k, _ in args.dataset]
    if len(names) != len(set(names)):
        raise ValueError('repeated datasets')

    # check if dataset exists
    args.dataset = {k: v for k, v in args.dataset}
    for name in names:
        if name not in DATASET_NAMES:
            raise ValueError('"{}" is not a valid dataset'.format(name))

    return args


if __name__ == '__main__':
    args = parse_args()

    mp.spawn(
        main,
        args=(args,),
        nprocs=args.gpu_num,
        join=True
    )
