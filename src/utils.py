import logging
import sys
import os
from datetime import datetime,timedelta
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader
import re
golden = {"emoji": ['neutral', 'angry', 'sad', 'happy', 'love', 'fear', 'surprise'],
          "meld": ['joy', 'neutral', 'fear', 'surprise', 'disgust', 'sad', 'angry'],
          "iemocap": ['neutral', 'angry', 'frustrate', 'sad', 'happy', 'excited'],
          "emory": ['joy', 'neutral', 'fear', 'mad', 'peaceful', 'sad', 'powerful'],
          "daily": ['neutral', 'angry', 'sad', 'happy', 'disgust', 'fear', 'surprise'],
          "sst": ['-1.0', '1.0'],
          "imdb": ['-1.0', '1.0'],
          "amazon": ['1.0', '2.0', '3.0', '4.0', '5.0'],
          "sarc": ['sarcasm', 'normal'],
          "emowoz": ['neutral', 'dissatisfied', 'sad', 'happy', 'apology', 'abusive', 'satisfied']}
labels_dic={
        "erc":{"positive":['joy','surprise','happy','excited','peaceful','powerful','satisfied'],"negative":['fear','disgust','sad','angry','frustrate','mad','dissatisfied','apology','abusive'],"neutral":['neutral']},
        "msa":{"positive":['extremely positive','very positive','lightly positive'],"negative":['extremely negative','very negative','lightly negative'],"neutral":['neutral']},
        "comment":{"positive":['4.0','5.0'],"negative":['1.0','2.0'],"neutral":['3.0']}
    }
def convert_aspect_label(pre,label):
    label=label.split('<sep>')[:-1]
    for i in range(len(label)):
        idx=label[i].index(';')
        label[i]=label[i][idx+1:]
    pre=pre.split('<sep>')[:-1]
    for i in range(len(pre)):
        if re.search('positive',pre[i]):
            pre[i]='positive'
        elif re.search('negative',pre[i]):
            pre[i]='negative'
        elif re.search('conflict',pre[i]):
            pre[i]='conflict'
        else:
            pre[i]='neutral'
    for i in range(len(label)-len(pre)):
        pre.append('neutral')
    pres,labels=[],[]
    for i in range(len(label)):
        pres.append(pre[i])
        labels.append(label[i])
    return pres,labels
def mean_pool(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings
def generate_pseudo_label(datasets, model, labels, args, device, collate_fn):

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
    for dataset in datasets:
        for i in range(len(dataset._dataset)):
            dataset[i]["pseudo_labels"] = dataset._dataset[i]["pseudo_labels"]
    
    return datasets

def convert_to_classes(label):
    new_label=""
    label=float(label)
    if abs(label)>2:
        new_label+="extremely "
    elif 1<abs(label)<=2:
        new_label+="very "
    elif 0<abs(label)<=1:
        new_label+="lightly "
    if label<0:
        new_label+="negative"
    elif label>0:
        new_label+="positive"
    else:
        new_label="neutral"
    return new_label
def convert_float_to_fp8(data):
    # 符号位
    y = "1" if data < 0 else "0"
    data = abs(data)
    # 指数
    y += ("1" if data >= 1 else "0")
    y += "1" if data >= 2 or 0.5 <= data < 1 else "0"
    # 尾数
    if data >= 0.5:
        while data >= 1:
            data /= 2
        data -= 0.5
    for i in range(12):
        y += str(int(data * 2 ** (i + 2)) & 1)
    output=""
    for i in y:
        if i=="0":
            output+="zero,"
        else:
            output+="one,"
    return output


def convert_fp8_to_float(data):
    output=""
    for i in data.split(','):
        if i=="zero":
            output+="0"
        else:
            output+="1"
    d = ""
    for i in range(15):
        d += output[i] if "0" <= output[i] <= "9" else "0"
    tail = 0
    for i in range(12):
        tail += int(d[i + 3]) * 2 ** (-i - 1)
    exp = int(d[1]) * 2 + int(d[2])
    tail = ((tail + 1) * 2 ** (exp - 2)) if exp > 0 else (0.5 * tail)
    return tail if d[0] == "0" else -tail

def setup_process(rank, world_size, master_port='12355'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    dist.init_process_group("nccl", rank=rank, world_size=world_size,timeout=timedelta(seconds=288000))


def cleanup_process():
    dist.destroy_process_group()


def save_training_data(path, optimizer=None, scaler=None, epoch=None):
    checkpoint = {
        'optimizer': None if optimizer is None else optimizer.state_dict(),
        'scaler': None if scaler is None else scaler.state_dict(),
        'epoch': epoch
    }

    torch.save(checkpoint, os.path.join(path, 'training_data.pt'))


def load_training_data(path, optimizer=None, scaler=None, map_location=None):
    checkpoint = torch.load(os.path.join(path, 'training_data.pt'), map_location=map_location)

    # if optimizer is not None and 'optimizer' in checkpoint:
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])

    return checkpoint


class Logger:
    def __init__(self, log_dir=None, enabled=True, pad_length=50):
        self._logger = self._get_logger(log_dir) if enabled else None
        self._pad_length = pad_length

    def _pad_message(self, message):
        return (" " + message + " ").center(self._pad_length, '=')

    def info(self, message, pad=False):
        if self._logger is not None:
            message = self._pad_message(message) if pad else message
            self._logger.info(message)

    def line(self):
        if self._logger is not None:
            self._logger.info('=' * self._pad_length)

    @staticmethod
    def _get_logger(log_dir=None):
        """
        get a logger for displaying information to console or log to file (optional)
        :param log_dir: str, logging path. None for not log to file
        :return: logger
        """
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.flush = sys.stdout.flush
        logger.addHandler(stream_handler)

        if log_dir is not None:
            file_handler = logging.FileHandler(log_dir)
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger


class TaskType:
    ERC = 'erc'
    MSA = 'msa'
    EMOJI = 'emoji'
    CAPTION = 'caption'
    REGION_CAPTION = 'region_caption'

    ALL_TYPES = {ERC, MSA, EMOJI, CAPTION, REGION_CAPTION}
