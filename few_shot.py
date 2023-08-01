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
from src.data.collation import Collator
from src.data.dataset import (MSADataset,EMOJIDataset,COMMENTDataset,ERCDataset,SARCDataset,COMMENTALLDataset,ABSADataset)
from src.data.tokenization import ConditionTokenizer
from src.model.config import MultiModalBartConfig
from src.model.model import MultiModalBartForConditionalGeneration,MultiModalBartForPreTraining
from src.training import pretrain,fine_tune
from src.validation import val
from src.generation import generate_text
from src.utils import (Logger, cleanup_process, load_training_data,
                       save_training_data, setup_process)
from new_sampler import BatchSchedulerSampler
DATASET_NAMES = (
   'sarc_few_train','emoji_few_train','emotion_few_train','sent_few_train',
   'sarc_few_test','emoji_few_test','emotion_few_test','sent_few_test'
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
        model = MultiModalBartForConditionalGeneration.from_pretrained(
            args.checkpoint,
            config=bart_config,
            error_on_mismatch=False
        )
    else:
        model = MultiModalBartForConditionalGeneration(bart_config)
    
    model.to(device)

    if not args.cpu:
        torch.cuda.set_device(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = AdamW(model.parameters(),lr=args.lr,weight_decay=0.001)
    scaler = GradScaler() if args.amp else None
    #grad_optimizer = PCGradAMP(4, optimizer, scaler = scaler, reduction='sum', cpu_offload = False)
    epoch = 0
    if args.continue_training:
        epoch = load_training_data(
            args.checkpoint,
            optimizer=optimizer,
            scaler=scaler,
            map_location=map_location
        )['epoch'] + 1

    # =========================== data =============================

    logger.info('Loading data...')

    collate_fn = Collator(
        tokenizer,
        lm_max_len=args.lm_max_len,
        max_aud_num=args.max_aud_num,
        context_max_len=args.context_max_len,
        max_img_num=args.max_img_num
    )
    
    
    dataset_list = []
    if 'sarc_few_train' in args.dataset:
        dataset_list.append(SARCDataset(
            args.dataset['sarc_few_train'],data_id='sarc',use_text=True,use_image=False,use_audio=False))

    if 'emoji_few_train' in args.dataset:
        dataset_list.append(EMOJIDataset(
            args.dataset['emoji_few_train'],data_id='emoji'))
    if 'emotion_few_train' in args.dataset:
        dataset_list.append(EMOJIDataset(
            args.dataset['emotion_few_train'],data_id='shot'))
    if 'sent_few_train' in args.dataset:
        dataset_list.append(EMOJIDataset(
            args.dataset['sent_few_train'],data_id='sent'))
    
    train_dataset = ConcatDataset(dataset_list)
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.gpu_num,
        rank=rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        #sampler=BatchSchedulerSampler(dataset=train_dataset,batch_size=args.batch_size),
        sampler=train_sampler,
        collate_fn=collate_fn
    )
    
    val_dataset_list = []
   
    #test dataset
    if 'sarc_few_test' in args.dataset:
        val_dataset_list.append(SARCDataset(
            args.dataset['sarc_few_test'],data_id='sarc',use_text=True,use_image=False,use_audio=False))
    if 'emotion_few_test' in args.dataset:
        val_dataset_list.append(EMOJIDataset(
            args.dataset['emotion_few_test'],data_id='shot'))
    if 'emoji_few_test' in args.dataset:
        val_dataset_list.append(EMOJIDataset(
            args.dataset['emoji_few_test'],data_id='emoji'))
    val_dataset=ConcatDataset(val_dataset_list)
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=args.gpu_num,
        rank=rank,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn
    )
    start = datetime.now()
    # ========================== training ============================
    logger.info('Start training', pad=True)
    
    best_val_loss = float("inf")
    early_count=0
    while epoch < args.epochs:
        logger.info('Epoch {}'.format(epoch + 1), pad=True)
        
        fine_tune(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            #grad_optimizer=grad_optimizer,
            args=args,
            device=device,
            logger=logger,
            log_interval=1,
            tb_writer=tb_writer,
            tb_interval=1,
            scaler=scaler
        )
        
            
        
        logger.info('Validating Epoch {}'.format(epoch + 1), pad=True)
        # save memory and faster with no_grad()
        with torch.no_grad():
            if epoch>25:
                
                stop,val_loss=val(
                    epoch=epoch,
                    model=model,
                    tokenizer=tokenizer,
                    val_loader=val_loader,
                    device=device,
                    is_pretrain_stage1=False,
                    args=args,
                    logger=logger,
                    log_interval=1,
                    tb_writer=tb_writer
                )
                if rank == 0: 
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_count=0
                        if args.cpu:
                            model.save_pretrained("few_shot_best")
                        else:
                            model.module.save_pretrained("few_shot_best")
                    else:
                        early_count+=1
        epoch += 1
        if early_count>3:
            break
    logger.info("Training complete in: " + str(datetime.now() - start), pad=True)
    if args.checkpoint:
        model = MultiModalBartForConditionalGeneration.from_pretrained(
            "few_shot_best",
            config=bart_config,
            error_on_mismatch=False
        )
    else:
        model = MultiModalBartForConditionalGeneration(bart_config)

    model.to(device)
    if not args.cpu:
        torch.cuda.set_device(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    with torch.no_grad():
        generate_text(
            model=model,
            tokenizer=tokenizer,
            gen_loader=val_loader,
            device=device,
            args=args,
            is_pretrain_stage1=False,
            logger=logger,
            log_interval=1  
        )
    
    if not args.cpu:
        cleanup_process()


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
    parser.add_argument('--no_mrm', dest='mrm_enabled', action='store_true',default=True,
                        help='do not use masked region modelling')
    parser.add_argument('--epochs', default=70, type=int,
                        help='number of training epoch')
    parser.add_argument('--lr', default=5e-5, type=float,
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
   

    # dropout
    parser.add_argument('--dropout', default=0.3, type=float,
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
    parser.add_argument('--cpu', default=False,action='store_false',
                        help='if only use cpu to run the model')
    parser.add_argument('--amp', action='store_true',
                        help='whether or not to use amp')
    parser.add_argument('--master_port', type=str, default='12777',
                        help='master port for DDP')
    parser.add_argument('--batch_size', type=int, default=128,
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
