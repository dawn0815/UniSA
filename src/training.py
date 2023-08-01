from datetime import datetime
import re
import numpy as np
from torch.cuda.amp import autocast
import torch
from src.utils import TaskType
import numpy as np

from sklearn.metrics import confusion_matrix,f1_score
from sklearn.metrics import accuracy_score


def pretrain(
        epoch,
        model,
        train_loader,
        optimizer,
        device,
        args,
        is_pretrain_stage1,
        logger=None,
        callback=None,
        log_interval=1,
        tb_writer=None,
        tb_interval=1,
        scaler=None
):
    total_step = len(train_loader)
    model.train()
    total_loss = 0
    total_len = 0
    start_time = datetime.now()

    for i, batch in enumerate(train_loader):
        # Forward pass
       
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
        total_len += data_len
        # Backward and optimize
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        else:
            loss.backward()
            optimizer.step()

        if logger is not None and i % log_interval == 0:
            logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, ETA: {}'.format(
                epoch + 1,
                args.epochs,
                i + 1,
                total_step,
                loss.item(),
                str((total_step - (i + 1)) / (i + 1) * (datetime.now() - start_time))
            ))
    total_loss/=total_len
    if tb_writer is not None:
        tb_writer.add_scalars('loss/epoch', {'train': total_loss / total_step}, epoch + 1)


def fine_tune(
        epoch,
        model,
        train_loader,
        optimizer,
        #grad_optimizer,
        device,
        args,
        logger=None,
        callback=None,
        log_interval=1,
        tb_writer=None,
        tb_interval=1,
        scaler=None
):
    total_step = len(train_loader)
    model.train()
    total_loss = 0
    
    start_time = datetime.now()
    total_len=0
    for i, batch in enumerate(train_loader):
        # Forward pass
        
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
            
            loss_list=[outputs[0]['loss1'],outputs[0]['loss2'],outputs[0]['loss3'],outputs[0]['loss4']]
            data_len=len(batch['labels'])
            
            
        total_loss += loss.item()
        total_len+=data_len

        # Backward and optimize
        
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # grad_optimizer.backward(loss_list)
            # grad_optimizer.step()
            
        else:
            loss.backward()
            optimizer.step()

        if logger is not None and i % log_interval == 0:
            logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, ETA: {}'.format(
                epoch + 1,
                args.epochs,
                i + 1,
                total_step,
                loss.item(),
                str((total_step - (i + 1)) / (i + 1) * (datetime.now() - start_time))
            ))

        if tb_writer is not None and i % tb_interval == 0:
            step = epoch * total_step + i + 1
            tb_writer.add_scalars('loss/step', {'loss': loss.item()}, step)

        if callback is not None:
            callback(
                step=i,
                epoch=epoch,
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                args=args,
                logger=logger
            )
    total_loss/=total_len
    
    if logger is not None:
        logger.info('Training loss', pad=True)
        logger.info('Epoch: {}, train loss: {} '.format(epoch + 1,   total_loss))
        logger.line()
    if tb_writer is not None:
        tb_writer.add_scalars('loss/epoch', {'train': total_loss}, epoch + 1)
    
    
