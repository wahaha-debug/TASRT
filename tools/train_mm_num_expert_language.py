import argparse
import os
import sys
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
sys.path.append(project_root)

import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.datasets import *
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from val_mm import evaluate

from sam2.build_sam import build_sam2
from sam2.sam_lora_moe2_DSCSE_language import LoRA_Sam

import clip

try:
    import clip as _clip
    _HAS_CLIP = True
except Exception:
    _clip = None
    _HAS_CLIP = False

def main(cfg, gpu, save_dir, logger=None):

    if logger is None:
        logger = get_logger(Path(save_dir) / 'train.log')
    start = time.time()  
    best_mIoU = 0.0  
    best_epoch = 0  
    num_workers = 8 
    device = torch.device(cfg['DEVICE']) 
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL'] 
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL'] 
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']  
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR'] 
    resume_path = cfg['MODEL']['RESUME'] 
    gpus = int(os.environ.get('WORLD_SIZE', 1)) 

    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])

    trainset = eval(dataset_cfg['NAME'])(
        dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'],
        use_text=True,
        rgb_text_json=dataset_cfg.get('RGB_TEXT_JSON'),
        thermal_text_json=dataset_cfg.get('THERMAL_TEXT_JSON')
    )
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'])

    class_names = trainset.CLASSES  
    checkpoint = model_cfg['PRETRAINED']
    model_cf = model_cfg['MODEL_CONFIG']

    sam2 = build_sam2(model_cf, checkpoint)

    num_experts = train_cfg.get('NUM_EXPERTS', 1)
    top_k = train_cfg.get('TOP_K', min(2, num_experts))
    clip_text_dim = 512

    model = LoRA_Sam(sam2, r=train_cfg['RANK'], num_classes=trainset.n_classes, num_experts=num_experts, top_k=top_k, text_dim=clip_text_dim)

    clip_tokenizer = None
    clip_text_model = None


    def count_parameters(model):
        total_parameters = 0
        trainable_parameters = 0
        for name, param in model.named_parameters():
            total_parameters += param.numel()
            if param.requires_grad:
                trainable_parameters += param.numel()
        return total_parameters, trainable_parameters

    total_parameters, trainable_parameters = count_parameters(model)
    print('Total number of parameters: %d' % total_parameters)
    print('Total number of trainable parameters: %d' % trainable_parameters)
    print('Percentage of trainable parameters: %.2f%%' % (trainable_parameters / total_parameters * 100))

    resume_checkpoint = None  
    if os.path.isfile(resume_path):  
        resume_checkpoint = torch.load(resume_path, map_location=torch.device('cpu')) 
        msg = model.load_state_dict(resume_checkpoint['model_state_dict'])  
        logger.info(msg)  
    # else:
        # model.init_pretrained(model_cfg['PRETRAINED']) 

    model = model.to(device)  
    clip_text_model, _ = clip.load("ViT-B/32", device=device)
    clip_text_model.eval()
    clip_tokenizer = clip.tokenize

    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE'] // gpus  
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)  
    start_epoch = 0  
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr,
                              optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(
        sched_cfg['NAME'], optimizer, int((epochs + 1) * iters_per_epoch),
        sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO']
    )

    lr_min = optim_cfg.get('LR_MIN', 1e-6)

    if train_cfg['DDP']:  

        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = None

        model = DDP(model, device_ids=[gpu], output_device=0, find_unused_parameters=True)
    else:
        sampler = RandomSampler(trainset) 
        sampler_val = None 

    if resume_checkpoint:
        start_epoch = resume_checkpoint['epoch'] - 1 
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])  
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])  
        loss = resume_checkpoint['loss'] 
        if 'best_miou' in resume_checkpoint:
            best_mIoU = resume_checkpoint['best_miou']  

    trainloader = DataLoader(
        trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers,
        drop_last=True, pin_memory=False, sampler=sampler
    )
    
    valloader = DataLoader(
        valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers,
        pin_memory=False, sampler=sampler_val
    )

    scaler = GradScaler(enabled=train_cfg['AMP'])  
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):

        writer = SummaryWriter(str(save_dir))  

        logger.info('================== model structure =====================')  
        logger.info(model) 
        logger.info('================== training config =====================') 
        logger.info(cfg) 

    switch_epoch_to_train2 = train_cfg.get('SWITCH_EPOCH_TO_TRAIN2', None)
    using_train2 = False

    def _load_split_list(root_dir, list_filename):
        src = os.path.join(root_dir, list_filename)
        files_out = []
        with open(src, 'r') as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                if ' ' in name:
                    name = name.split(' ')[0]
                files_out.append(name)
        return files_out

    for epoch in range(start_epoch, epochs):
        model.train()  

        if (switch_epoch_to_train2 is not None) and (not using_train2) and (epoch >= int(switch_epoch_to_train2)):
            try:
                new_files = _load_split_list(dataset_cfg['ROOT'], 'train2.txt')
            except Exception as e:
                raise RuntimeError(f"切换到 train2 失败：无法读取 {os.path.join(dataset_cfg['ROOT'], 'train2.txt')}，错误：{e}")
            trainset.files = new_files

            if train_cfg['DDP']:
                sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
            else:
                sampler = RandomSampler(trainset)
            trainloader = DataLoader(
                trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers,
                drop_last=True, pin_memory=False, sampler=sampler
            )
            iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE'] // gpus
            using_train2 = True
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                logger.info(f"Switched training list to train2.txt at epoch {epoch + 1}. Num images: {len(trainset)}")
                try:
                    writer.add_text('train/switch', f'switched to train2 at epoch {epoch + 1}', epoch)
                except Exception:
                    pass

        if train_cfg['DDP']:
            sampler.set_epoch(epoch)  

        train_loss = 0.0  

        lr = sum([g['lr'] for g in optimizer.param_groups]) / max(1, len(optimizer.param_groups))

        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch,
                    desc=f"Epoch: [{epoch + 1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                sample, lbl, text = batch
            else:
                sample, lbl = batch
                text = None 


            sample = [x.to(device) for x in sample]
            lbl = lbl.to(device)
            T = 8  
            B = lbl.shape[0]

            if text is not None:
                rgb_texts = [t for t in text['rgb_text']]
                th_texts = [t for t in text['thermal_text']]
                rgb_tok = clip_tokenizer(rgb_texts, truncate=True).to(device)
                th_tok = clip_tokenizer(th_texts, truncate=True).to(device)
                with torch.no_grad():
                    cls_rgb = clip_text_model.encode_text(rgb_tok)
                    cls_th  = clip_text_model.encode_text(th_tok)
                tokens_rgb = cls_rgb.unsqueeze(1).repeat(1, T, 1)
                tokens_th = cls_th.unsqueeze(1).repeat(1, T, 1)

            outputs = model(sample, multimask_output=True,
                            text_tokens_rgb=tokens_rgb, text_tokens_th=tokens_th)

            lambda_final = loss_cfg.get('LAMBDA_FINAL', 1.0)
            lambda_rgb = loss_cfg.get('LAMBDA_RGB', 0.4)
            lambda_thermal = loss_cfg.get('LAMBDA_THERMAL', 0.4)
            lambda_structure = loss_cfg.get('LAMBDA_STRUCTURE', 0.6)

            lambda_language = loss_cfg.get('LAMBDA_LANGUAGE', 0.00)
            loss_final = loss_fn(outputs['final'], lbl)
            loss_rgb = loss_fn(outputs['rgb'], lbl)
            loss_thermal = loss_fn(outputs['thermal'], lbl)
            loss_structure = loss_fn(outputs['structure'], lbl)
            loss_language = outputs.get('language_align_loss', torch.tensor(0.0, device=device))
            
            loss = (lambda_final * loss_final) + \
                   (lambda_rgb * loss_rgb) + \
                   (lambda_thermal * loss_thermal) + \
                   (lambda_structure * loss_structure) + \
                   (lambda_language * loss_language)

            scaler.scale(loss).backward()  
            scaler.step(optimizer)  
            scaler.update()  

            scheduler.step() 
            for g in optimizer.param_groups:
                if g.get('lr', 0.0) < lr_min:
                    g['lr'] = lr_min
            torch.cuda.synchronize()  
            lr = sum([g['lr'] for g in optimizer.param_groups]) / max(1, len(optimizer.param_groups))

            train_loss += loss.item()  
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                global_step = epoch * iters_per_epoch + iter
                writer.add_scalar('loss/step_final', float(loss_final.detach().item()), global_step)
                writer.add_scalar('loss/step_rgb', float(loss_rgb.detach().item()), global_step)
                writer.add_scalar('loss/step_thermal', float(loss_thermal.detach().item()), global_step)
                writer.add_scalar('loss/step_structure', float(loss_structure.detach().item()), global_step)
                writer.add_scalar('loss/step_language', float(loss_language.detach().item()), global_step)
                writer.add_scalar('loss/step_total', float(loss.detach().item()), global_step)
                writer.add_scalar('opt/lr', float(lr), global_step)

            pbar.set_description(f"Epoch: [{epoch + 1}/{epochs}] Iter: [{iter + 1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter + 1):.8f}")

        train_loss /= iter + 1

        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/loss_final', float(loss_final.detach().item()), epoch)
            writer.add_scalar('train/loss_rgb', float(loss_rgb.detach().item()), epoch)
            writer.add_scalar('train/loss_thermal', float(loss_thermal.detach().item()), epoch)
            writer.add_scalar('train/loss_structure', float(loss_structure.detach().item()), epoch)
            writer.add_scalar('train/loss_language', float(loss_language.detach().item()), epoch)
            writer.add_scalar('train/loss_total', float(loss.detach().item()), epoch)

        torch.cuda.empty_cache()

        if ((epoch + 1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch + 1) > train_cfg['EVAL_START']) or (epoch + 1) == epochs:
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                acc, macc, _, _, ious, miou = evaluate(model, valloader, device)
                writer.add_scalar('val/mIoU', miou, epoch)

                if miou > best_mIoU:
                    prev_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    prev_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    if os.path.isfile(prev_best): os.remove(prev_best)
                    if os.path.isfile(prev_best_ckp): os.remove(prev_best_ckp)

                    best_mIoU = miou
                    best_epoch = epoch + 1

                    cur_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    cur_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"

                    torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), cur_best)

                    torch.save({'epoch': best_epoch,
                                'model_state_dict': model.module.state_dict() if train_cfg[
                                    'DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_miou': best_mIoU,
                                }, cur_best_ckp)

                    logger.info(print_iou(epoch, ious, miou, acc, macc, class_names))

                logger.info(f"Current epoch:{epoch} mIoU: {miou} Best mIoU: {best_mIoU}")

        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.close()

        pbar.close()

        end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logger.info(tabulate(table, numalign='right'))
    return best_mIoU


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/PST900.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = None
    if cfg['TRAIN']['DDP']:
        gpu = setup_ddp()

    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    BATCH_SIZE = cfg['TRAIN']['BATCH_SIZE']
    NUM_EXPERTS = cfg['TRAIN']['NUM_EXPERTS']
    lambda_final = cfg['LOSS']['LAMBDA_FINAL']
    lambda_rgb = cfg['LOSS']['LAMBDA_RGB']
    lambda_thermal = cfg['LOSS']['LAMBDA_THERMAL']
    lambda_structure = cfg['LOSS']['LAMBDA_STRUCTURE']
    lambda_lb = cfg['LOSS']['LAMBDA_LB']
    lambda_language = cfg['LOSS']['LAMBDA_LANGUAGE']
    exp_name = '_'.join(
        [cfg['DATASET']['NAME'], str(BATCH_SIZE), str(NUM_EXPERTS),model, modals,'final'+str(lambda_final),'rgb'+str(lambda_rgb),'thermal'+str(lambda_thermal),'structure'+str(lambda_structure),'lb'+str(lambda_lb),'language'+str(lambda_language)])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    if os.path.isfile(cfg['MODEL']['RESUME']):
        save_dir =  Path(os.path.dirname(cfg['MODEL']['RESUME']))
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')
    main(cfg, gpu, save_dir, logger)
    if cfg['TRAIN']['DDP']:
        cleanup_ddp()