import os
import time

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
from dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from losses import DiceLoss
from losses import FocalLoss
from model import Model
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou


# Assuming necessary imports for your project
# from your_project import FocalLoss, DiceLoss, AverageMeter, calc_iou, validate, Model, load_datasets, configure_opt

# Your configuration
config = {
    "num_devices": 1,
    "batch_size": 1,
    "num_workers": 4,
    "num_epochs": 10,
    "eval_interval": 2,
    "out_dir": "/home/mdjaberal.nahian/output_wheat_head_small",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "/home/mdjaberal.nahian/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/home/mdjaberal.nahian/wheat-1/train",
            "annotation_file": "/home/mdjaberal.nahian/wheat-1/train_annotations.coco.json"
        },
        "val": {
            "root_dir": "/home/mdjaberal.nahian/wheat-1/valid",
            "annotation_file": "/home/mdjaberal.nahian/wheat-1/valid_annotations.coco.json"
        }
    }
}

cfg = Box(config)

def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()

    for epoch in range(1, cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()
        validated = False

        for iter, data in enumerate(train_dataloader):
            if epoch > 1 and epoch % cfg.eval_interval == 0 and not validated:
                validate(fabric, model, val_dataloader, epoch)
                validated = True

            data_time.update(time.time() - end)
            images, bboxes, gt_masks = data
            batch_size = images.size(0)
            pred_masks, iou_predictions = model(images, bboxes)
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            loss_total = 20. * loss_focal + loss_dice + loss_iou
            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')
            # Log metrics to TensorBoard
            step = epoch * len(train_dataloader) + iter
            fabric.log('train/focal_loss', focal_losses.val, step=step)
            fabric.log('train/dice_loss', dice_losses.val, step=step)
            fabric.log('train/iou_loss', iou_losses.val, step=step)
            fabric.log('train/total_loss', total_losses.val, step=step)

        # Log average losses at the end of each epoch
        fabric.log('train/epoch_focal_loss', focal_losses.avg, step=epoch)
        fabric.log('train/epoch_dice_loss', dice_losses.avg, step=epoch)
        fabric.log('train/epoch_iou_loss', iou_losses.avg, step=epoch)
        fabric.log('train/epoch_total_loss', total_losses.avg, step=epoch)
def validate(fabric, model, val_dataloader, epoch: int):
    model.eval()
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    val_focal_losses = AverageMeter()
    val_dice_losses = AverageMeter()
    val_iou_losses = AverageMeter()
    val_total_losses = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            batch_size = images.size(0)
            pred_masks, iou_predictions = model(images, bboxes)
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            loss_total = 20. * loss_focal + loss_dice + loss_iou

            val_focal_losses.update(loss_focal.item(), batch_size)
            val_dice_losses.update(loss_dice.item(), batch_size)
            val_iou_losses.update(loss_iou.item(), batch_size)
            val_total_losses.update(loss_total.item(), batch_size)

    fabric.logger.experiment.add_scalar('val/focal_loss', val_focal_losses.avg, epoch)
    fabric.logger.experiment.add_scalar('val/dice_loss', val_dice_losses.avg, epoch)
    fabric.logger.experiment.add_scalar('val/iou_loss', val_iou_losses.avg, epoch)
    fabric.logger.experiment.add_scalar('val/total_loss', val_total_losses.avg, epoch)

    fabric.print(f'Validation - Epoch: [{epoch}]'
                 f' | Focal Loss [{val_focal_losses.avg:.4f}]'
                 f' | Dice Loss [{val_dice_losses.avg:.4f}]'
                 f' | IoU Loss [{val_iou_losses.avg:.4f}]'
                 f' | Total Loss [{val_total_losses.avg:.4f}]')

def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler
def main(cfg: Box) -> None:
    fabric = L.Fabric(
        accelerator="auto",
        devices=cfg.num_devices,
        strategy="auto",
        loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")]
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    train_data, val_data = load_datasets(cfg, model.model.image_encoder.img_size)
    train_dataloader = fabric._setup_dataloader(train_data)
    val_dataloader = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    train_sam(cfg, fabric, model, optimizer, scheduler, train_dataloader, val_dataloader)
    validate(fabric, model, val_dataloader, epoch=0)



if __name__ == "__main__":
    main(cfg)
