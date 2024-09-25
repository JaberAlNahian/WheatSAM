import os
import sys
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

torch.set_float32_matmul_precision('high')

# Path to save the log file
log_file_path = os.path.join(cfg.out_dir, "log.txt")

# Redirect stdout to the log file
sys.stdout = open(log_file_path, "w")

def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, epoch: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            num_images = images.size(0)
            pred_masks, _ = model(images, bboxes)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            fabric.print(
                f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    fabric.print(f"Saving checkpoint to {cfg.out_dir}")
    state_dict = model.model.state_dict()
    if fabric.global_rank == 0:
        # Save the last epoch trained file
        torch.save(state_dict, os.path.join(cfg.out_dir, "last_epoch.pth"))
        # Check if the current epoch has the best F1 score
        if epoch == cfg.num_epochs - 1 or f1_scores.avg >= fabric.best_f1_score:
            # Save the best epoch trained file
            torch.save(state_dict, os.path.join(cfg.out_dir, "best_epoch.pth"))
            # Update the best F1 score
            fabric.best_f1_score = f1_scores.avg
    model.train()


# def train_sam(
#     cfg: Box,
#     fabric: L.Fabric,
#     model: Model,
#     optimizer: _FabricOptimizer,
#     scheduler: _FabricOptimizer,
#     train_dataloader: DataLoader,
#     val_dataloader: DataLoader,
# ):
#     """The SAM training loop."""

#     focal_loss = FocalLoss()
#     dice_loss = DiceLoss()

#     for epoch in range(1, cfg.num_epochs):
#         batch_time = AverageMeter()
#         data_time = AverageMeter()
#         focal_losses = AverageMeter()
#         dice_losses = AverageMeter()
#         iou_losses = AverageMeter()
#         total_losses = AverageMeter()
#         end = time.time()
#         validated = False

#         for iter, data in enumerate(train_dataloader):
#             if epoch > 1 and epoch % cfg.eval_interval == 0 and not validated:
#                 validate(fabric, model, val_dataloader, epoch)
#                 validated = True

#             data_time.update(time.time() - end)
#             images, bboxes, gt_masks = data
#             batch_size = images.size(0)
#             pred_masks, iou_predictions = model(images, bboxes)
#             num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
#             loss_focal = torch.tensor(0., device=fabric.device)
#             loss_dice = torch.tensor(0., device=fabric.device)
#             loss_iou = torch.tensor(0., device=fabric.device)
#             for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
#                 batch_iou = calc_iou(pred_mask, gt_mask)
#                 loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
#                 loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
#                 loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

#             loss_total = 20. * loss_focal + loss_dice + loss_iou
#             optimizer.zero_grad()
#             fabric.backward(loss_total)
#             optimizer.step()
#             scheduler.step()
#             batch_time.update(time.time() - end)
#             end = time.time()

#             focal_losses.update(loss_focal.item(), batch_size)
#             dice_losses.update(loss_dice.item(), batch_size)
#             iou_losses.update(loss_iou.item(), batch_size)
#             total_losses.update(loss_total.item(), batch_size)

#             fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
#                          f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
#                          f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
#                          f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
#                          f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
#                          f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
#                          f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

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

        # Log after each epoch
        fabric.print(f'Epoch [{epoch}] completed. Logging to file...')
        with open(log_file_path, 'a') as log_file:
            log_file.write(f'Epoch: [{epoch}]'
                           f' | Focal Loss: [{focal_losses.avg:.4f}]'
                           f' | Dice Loss: [{dice_losses.avg:.4f}]'
                           f' | IoU Loss: [{iou_losses.avg:.4f}]'
                           f' | Total Loss: [{total_losses.avg:.4f}]\n')


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
    fabric = L.Fabric(accelerator="auto",
                      devices=cfg.num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    train_data, val_data = load_datasets(cfg, model.model.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    fabric.best_f1_score = 0.0  # Initialize best F1 score

    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    validate(fabric, model, val_data, epoch=0)
    sys.stdout.close()

if __name__ == "__main__":
    main(cfg)

