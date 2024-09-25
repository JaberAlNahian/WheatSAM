# import os
# import cv2
# import torch
# from box import Box
# from dataset import COCODataset
# from model import Model
# from torchvision.utils import draw_bounding_boxes
# from torchvision.utils import draw_segmentation_masks
# from tqdm import tqdm
# from pycocotools import mask as coco_mask
# import numpy as np

# class AverageMeter:
#     """Computes and stores the average and current value."""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
#     pred_mask = (pred_mask >= 0.5).float()
#     intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
#     union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
#     epsilon = 1e-7
#     batch_iou = intersection / (union + epsilon)

#     batch_iou = batch_iou.unsqueeze(1)
#     return batch_iou

# def calc_dice_loss(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
#     pred_mask = (pred_mask >= 0.5).float()
#     intersection = torch.sum(pred_mask * gt_mask, dim=(1, 2))
#     cardinality = torch.sum(pred_mask + gt_mask, dim=(1, 2))
#     epsilon = 1e-7
#     dice_loss = 1 - (2. * intersection / (cardinality + epsilon))
#     return dice_loss

# def draw_image(image, masks, boxes, labels, alpha=0.4):
#     image = torch.from_numpy(image).permute(2, 0, 1)
#     if boxes is not None:
#         image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
#     if masks is not None:
#         image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
#     return image.numpy().transpose(1, 2, 0)

# def visualize(cfg: Box):
#     model = Model(cfg)
#     model.setup()
#     model.eval()
#     model.cuda()
#     gt_dataset = COCODataset(root_dir=cfg.dataset.val.root_dir,
#                              annotation_file=cfg.dataset.val.annotation_file,
#                              transform=None)
#     bbox_dataset = COCODataset(root_dir=cfg.dataset.val.root_dir,
#                                annotation_file=cfg.dataset.bbox.annotation_file,
#                                transform=None)
#     predictor = model.get_predictor()
#     os.makedirs(cfg.out_dir, exist_ok=True)

#     dice_meter = AverageMeter()
#     iou_meter = AverageMeter()

#     for image_id in tqdm(gt_dataset.image_ids):
#         image_info = gt_dataset.coco.loadImgs(image_id)[0]
#         image_path = os.path.join(gt_dataset.root_dir, image_info['file_name'])
#         image_output_path = os.path.join(cfg.out_dir, image_info['file_name'])
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Load ground truth masks
#         gt_ann_ids = gt_dataset.coco.getAnnIds(imgIds=image_id)
#         gt_anns = gt_dataset.coco.loadAnns(gt_ann_ids)
#         gt_masks = []
#         for ann in gt_anns:
#             gt_mask = gt_dataset.coco.annToMask(ann)
#             gt_masks.append(gt_mask)
#         gt_masks = torch.tensor(np.stack(gt_masks), dtype=torch.float32, device=model.model.device)

#         # Load bounding boxes
#         bbox_ann_ids = bbox_dataset.coco.getAnnIds(imgIds=image_id)
#         bbox_anns = bbox_dataset.coco.loadAnns(bbox_ann_ids)
#         bboxes = []
#         for ann in bbox_anns:
#             x, y, w, h = ann['bbox']
#             bboxes.append([x, y, x + w, y + h])
#         bboxes = torch.as_tensor(bboxes, device=model.model.device)
#         transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])

#         predictor.set_image(image)
#         masks, _, _ = predictor.predict_torch(
#             point_coords=None,
#             point_labels=None,
#             boxes=transformed_boxes,
#             multimask_output=False,
#         )

#         # Calculate Dice Score and IoU
#         batch_iou = calc_iou(masks.squeeze(1), gt_masks)
#         iou_meter.update(batch_iou.mean().item(), len(batch_iou))
#         dice_loss = calc_dice_loss(masks.squeeze(1), gt_masks)
#         dice_meter.update((2 * batch_iou / (1 + batch_iou)).mean().item(), len(batch_iou))

#         # image_output = draw_image(image, masks.squeeze(1), boxes=None, labels=None)
#         # print("saving to : ", image_output_path)
#         # cv2.imwrite(image_output_path, image_output)

#     print(f"Average Dice Score: {dice_meter.avg}")
#     print(f"Average IoU: {iou_meter.avg}")

# if __name__ == "__main__":
#     from config import cfg
#     visualize(cfg)
"""
Update---1
"""
# import os
# import cv2
# import torch
# from box import Box
# from dataset import COCODataset
# from model import Model
# from torchvision.utils import draw_bounding_boxes
# from torchvision.utils import draw_segmentation_masks
# from tqdm import tqdm
# from pycocotools import mask as coco_mask
# import numpy as np

# class AverageMeter:
#     """Computes and stores the average and current value."""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
#     pred_mask = (pred_mask >= 0.5).float()
#     intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
#     union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
#     epsilon = 1e-7
#     batch_iou = intersection / (union + epsilon)

#     batch_iou = batch_iou.unsqueeze(1)
#     return batch_iou

# def calc_dice_loss(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
#     pred_mask = (pred_mask >= 0.5).float()
#     intersection = torch.sum(pred_mask * gt_mask, dim=(1, 2))
#     cardinality = torch.sum(pred_mask + gt_mask, dim=(1, 2))
#     epsilon = 1e-7
#     dice_loss = 1 - (2. * intersection / (cardinality + epsilon))
#     return dice_loss

# def draw_image(image, masks, boxes, labels, alpha=0.4):
#     image = torch.from_numpy(image).permute(2, 0, 1)
#     if boxes is not None:
#         image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
#     if masks is not None:
#         image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
#     return image.numpy().transpose(1, 2, 0)

# def visualize(cfg: Box):
#     model = Model(cfg)
#     model.setup()
#     model.eval()
#     model.cuda()
#     gt_dataset = COCODataset(root_dir=cfg.dataset.val.root_dir,
#                              annotation_file=cfg.dataset.val.annotation_file,
#                              transform=None)
#     bbox_dataset = COCODataset(root_dir=cfg.dataset.val.root_dir,
#                                annotation_file=cfg.dataset.bbox.annotation_file,
#                                transform=None)
#     predictor = model.get_predictor()
#     os.makedirs(cfg.out_dir, exist_ok=True)

#     dice_meter = AverageMeter()
#     iou_meter = AverageMeter()

#     for image_id in tqdm(gt_dataset.image_ids):
#         image_info = gt_dataset.coco.loadImgs(image_id)[0]
#         image_path = os.path.join(gt_dataset.root_dir, image_info['file_name'])
#         image_output_path = os.path.join(cfg.out_dir, image_info['file_name'])
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Load ground truth masks
#         gt_ann_ids = gt_dataset.coco.getAnnIds(imgIds=image_id)
#         gt_anns = gt_dataset.coco.loadAnns(gt_ann_ids)
#         gt_masks = []
#         for ann in gt_anns:
#             gt_mask = gt_dataset.coco.annToMask(ann)
#             gt_masks.append(gt_mask)
#         gt_masks = torch.tensor(np.stack(gt_masks), dtype=torch.float32, device=model.model.device)

#         # Load bounding boxes
#         bbox_ann_ids = bbox_dataset.coco.getAnnIds(imgIds=image_id)
#         bbox_anns = bbox_dataset.coco.loadAnns(bbox_ann_ids)
#         bboxes = []
#         for ann in bbox_anns:
#             x, y, w, h = ann['bbox']
#             bboxes.append([x, y, x + w, y + h])
#         bboxes = torch.as_tensor(bboxes, device=model.model.device)
#         transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])

#         predictor.set_image(image)
#         masks, _, _ = predictor.predict_torch(
#             point_coords=None,
#             point_labels=None,
#             boxes=transformed_boxes,
#             multimask_output=False,
#         )

#         # Check the number of masks and ground truth masks
#         if masks.size(0) != gt_masks.size(0):
#             print(f"Warning: Mismatch in number of masks for image {image_id}. Predicted: {masks.size(0)}, GT: {gt_masks.size(0)}")
#             # Optionally, handle this case (e.g., by skipping this image or taking some other action)
#             continue

#         # Calculate Dice Score and IoU
#         batch_iou = calc_iou(masks.squeeze(1), gt_masks)
#         iou_meter.update(batch_iou.mean().item(), len(batch_iou))
#         dice_loss = calc_dice_loss(masks.squeeze(1), gt_masks)
#         dice_meter.update((2 * batch_iou / (1 + batch_iou)).mean().item(), len(batch_iou))

#         image_output = draw_image(image, masks.squeeze(1), boxes=None, labels=None)
#         print("saving to : ", image_output_path)
#         cv2.imwrite(image_output_path, image_output)

#     print(f"Average Dice Score: {dice_meter.avg}")
#     print(f"Average IoU: {iou_meter.avg}")

# if __name__ == "__main__":
#     from config import cfg
#     visualize(cfg)

"""
Update---2
"""

# import os
# import cv2
# import torch
# from box import Box
# from dataset import COCODataset
# from model import Model
# from torchvision.utils import draw_bounding_boxes
# from torchvision.utils import draw_segmentation_masks
# from tqdm import tqdm
# from pycocotools import mask as coco_mask
# import numpy as np

# class AverageMeter:
#     """Computes and stores the average and current value."""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
#     pred_mask = (pred_mask >= 0.5).float()
#     intersection = torch.sum(torch.mul(pred_mask, gt_mask))
#     union = torch.sum(pred_mask) + torch.sum(gt_mask) - intersection
#     epsilon = 1e-7
#     iou = intersection / (union + epsilon)
#     return iou

# def calc_dice_loss(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
#     pred_mask = (pred_mask >= 0.5).float()
#     intersection = torch.sum(pred_mask * gt_mask)
#     cardinality = torch.sum(pred_mask + gt_mask)
#     epsilon = 1e-7
#     dice_loss = 1 - (2. * intersection / (cardinality + epsilon))
#     return dice_loss

# def draw_image(image, masks, boxes, labels, alpha=0.4):
#     image = torch.from_numpy(image).permute(2, 0, 1)
#     if boxes is not None:
#         image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
#     if masks is not None:
#         image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
#     return image.numpy().transpose(1, 2, 0)

# def visualize(cfg: Box):
#     model = Model(cfg)
#     model.setup()
#     model.eval()
#     model.cuda()
#     gt_dataset = COCODataset(root_dir=cfg.dataset.val.root_dir,
#                              annotation_file=cfg.dataset.val.annotation_file,
#                              transform=None)
#     bbox_dataset = COCODataset(root_dir=cfg.dataset.val.root_dir,
#                                annotation_file=cfg.dataset.bbox.annotation_file,
#                                transform=None)
#     predictor = model.get_predictor()
#     os.makedirs(cfg.out_dir, exist_ok=True)

#     dice_meter = AverageMeter()
#     iou_meter = AverageMeter()

#     for image_id in tqdm(gt_dataset.image_ids):
#         image_info = gt_dataset.coco.loadImgs(image_id)[0]
#         image_path = os.path.join(gt_dataset.root_dir, image_info['file_name'])
#         image_output_path = os.path.join(cfg.out_dir, image_info['file_name'])
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Load ground truth masks
#         gt_ann_ids = gt_dataset.coco.getAnnIds(imgIds=image_id)
#         gt_anns = gt_dataset.coco.loadAnns(gt_ann_ids)
#         gt_masks = []
#         for ann in gt_anns:
#             gt_mask = gt_dataset.coco.annToMask(ann)
#             gt_masks.append(gt_mask)
#         gt_masks = torch.tensor(np.stack(gt_masks), dtype=torch.float32, device=model.model.device)
#         combined_gt_mask = torch.sum(gt_masks, dim=0).clamp(0, 1)

#         # Load bounding boxes
#         bbox_ann_ids = bbox_dataset.coco.getAnnIds(imgIds=image_id)
#         bbox_anns = bbox_dataset.coco.loadAnns(bbox_ann_ids)
#         bboxes = []
#         for ann in bbox_anns:
#             x, y, w, h = ann['bbox']
#             bboxes.append([x, y, x + w, y + h])
#         bboxes = torch.as_tensor(bboxes, device=model.model.device)
#         transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])

#         predictor.set_image(image)
#         masks, _, _ = predictor.predict_torch(
#             point_coords=None,
#             point_labels=None,
#             boxes=transformed_boxes,
#             multimask_output=False,
#         )
#         combined_pred_mask = torch.sum(masks.squeeze(1), dim=0).clamp(0, 1)

#         # Calculate Dice Score and IoU
#         iou = calc_iou(combined_pred_mask, combined_gt_mask)
#         iou_meter.update(iou.item())
#         dice_loss = calc_dice_loss(combined_pred_mask, combined_gt_mask)
#         dice_meter.update(1 - dice_loss.item())

#         # Optionally, visualize and save the results
#         # image_output = draw_image(image, combined_pred_mask.unsqueeze(0), boxes=None, labels=None)
#         # print("saving to : ", image_output_path)
#         # cv2.imwrite(image_output_path, image_output)

#     print(f"Average Dice Score: {dice_meter.avg}")
#     print(f"Average IoU: {iou_meter.avg}")

# if __name__ == "__main__":
#     from config import cfg
#     visualize(cfg)

"""
Update 3
"""
# import os
# import cv2
# import torch
# from box import Box
# from dataset import COCODataset
# from model import Model
# from torchvision.utils import draw_bounding_boxes
# from torchvision.utils import draw_segmentation_masks
# from tqdm import tqdm
# from pycocotools import mask as coco_mask
# import numpy as np

# class AverageMeter:
#     """Computes and stores the average and current value."""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
#     pred_mask = (pred_mask >= 0.5).float()
#     intersection = torch.sum(torch.mul(pred_mask, gt_mask))
#     union = torch.sum(pred_mask) + torch.sum(gt_mask) - intersection
#     epsilon = 1e-7
#     iou = intersection / (union + epsilon)
#     return iou

# def calc_dice_loss(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
#     pred_mask = (pred_mask >= 0.5).float()
#     intersection = torch.sum(pred_mask * gt_mask)
#     cardinality = torch.sum(pred_mask + gt_mask)
#     epsilon = 1e-7
#     dice_loss = 1 - (2. * intersection / (cardinality + epsilon))
#     return dice_loss

# def draw_image(image, masks, boxes, labels, alpha=0.4):
#     image = torch.from_numpy(image).permute(2, 0, 1)
#     if boxes is not None:
#         image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
#     if masks is not None:
#         masks = masks.bool()  # Ensure masks are of dtype bool
#         image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
#     return image.numpy().transpose(1, 2, 0)

# def visualize(cfg: Box):
#     model = Model(cfg)
#     model.setup()
#     model.eval()
#     model.cuda()
#     gt_dataset = COCODataset(root_dir=cfg.dataset.val.root_dir,
#                              annotation_file=cfg.dataset.val.annotation_file,
#                              transform=None)
#     bbox_dataset = COCODataset(root_dir=cfg.dataset.val.root_dir,
#                                annotation_file=cfg.dataset.bbox.annotation_file,
#                                transform=None)
#     predictor = model.get_predictor()
#     os.makedirs(cfg.out_dir, exist_ok=True)

#     dice_meter = AverageMeter()
#     iou_meter = AverageMeter()

#     for image_id in tqdm(gt_dataset.image_ids):
#         image_info = gt_dataset.coco.loadImgs(image_id)[0]
#         image_path = os.path.join(gt_dataset.root_dir, image_info['file_name'])
#         image_output_path = os.path.join(cfg.out_dir, image_info['file_name'])
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Load ground truth masks
#         gt_ann_ids = gt_dataset.coco.getAnnIds(imgIds=image_id)
#         gt_anns = gt_dataset.coco.loadAnns(gt_ann_ids)
#         gt_masks = []
#         for ann in gt_anns:
#             gt_mask = gt_dataset.coco.annToMask(ann)
#             gt_masks.append(gt_mask)
#         gt_masks = torch.tensor(np.stack(gt_masks), dtype=torch.float32, device=model.model.device)
#         combined_gt_mask = torch.sum(gt_masks, dim=0).clamp(0, 1)

#         # Load bounding boxes
#         bbox_ann_ids = bbox_dataset.coco.getAnnIds(imgIds=image_id)
#         bbox_anns = bbox_dataset.coco.loadAnns(bbox_ann_ids)
#         bboxes = []
#         for ann in bbox_anns:
#             x, y, w, h = ann['bbox']
#             bboxes.append([x, y, x + w, y + h])
#         bboxes = torch.as_tensor(bboxes, device=model.model.device)
#         transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])

#         predictor.set_image(image)
#         masks, _, _ = predictor.predict_torch(
#             point_coords=None,
#             point_labels=None,
#             boxes=transformed_boxes,
#             multimask_output=False,
#         )
#         combined_pred_mask = torch.sum(masks.squeeze(1), dim=0).clamp(0, 1)

#         # Calculate Dice Score and IoU
#         iou = calc_iou(combined_pred_mask, combined_gt_mask)
#         iou_meter.update(iou.item())
#         dice_loss = calc_dice_loss(combined_pred_mask, combined_gt_mask)
#         dice_meter.update(1 - dice_loss.item())

#         # Optionally, visualize and save the results
#         image_output = draw_image(image, combined_pred_mask.unsqueeze(0), boxes=None, labels=None)
#         print("saving to : ", image_output_path)
#         cv2.imwrite(image_output_path, cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR))

#     print(f"Average Dice Score: {dice_meter.avg}")
#     print(f"Average IoU: {iou_meter.avg}")

# if __name__ == "__main__":
#     from config import cfg
#     visualize(cfg)


"""UPDATE 4"""
import os
import cv2
import torch
import logging
from box import Box
from dataset import COCODataset
from model import Model
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm
from pycocotools import mask as coco_mask
import numpy as np

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask))
    union = torch.sum(pred_mask) + torch.sum(gt_mask) - intersection
    epsilon = 1e-7
    iou = intersection / (union + epsilon)
    return iou

def calc_dice_loss(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(pred_mask * gt_mask)
    cardinality = torch.sum(pred_mask + gt_mask)
    epsilon = 1e-7
    dice_loss = 1 - (2. * intersection / (cardinality + epsilon))
    return dice_loss

def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if boxes is not None:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if masks is not None:
        masks = masks.bool()  # Ensure masks are of dtype bool
        image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)

def visualize(cfg: Box):
    # Ensure the output directory exists
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(cfg.out_dir, 'output.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

    model = Model(cfg)
    model.setup()
    model.eval()
    model.cuda()
    gt_dataset = COCODataset(root_dir=cfg.dataset.test.root_dir,
                             annotation_file=cfg.dataset.test.annotation_file,
                             transform=None)
    bbox_dataset = COCODataset(root_dir=cfg.dataset.test.root_dir,
                               annotation_file=cfg.dataset.test.bbox_annotation_file,
                               transform=None)
    predictor = model.get_predictor()
    os.makedirs(cfg.out_dir, exist_ok=True)

    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    for image_id in tqdm(gt_dataset.image_ids):
        image_info = gt_dataset.coco.loadImgs(image_id)[0]
        image_path = os.path.join(gt_dataset.root_dir, image_info['file_name'])
        image_output_path = os.path.join(cfg.out_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load ground truth masks
        gt_ann_ids = gt_dataset.coco.getAnnIds(imgIds=image_id)
        gt_anns = gt_dataset.coco.loadAnns(gt_ann_ids)
        gt_masks = []
        for ann in gt_anns:
            gt_mask = gt_dataset.coco.annToMask(ann)
            gt_masks.append(gt_mask)
        gt_masks = torch.tensor(np.stack(gt_masks), dtype=torch.float32, device=model.model.device)
        combined_gt_mask = torch.sum(gt_masks, dim=0).clamp(0, 1)

        # Load bounding boxes
        bbox_ann_ids = bbox_dataset.coco.getAnnIds(imgIds=image_id)
        bbox_anns = bbox_dataset.coco.loadAnns(bbox_ann_ids)
        bboxes = []
        for ann in bbox_anns:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x + w, y + h])
        bboxes = torch.as_tensor(bboxes, device=model.model.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])

        predictor.set_image(image)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        combined_pred_mask = torch.sum(masks.squeeze(1), dim=0).clamp(0, 1)

        # Calculate Dice Score and IoU
        iou = calc_iou(combined_pred_mask, combined_gt_mask)
        iou_meter.update(iou.item())
        dice_loss = calc_dice_loss(combined_pred_mask, combined_gt_mask)
        dice_meter.update(1 - dice_loss.item())

        # Optionally, visualize and save the results
        # image_output = draw_image(image, combined_pred_mask.unsqueeze(0), boxes=None, labels=None)
        # logging.info(f"saving to: {image_output_path}")
        # cv2.imwrite(image_output_path, cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR))

    logging.info(f"Average Dice Score: {dice_meter.avg}")
    logging.info(f"Average IoU: {iou_meter.avg}")

if __name__ == "__main__":
    from config_Test import cfg
    visualize(cfg)
