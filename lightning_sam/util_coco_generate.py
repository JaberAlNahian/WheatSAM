import os
import json
import cv2
import torch
from box import Box
from dataset import COCODataset
from model import Model
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from tqdm import tqdm
from pycocotools import mask as coco_mask
import numpy as np

class AverageMeter:
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
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)
    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou

def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if boxes is not None:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if masks is not None:
        image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)

def mask_to_polygons(mask):
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    polygons = coco_mask.toBbox(rle).tolist()
    return polygons

def create_coco_annotation(coco_data, output_dir, predicted_masks):
    annotation_id = 1
    output_annotations = {
        "images": [],
        "annotations": [],
        "categories": coco_data['categories'],
    }
    
    for image in coco_data['images']:
        image_id = image['id']
        image_filename = image['file_name']
        image_output_path = os.path.join(output_dir, image_filename)
        
        if os.path.exists(image_output_path):
            output_annotations["images"].append(image)
            for ann in coco_data['annotations']:
                if ann['image_id'] == image_id:
                    if image_id in predicted_masks:
                        segmentation = []
                        for mask in predicted_masks[image_id]:
                            polygons = mask_to_polygons(mask)
                            segmentation.append(polygons)
                    else:
                        segmentation = ann['segmentation']
                    
                    output_annotations["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": ann['category_id'],
                        "bbox": ann['bbox'],
                        "segmentation": segmentation,
                        "area": ann['area'],
                        "iscrowd": ann['iscrowd']
                    })
                    annotation_id += 1
    
    return output_annotations

def visualize(cfg: Box):
    model = Model(cfg)
    model.setup()
    model.eval()
    model.cuda()
    dataset = COCODataset(root_dir=cfg.dataset.test.root_dir, annotation_file=cfg.dataset.test.annotation_file, transform=None)
    predictor = model.get_predictor()
    os.makedirs(cfg.out_dir, exist_ok=True)

    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    predicted_masks = {}

    for image_id in tqdm(dataset.image_ids):
        image_info = dataset.coco.loadImgs(image_id)[0]
        image_path = os.path.join(dataset.root_dir, image_info['file_name'])
        image_output_path = os.path.join(cfg.out_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ann_ids = dataset.coco.getAnnIds(imgIds=image_id)
        anns = dataset.coco.loadAnns(ann_ids)
        gt_masks = []
        bboxes = []
        for ann in anns:
            if ann['segmentation']:  # Skip annotations with empty segmentation
                try:
                    gt_mask = dataset.coco.annToMask(ann)
                    gt_masks.append(gt_mask)
                    x, y, w, h = ann['bbox']
                    bboxes.append([x, y, x + w, y + h])
                except IndexError as e:
                    print(f"IndexError: {e}")
                    print(f"Problematic annotation ID: {ann['id']}")
                    print(f"Annotation data: {ann}")
                    continue
        if not gt_masks:
            print(f"No valid masks for image ID {image_id}, skipping.")
            continue
        
        gt_masks = torch.tensor(np.stack(gt_masks), dtype=torch.float32, device=model.model.device)
        bboxes = torch.as_tensor(bboxes, device=model.model.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])
        predictor.set_image(image)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        batch_iou = calc_iou(masks.squeeze(1), gt_masks)
        iou_meter.update(batch_iou.mean().item(), len(batch_iou))
        dice_meter.update((2 * batch_iou / (1 + batch_iou)).mean().item(), len(batch_iou))

        predicted_masks[image_id] = masks.squeeze(1).cpu().numpy()

        image_output = draw_image(image, masks.squeeze(1), boxes=None, labels=None)
        print("saving to : ", image_output_path)
        cv2.imwrite(image_output_path, cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR))

    print(f"Average Dice Score: {dice_meter.avg}")
    print(f"Average IoU: {iou_meter.avg}")

    # Load the original COCO data
    with open(cfg.dataset.test.annotation_file, 'r') as file:
        coco_data = json.load(file)
    
    # Create the new COCO annotation structure
    output_annotations = create_coco_annotation(coco_data, cfg.out_dir, predicted_masks)
    
    # Save the new COCO annotations to a JSON file
    output_annotation_file = os.path.join(cfg.out_dir, 'output_annotations.json')
    with open(output_annotation_file, 'w') as file:
        json.dump(output_annotations, file, indent=4)

if __name__ == "__main__":
    from config_Test import cfg
    visualize(cfg)
