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

    predicted_masks = {}

    for image_id in tqdm(dataset.image_ids):
        image_info = dataset.coco.loadImgs(image_id)[0]
        image_path = os.path.join(dataset.root_dir, image_info['file_name'])
        image_output_path = os.path.join(cfg.out_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ann_ids = dataset.coco.getAnnIds(imgIds=image_id)
        anns = dataset.coco.loadAnns(ann_ids)
        bboxes = []
        for ann in anns:
            if ann['segmentation']:  # Skip annotations with empty segmentation
                try:
                    x, y, w, h = ann['bbox']
                    bboxes.append([x, y, x + w, y + h])
                except IndexError as e:
                    print(f"IndexError: {e}")
                    print(f"Problematic annotation ID: {ann['id']}")
                    print(f"Annotation data: {ann}")
                    continue
        
        if not bboxes:
            print(f"No valid bounding boxes for image ID {image_id}, skipping.")
            continue
        
        bboxes = torch.as_tensor(bboxes, device=model.model.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])
        predictor.set_image(image)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        predicted_masks[image_id] = masks.squeeze(1).cpu().numpy()

        image_output = draw_image(image, masks.squeeze(1), boxes=None, labels=None)
        print("saving to : ", image_output_path)
        cv2.imwrite(image_output_path, cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR))

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
