import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader, Dataset

def shift_bboxes(bboxes, image_w, image_h, shift_x, shift_y):
    shifted_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(image_w, x1 + shift_x))
        y1 = max(0, min(image_h, y1 + shift_y))
        x2 = max(0, min(image_w, x2 + shift_x))
        y2 = max(0, min(image_h, y2 + shift_y))
        shifted_bboxes.append([x1, y1, x2, y2])
    return shifted_bboxes

def scale_bboxes(bboxes, scale_factor, image_w, image_h):
    scaled_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        new_w = (x2 - x1) * scale_factor
        new_h = (y2 - y1) * scale_factor
        x1 = max(0, min(image_w, x1 * scale_factor))
        y1 = max(0, min(image_h, y1 * scale_factor))
        x2 = max(0, min(image_w, x1 + new_w))
        y2 = max(0, min(image_h, y1 + new_h))
        scaled_bboxes.append([x1, y1, x2, y2])
    return scaled_bboxes

def change_aspect_ratio_bboxes(bboxes, aspect_factor, image_w, image_h):
    new_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        new_w = (x2 - x1) * aspect_factor
        new_h = (y2 - y1)
        x2 = x1 + new_w
        y2 = y1 + new_h
        x1 = max(0, min(image_w, x1))
        y1 = max(0, min(image_h, y1))
        x2 = max(0, min(image_w, x2))
        y2 = max(0, min(image_h, y2))
        new_bboxes.append([x1, y1, x2, y2])
    return new_bboxes

def jitter_bboxes(bboxes, jitter_amount, image_w, image_h):
    jittered_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        jitter_x = np.random.uniform(-jitter_amount, jitter_amount)
        jitter_y = np.random.uniform(-jitter_amount, jitter_amount)
        x1 = max(0, min(image_w, x1 + jitter_x))
        y1 = max(0, min(image_h, y1 + jitter_y))
        x2 = max(0, min(image_w, x2 + jitter_x))
        y2 = max(0, min(image_h, y2 + jitter_y))
        jittered_bboxes.append([x1, y1, x2, y2])
    return jittered_bboxes

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x + w, y + h])
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            try:
                mask = self.coco.annToMask(ann)
                masks.append(mask)
            except Exception as e:
                print(f"Error processing annotation {ann['id']}: {e}")

        if self.transform:
            image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        return image, torch.tensor(bboxes), torch.tensor(masks).float()

def collate_fn(batch):
    images, bboxes, masks = zip(*batch)
    images = torch.stack(images)
    return images, bboxes, masks

class ResizeAndPad:
    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes):
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)

        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        return image, masks, bboxes

class Augmentation:
    def __init__(self, shift_prob=0.2, shift_range=20, scale_prob=0.2, scale_range=0.2,
                 aspect_ratio_prob=0.2, aspect_ratio_range=0.2, jitter_prob=0.2, jitter_amount=5):
        self.shift_prob = shift_prob
        self.shift_range = shift_range
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.aspect_ratio_prob = aspect_ratio_prob
        self.aspect_ratio_range = aspect_ratio_range
        self.jitter_prob = jitter_prob
        self.jitter_amount = jitter_amount

    def __call__(self, image, masks, bboxes):
        image_h, image_w, _ = image.shape

        if len(bboxes) == 0:
            return image, masks, bboxes
        
        # Randomly select a subset of bounding boxes to transform
        num_boxes = len(bboxes)
        num_boxes_to_transform = max(1, int(num_boxes * 0.2))
        selected_indices = np.random.choice(num_boxes, num_boxes_to_transform, replace=False)
        selected_bboxes = [bboxes[i] for i in selected_indices]

        if np.random.rand() < self.shift_prob:
            shift_x = np.random.uniform(-self.shift_range, self.shift_range)
            shift_y = np.random.uniform(-self.shift_range, self.shift_range)
            selected_bboxes = shift_bboxes(selected_bboxes, image_w, image_h, shift_x, shift_y)

        if np.random.rand() < self.scale_prob:
            scale_factor = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
            selected_bboxes = scale_bboxes(selected_bboxes, scale_factor, image_w, image_h)

        if np.random.rand() < self.aspect_ratio_prob:
            aspect_factor = np.random.uniform(1 - self.aspect_ratio_range, 1 + self.aspect_ratio_range)
            selected_bboxes = change_aspect_ratio_bboxes(selected_bboxes, aspect_factor, image_w, image_h)

        if np.random.rand() < self.jitter_prob:
            selected_bboxes = jitter_bboxes(selected_bboxes, self.jitter_amount, image_w, image_h)

        # Replace the selected bounding boxes in the original list with the transformed ones
        for idx, transformed_bbox in zip(selected_indices, selected_bboxes):
            bboxes[idx] = transformed_bbox

        return image, masks, bboxes

def load_datasets(cfg, img_size):
    resize_and_pad = ResizeAndPad(img_size)
    augment = Augmentation()
    transform = lambda img, masks, bboxes: augment(*resize_and_pad(img, masks, bboxes))

    train = COCODataset(root_dir=cfg.dataset.train.root_dir,
                        annotation_file=cfg.dataset.train.annotation_file,
                        transform=transform)
    val = COCODataset(root_dir=cfg.dataset.val.root_dir,
                      annotation_file=cfg.dataset.val.annotation_file,
                      transform=transform)
    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)
    return train_dataloader, val_dataloader
