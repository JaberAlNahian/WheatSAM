import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader, Dataset
import random

class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.transform = transform
        self.augmentation = augmentation
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
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
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        if self.augmentation:
            image, masks, bboxes = self.augmentation(image, masks, bboxes)

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

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # Adjust bounding boxes
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        return image, masks, bboxes


class AugmentationPipeline:

    def __init__(self, shift_prob=0.2, scale_prob=0.2, rotation_prob=0.2, aspect_ratio_prob=0.2, crop_prob=0.2, noise_prob=0.2):
        self.shift_prob = shift_prob
        self.scale_prob = scale_prob
        self.rotation_prob = rotation_prob
        self.aspect_ratio_prob = aspect_ratio_prob
        self.crop_prob = crop_prob
        self.noise_prob = noise_prob

    def __call__(self, image, masks, bboxes):
        og_h, og_w, _ = image.shape

        # Shift
        if random.random() < self.shift_prob:
            shift_x = np.random.uniform(-20, 20)
            shift_y = np.random.uniform(-20, 20)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            image = cv2.warpAffine(image, M, (og_w, og_h))
            masks = [cv2.warpAffine(mask, M, (og_w, og_h)) for mask in masks]
            bboxes = [[bbox[0] + shift_x, bbox[1] + shift_y, bbox[2] + shift_x, bbox[3] + shift_y] for bbox in bboxes]

        # Scale
        if random.random() < self.scale_prob:
            scale_factor = np.random.uniform(0.8, 1.2)
            image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
            masks = [cv2.resize(mask, None, fx=scale_factor, fy=scale_factor) for mask in masks]
            bboxes = [[bbox[0] * scale_factor, bbox[1] * scale_factor, bbox[2] * scale_factor, bbox[3] * scale_factor] for bbox in bboxes]

        # Rotation
        if random.random() < self.rotation_prob:
            angle = np.random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((og_w // 2, og_h // 2), angle, 1)
            image = cv2.warpAffine(image, M, (og_w, og_h))
            masks = [cv2.warpAffine(mask, M, (og_w, og_h)) for mask in masks]

        # Aspect Ratio Change
        if random.random() < self.aspect_ratio_prob:
            aspect_factor = np.random.uniform(0.8, 1.2)
            new_w = int(og_w * aspect_factor)
            image = cv2.resize(image, (new_w, og_h))
            masks = [cv2.resize(mask, (new_w, og_h)) for mask in masks]
            bboxes = [[bbox[0] * aspect_factor, bbox[1], bbox[2] * aspect_factor, bbox[3]] for bbox in bboxes]

        # Random Cropping
        if random.random() < self.crop_prob:
            x1 = random.randint(0, og_w // 4)
            y1 = random.randint(0, og_h // 4)
            x2 = random.randint(3 * og_w // 4, og_w)
            y2 = random.randint(3 * og_h // 4, og_h)
            image = image[y1:y2, x1:x2]
            masks = [mask[y1:y2, x1:x2] for mask in masks]
            bboxes = [[max(bbox[0] - x1, 0), max(bbox[1] - y1, 0), max(bbox[2] - x1, 0), max(bbox[3] - y1, 0)] for bbox in bboxes]

        # Gaussian Noise
        if random.random() < self.noise_prob:
            noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
            image = np.clip(image + noise, 0, 255)

        return image, masks, bboxes


def load_datasets(cfg, img_size):
    augmentation = AugmentationPipeline()
    transform = ResizeAndPad(img_size)
    train = COCODataset(root_dir=cfg.dataset.train.root_dir,
                        annotation_file=cfg.dataset.train.annotation_file,
                        transform=transform,
                        augmentation=augmentation)
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
