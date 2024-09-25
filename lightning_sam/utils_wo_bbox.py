import os
import cv2
import torch
from box import Box
from dataset1 import TestDataset
import torchvision.transforms as transforms
from model_bbox import Model
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm


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
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


def draw_image(image, masks, boxes=None, labels=None, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if boxes is not None:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if masks is not None:
        image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)


# def visualize(cfg: Box):
#     model = Model(cfg)
#     model.setup()
#     model.eval()
#     model.cuda()

#     _, _, test_loader = load_datasets(cfg, img_size=model.image_encoder.img_size)
#     predictor = model.get_predictor()
#     os.makedirs(cfg.out_dir, exist_ok=True)

#     for images, image_filenames in tqdm(test_loader):
#         images = images.cuda()
#         for image, image_filename in zip(images, image_filenames):
#             image = image.permute(1, 2, 0).cpu().numpy()  # Convert back to HWC for processing
#             predictor.set_image(image)
#             masks, _, _ = predictor.predict_torch(
#                 point_coords=None,
#                 point_labels=None,
#                 boxes=None,
#                 multimask_output=False,
#             )
#             image_output = draw_image(image, masks.squeeze(1), boxes=None, labels=None)
#             image_output_path = os.path.join(cfg.out_dir, image_filename)
#             print("saving to : ", image_output_path)
#             cv2.imwrite(image_output_path, cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR))

def visualize(cfg: Box):
    model = Model(cfg)
    model.setup()
    model.eval()
    model.cuda()

    dataset = TestDataset(root_dir=cfg.dataset.test.root_dir,
                          transform=None)

    os.makedirs(cfg.out_dir, exist_ok=True)

    for images, image_filenames in tqdm(dataset):
        images = torch.tensor(images).cuda()
        print(images)
        for image, image_filename in zip(images, image_filenames):
            print("Original image shape:", image.shape)
            image = image.permute(1, 2, 0)
            print("Permuted image shape:", image.shape)
            image = image.cpu().numpy()  # Convert to HWC for processing

            image = image.permute(1, 2, 0).cpu().numpy()  # Convert back to HWC for processing
            with torch.no_grad():
                masks, _ = model(image.unsqueeze(0))  # Forward pass through the model
            image_output = draw_image(image, masks.squeeze(0).cpu().numpy())
            image_output_path = os.path.join(cfg.out_dir, image_filename)
            print("saving to:", image_output_path)
            cv2.imwrite(image_output_path, cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    from config import cfg
    visualize(cfg)
