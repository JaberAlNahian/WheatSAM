import os
import numpy as np
import cv2

def load_binary_mask(mask_path):
    # Load the binary mask image and convert it to binary format
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #print(mask_path,mask.shape)
    mask = (mask > 128).astype(np.uint8)
    return mask

def dice_score(gt_mask, pred_mask):
    intersection = np.sum(gt_mask * pred_mask)
    return (2. * intersection) / (np.sum(gt_mask) + np.sum(pred_mask))

def iou_score(gt_mask, pred_mask):
    intersection = np.sum(gt_mask * pred_mask)
    union = np.sum(gt_mask) + np.sum(pred_mask) - intersection
    return intersection / union

def evaluate_masks(gt_dir, pred_dir):
    dice_scores = []
    iou_scores = []
    print(len(os.listdir(gt_dir)),len(os.listdir(pred_dir)))

    for mask_filename in os.listdir(gt_dir):
        gt_mask_path = os.path.join(gt_dir, mask_filename)
        pred_mask_path = os.path.join(pred_dir, mask_filename)
        #print(pred_mask_path,gt_mask_path)
        if os.path.exists(pred_mask_path)==False:
            # pred_mask_path = pred_mask_path.split("_1.png")[0]+".png"
            # if os.path.exists(pred_mask_path):
            #     # print(pred_mask_path)
            #     # print("YEAHHHHHHHHHHHHH")
            #     pass
            # else:
            #     print(pred_mask_path)
            #     print("NO")
                print("not found")
                continue

        gt_mask = load_binary_mask(gt_mask_path)
        pred_mask = load_binary_mask(pred_mask_path)

        dice = dice_score(gt_mask, pred_mask)
        iou = iou_score(gt_mask, pred_mask)
        #print(dice,iou)

        dice_scores.append(dice)
        iou_scores.append(iou)

    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)

    return mean_dice, mean_iou

# Directory paths
gt_mask_dir = '/home/mdjaberal.nahian/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Psi_NRRDtoPNG'
pred_mask_dir = '/home/mdjaberal.nahian/__SAM_OUTPUT_WHEAT_HEAD/keyhan_internal'

# Evaluate masks
mean_dice, mean_iou = evaluate_masks(gt_mask_dir, pred_mask_dir)

print(f"Mean Dice Score: {mean_dice}")
print(f"Mean IoU: {mean_iou}")
