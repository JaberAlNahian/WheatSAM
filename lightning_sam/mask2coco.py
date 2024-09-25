import os
import json
import numpy as np
import cv2
import nrrd
from PIL import Image
from pycocotools import mask as maskUtils

def create_coco_annotation(data_folder, output_file):
    images = []
    annotations = []
    categories = [{"id": 1, "name": "wheat", "supercategory": "wheat"}]
    
    image_id = 0
    annotation_id = 0
    
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.jpg'):
            image_id += 1
            
            img_path = os.path.join(data_folder, file_name)
            mask_path = os.path.join(data_folder, file_name.rsplit('.', 1)[0] + '.nrrd')
            
            img = Image.open(img_path)
            width, height = img.size
            
            images.append({
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height
            })
            
            mask, header = nrrd.read(mask_path)
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]  # Use the first slice if the mask has more than 2 dimensions
            
            unique_objects = np.unique(mask)
            
            for obj_id in unique_objects:
                if obj_id == 0:  # Skip the background
                    continue
                
                obj_mask = np.uint8(mask == obj_id)
                contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # for contour in contours:
                #     if len(contour) < 3:
                #         continue
                    
                #     segmentation = contour.flatten().tolist()
                    
                #     rle = maskUtils.encode(np.asfortranarray(obj_mask))
                #     area = float(maskUtils.area(rle))
                #     bbox = maskUtils.toBbox(rle).tolist()
                    
                #     annotations.append({
                #         "id": annotation_id,
                #         "image_id": image_id,
                #         "category_id": 1,
                #         "segmentation": [segmentation],
                #         "area": area,
                #         "bbox": bbox,
                #         "iscrowd": 0
                #     })
                    
                #     annotation_id += 1
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    
                    segmentation = contour.flatten().tolist()
                    
                    # Calculate bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    bbox = [float(x), float(y), float(w), float(h)]
                    
                    rle = maskUtils.encode(np.asfortranarray(obj_mask))
                    area = float(maskUtils.area(rle))

                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "segmentation": [segmentation],
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    })
                    
                    annotation_id += 1

    
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

if __name__ == "__main__":
    data_folder = '/home/mdjaberal.nahian/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Psi'  # Folder containing both images and masks
    output_file = '/home/mdjaberal.nahian/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Psi_coco_segmentation.json'
    
    create_coco_annotation(data_folder, output_file)
    print(f"COCO segmentation JSON file created at {output_file}")
