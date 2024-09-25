import os
import json
import numpy as np
from skimage import measure

def binary_mask_to_coco(mask_folder, output_json_path):
    # Initialize COCO dataset structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "wheat", "supercategory": "object"}]  # Define categories
    }
    annotation_id = 1

    for filename in os.listdir(mask_folder):
        if filename.endswith('.png'):
            image_id = int(filename.split('.')[0])  # Extract image ID from filename
            image_info = {"id": image_id, "file_name": filename, "height": 1024, "width": 1024}  # Add image info
            coco_data["images"].append(image_info)

            mask_path = os.path.join(mask_folder, filename)
            mask = np.array(Image.open(mask_path))  # Read mask image

            # Find contours in the binary mask
            contours = measure.find_contours(mask, 0.5, positive_orientation='low')

            for contour in contours:
                contour = np.flip(contour, axis=1)  # Flip coordinates to (y, x) format
                segmentation = contour.ravel().tolist()

                # Create annotation data
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # Wheat category ID
                    "segmentation": [segmentation],
                    "bbox": [int(np.min(contour[:, 1])), int(np.min(contour[:, 0])),
                             int(np.max(contour[:, 1]) - np.min(contour[:, 1])),
                             int(np.max(contour[:, 0]) - np.min(contour[:, 0]))],
                    "area": measure.perimeter(contour),
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1

    # Write COCO JSON data to file
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f)

# Example usage
mask_folder = '/home/mdjaberal.nahian/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Psi_NRRDtoPNG'  # Folder containing binary mask images
output_json_path = '/home/mdjaberal.nahian/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Psi_NRRDtoPNG_annotations.json'  # Output JSON file path

binary_mask_to_coco(mask_folder, output_json_path)
