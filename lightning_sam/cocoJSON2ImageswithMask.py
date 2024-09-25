import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.mask import decode

def visualize_annotations(image_folder, annotation_file, output_folder):
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    for image_info in images:
        image_id = image_info['id']
        image_file = image_info['file_name']
        if not image_file.endswith('.jpg'):
            continue  # Skip non-jpg files
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

        for ann in image_annotations:
            category_id = ann['category_id']
            category_info = [cat for cat in categories if cat['id'] == category_id][0]
            category_name = category_info['name']
            bbox = ann['bbox']
            segmentation = ann['segmentation']

            # Convert segmentation to polygon format
            polygons = []
            for seg in segmentation:
                poly = np.array(seg).reshape(-1, 2)
                polygons.append(poly.astype(int))

            # Draw bounding box
            x, y, w, h = map(int, bbox)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw segmentation mask
            # color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            # for poly in polygons:
            #     cv2.fillPoly(image, [poly], color_mask)
            # Draw segmentation mask
            color_mask = (int(np.random.randint(0, 256)), int(np.random.randint(0, 256)), int(np.random.randint(0, 256)))
            for poly in polygons:
                cv2.fillPoly(image, [poly], color_mask)
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)

# Example usage
image_folder = '/home/mdjaberal.nahian/wheat-1/test'
annotation_file = '/home/mdjaberal.nahian/wheat-1/test/test_new_annotations_with_errors.json'
output_folder = '/home/mdjaberal.nahian/wheat-1/error_test_over;ap'
# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

visualize_annotations(image_folder, annotation_file, output_folder)
