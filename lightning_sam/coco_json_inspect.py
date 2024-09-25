import json
import gc
from pycocotools.coco import COCO

def check_annotations(annotation_file, batch_size=2):
    # Load the COCO annotations
    coco = COCO(annotation_file)

    bbox_data_types = []
    segmentation_data_types = []

    ann_ids = list(coco.anns.keys())
    
    # Process annotations in batches
    for i in range(0, len(ann_ids), batch_size):
        batch_ids = ann_ids[i:i + batch_size]
        for ann_id in batch_ids:
            ann = coco.anns[ann_id]

            # Check bounding box data type
            bbox = ann.get('bbox', None)
            if bbox is not None:
                if isinstance(bbox, list):
                    bbox_data_types.extend([type(val).__name__ for val in bbox])
                else:
                    bbox_data_types.append(type(bbox).__name__)

            # Check segmentation data type
            segmentation = ann.get('segmentation', None)
            if segmentation is not None:
                if isinstance(segmentation, list):
                    for seg in segmentation:
                        if isinstance(seg, list):
                            segmentation_data_types.extend([type(val).__name__ for val in seg])
                        else:
                            segmentation_data_types.append(type(seg).__name__)
                else:
                    segmentation_data_types.append(type(segmentation).__name__)

        # Force garbage collection
        gc.collect()

    # Print results
    print("BBox Data Types:")
    print(set(bbox_data_types))  # Unique data types for bbox
    print("\nSegmentation Data Types:")
    print(set(segmentation_data_types))  # Unique data types for segmentation



if __name__ == "__main__":
    annotation_file = '/home/mdjaberal.nahian/combined_synthetic_wheat/train_annotations.json'
    check_annotations(annotation_file)


