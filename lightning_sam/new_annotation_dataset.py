import json
import random

def create_small_dataset(large_annotation_file, output_annotation_file, sample_size):
    # Load the large annotation file
    with open(large_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Randomly sample image IDs
    all_images = coco_data['images']
    sampled_images = random.sample(all_images, sample_size)

    # Get the corresponding annotations
    sampled_image_ids = {img['id'] for img in sampled_images}
    sampled_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in sampled_image_ids]

    # Reindex the image and annotation IDs starting from 1
    image_id_mapping = {img['id']: idx+1 for idx, img in enumerate(sampled_images)}
    annotation_id_mapping = {ann['id']: idx+1 for idx, ann in enumerate(sampled_annotations)}

    # Update image and annotation IDs
    for img in sampled_images:
        img['id'] = image_id_mapping[img['id']]

    for ann in sampled_annotations:
        ann['image_id'] = image_id_mapping[ann['image_id']]
        ann['id'] = annotation_id_mapping[ann['id']]

    # Create a new dataset
    small_coco_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': sampled_images,
        'annotations': sampled_annotations
    }

    # Save the new small annotation file
    with open(output_annotation_file, 'w') as f:
        json.dump(small_coco_data, f, indent=4)

    print(f"Small dataset with reindexed IDs created: {output_annotation_file}")

# Example usage
large_annotation_file = '/home/mdjaberal.nahian/combined_synthetic_wheat/train_annotations.json'
output_annotation_file = '/home/mdjaberal.nahian/combined_synthetic_wheat/small_train_annotations.json'
sample_size = 1000  # Number of images for the small dataset

create_small_dataset(large_annotation_file, output_annotation_file, sample_size)
