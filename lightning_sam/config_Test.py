from box import Box

config = {
    "random_state": 42,
    "num_devices": 1,
    "batch_size": 1,
    "num_workers": 8,
    "num_epochs": 1,
    "eval_interval": 1,
    "out_dir": "/home/mdjaberal.nahian/__SAM_OUTPUT_WHEAT_HEAD/Encoder_Decoder_B_Error_OntheFLY_30_Epoch",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        # "checkpoint": "/home/mdjaberal.nahian/lightning-sam/lightning_sam/Out_H_5/best_epoch.pth",
        # "checkpoint": "/home/mdjaberal.nahian/sam_vit_b_01ec64.pth",
        # "checkpoint": "/home/mdjaberal.nahian/lightning-sam/lightning_sam/model/sam_vit_l_0b3195.pth",
        "checkpoint": "/home/mdjaberal.nahian/lightning-sam/lightning_sam/Error_OUT/Encoder_Decoder_B_Error_OntheFLY_30_Epoch/best_epoch.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/home/mdjaberal.nahian/wheat_segmentation.v2i.coco-segmentation/train",
            "annotation_file": "/home/mdjaberal.nahian/wheat_segmentation.v2i.coco-segmentation/train_annotations.coco.json"
        },
        "val": {
            "root_dir": "/home/mdjaberal.nahian/wheat_segmentation.v2i.coco-segmentation/valid",
            "annotation_file": "/home/mdjaberal.nahian/wheat_segmentation.v2i.coco-segmentation/valid_annotations.coco.json"
        },
         'bbox': {
            'annotation_file': '/home/mdjaberal.nahian/FasterRCNN/Outputs/Valid_Pre_Matched_Image_V2i/_annotations.json'
        },
        "test": {
            "root_dir": "/home/mdjaberal.nahian/combined_synthetic_wheat/test/images",
            "annotation_file": "/home/mdjaberal.nahian/combined_synthetic_wheat/test/test_annotations.json",
            "bbox_annotation_file":'/home/mdjaberal.nahian/YOLOV9/Test_Output_v2i/coco_results.json'
            # "bbox_annotation_file": "/home/mdjaberal.nahian/FasterRCNN/Outputs/Test_Pre_Matched_V2i/_annotations.json"
        }
        
    }
}

cfg = Box(config)
