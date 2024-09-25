from box import Box

config = {
    "random_state": 42,
    "num_devices": 1,
    "batch_size": 2,
    "num_workers": 4,
    "num_epochs": 30,
    "eval_interval": 1,
    "finetune_type": "lora",  # Options: "vanilla", "adapter", "lora"
    "lora_rank": 4,  # LoRA rank if using finetune_type='lora' (optional)
    "if_encoder_lora_layer": True,
    "if_decoder_lora_layer": False,
    "encoder_lora_layer": [],
    "out_dir": "/home/mdjaberal.nahian/lightning-sam/lightning_sam/Error_OUT/Encoder_Decoder_B_Error_OntheFLY_30_Epoch",
    "opt": {
        "learning_rate": 1e-7,
        "weight_decay": 1e-5,
        "decay_factor": 0.1,
        "steps": [10, 15],
        "warmup_steps": 5
    # "random_state": 42,
    # "num_devices": 1,
    # "batch_size": 2,
    # "num_workers": 8,
    # "num_epochs": 50,
    # "eval_interval": 2,
    # "out_dir": "/home/mdjaberal.nahian/lightning-sam/OUTPUTS_Vit_B",
    # "opt": {
    #     "learning_rate": 8e-4,
    #     "weight_decay": 1e-4,
    #     "decay_factor": 10,
    #     "steps": [60000, 86666],
    #     "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        # "checkpoint": "/home/mdjaberal.nahian/lightning-sam/OUTPUTS_Vit_B_Cookie_dataset/best_epoch.pth",
        "checkpoint": "/home/mdjaberal.nahian/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": False,
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
            "root_dir": "/home/mdjaberal.nahian/wheat_segmentation.v2i.coco-segmentation/test",
            "annotation_file": "/home/mdjaberal.nahian/wheat_segmentation.v2i.coco-segmentation/test_annotations.coco.json",
            "bbox_annotation_file":'/home/mdjaberal.nahian/FasterRCNN/Outputs/Test_Pre_Matched_V2i/_annotations.json'
        }
        
    }
}

cfg = Box(config)
