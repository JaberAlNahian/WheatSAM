from box import Box

config = {
    "random_state": 42,
    "num_devices": 1,
    "batch_size": 2,
    "num_workers": 4,
    "num_epochs": 10,
    "eval_interval": 1,
    "finetune_type": "lora",  # Options: "vanilla", "adapter", "lora"
    "lora_rank": 4,  # LoRA rank if using finetune_type='lora' (optional)
    "if_encoder_lora_layer": True,
    "if_decoder_lora_layer": False,
    "encoder_lora_layer": [],
    "out_dir": "/home/mdjaberal.nahian/lightning-sam/lightning_sam/SYNTHETIC/Decoder_B_10_Epochs",
    "opt": {
        "learning_rate": 1e-6,
        "weight_decay": 1e-5,
        "decay_factor": 0.1,
        "steps": [2, 5],
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
        "checkpoint": "/home/mdjaberal.nahian/lightning-sam/lightning_sam/model/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/home/mdjaberal.nahian/instanceSeg2/green_young_longheads_inrows_seg/images",
            "annotation_file": "/home/mdjaberal.nahian/instanceSeg2/green_young_longheads_inrows_seg/coco_annotations_onlywheatheads.json"
        },
        "val": {
            "root_dir": "/home/mdjaberal.nahian/instanceSeg2/green_young_longheads_seg/images",
            "annotation_file": "/home/mdjaberal.nahian/instanceSeg2/green_young_longheads_seg/coco_annotations_onlywheatheads.json"
        },
         'bbox': {
            'annotation_file': '/home/mdjaberal.nahian/FasterRCNN/Outputs/Valid_Pre_Matched_Image_V2i/_annotations.json'
        },
        "test": {
            "root_dir": "/home/mdjaberal.nahian/YOLOV9/test_out_Keyan",
            "annotation_file": "/home/mdjaberal.nahian/YOLOV9/test_out_Keyan/coco_results.json",
            "bbox_annotation_file":'/home/mdjaberal.nahian/FasterRCNN/Outputs/Test_Pre_Matched_V2i/_annotations.json'
        }
        
    }
}

cfg = Box(config)
