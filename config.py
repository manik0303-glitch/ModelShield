CONFIG = {
    "model_name": "RAM++",
    "ram_root": "C:/projects/main el sem 3/RAM++",
    "checkpoint_path": "C:/projects/main el sem 3/RAM++/pretrained/ram_plus_swin_large_14m.pth",
    "dataset": "RAM++ Tags",
    "batch_size": 4,
    "test_size": 50,
    "image_size": 384,
    "tag_threshold": 0.68,

    "attacks": {
        "fgsm": {
            "epsilon": 4 / 255
        },

        "pgd": {
            "epsilon": 4 / 255,
            "step_size": 1 / 255,
            "steps": 10
        }
    }
}
