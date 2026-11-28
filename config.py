CONFIG = {
    "dataset": "CIFAR-10",
    "batch_size": 16,
    "test_size": 500,

    "attacks": {
        "fgsm": {
            "epsilon": 0.03
        },

        "pgd": {
            "epsilon": 0.03,
            "step_size": 0.007,
            "steps": 10
        }
    }
}