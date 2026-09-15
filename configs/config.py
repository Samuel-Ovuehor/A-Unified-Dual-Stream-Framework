class CFG:
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    SEED = 42
    IMG_SIZE = 192
    NUM_CLASSES = 5  # Replace 5 with your actual number of classes

    FINAL_EPOCHS = 100
    HMS_TRIALS = 12
    HMS_EPOCHS = 12

    CLASS_NAMES = [
        "class_1",
        "class_2",
        "class_3",
        "class_4",
        "class_5",
    ]
