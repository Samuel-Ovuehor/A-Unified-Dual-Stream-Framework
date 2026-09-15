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

    TRAIN_CSV = os.path.join(BASE_DIR, "traindata.csv") # Replace with your actual data path
    TEST_CSV  = os.path.join(BASE_DIR, "testdata.csv") # Replace with your actual data path
    
    TRAIN_DIR = TRAIN_DIR     #Replace with your actual data dir
    TEST_DIR  =  TEST_DIR     #Replace with your actual data dir
    

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG.SEED)
NUM_WORKERS = 0
