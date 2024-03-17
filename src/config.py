


AMP = True
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 32
EPOCHS = 4
FOLDS = 5
FREEZE = False
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1e7
MODEL = "tf_efficientnet_b0"
NUM_FROZEN_LAYERS = 39
NUM_WORKERS = 0 # multiprocessing.cpu_count()
PRINT_FREQ = 20
SEED = 20
TRAIN_FULL_DATA = False
VISUALIZE = True
WEIGHT_DECAY = 0.01

PREPROCESSED_TRAIN = "data/train_preprocessed.csv"
OUTPUT_DIR = "data/working/"
PRE_LOADED_EEGS = 'data/eeg_specs.npy'
PRE_LOADED_SPECTOGRAMS = 'data/specs.npy'
TRAIN_CSV = "data/train.csv"
TRAIN_EEGS = "data/train_eegs/"
TRAIN_SPECTOGRAMS = "data/train_spectrograms/"
