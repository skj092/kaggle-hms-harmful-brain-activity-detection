import albumentations as A
import numpy as np
import os
import pandas as pd
import time
import torch
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')

# Read Train Spectrogram
READ_SPEC_FILES = False

paths_spectograms = glob(config.TRAIN_SPECTOGRAMS + "*.parquet")
print(f'There are {len(paths_spectograms)} spectrogram parquets')


tik = time.perf_counter()
if READ_SPEC_FILES:
    all_spectrograms = {}
    for file_path in tqdm(paths_spectograms):
        aux = pd.read_parquet(file_path)
        name = int(file_path.split("/")[-1].split('.')[0])
        all_spectrograms[name] = aux.iloc[:,1:].values
        del aux
else:
    all_spectrograms = np.load(config.PRE_LOADED_SPECTOGRAMS, allow_pickle=True).item()
tok = time.perf_counter()
print(f"Read spectrograms time: {tok-tik:.2f} seconds")

# Read Train EEG
READ_EEG_SPEC_FILES = False

paths_eegs = glob(config.TRAIN_EEGS + "*.parquet")
print(f'There are {len(paths_eegs)} EEG spectograms')

tik = time.perf_counter()
if READ_EEG_SPEC_FILES:
    all_eegs = {}
    for file_path in tqdm(paths_eegs):
        eeg_id = file_path.split("/")[-1].split(".")[0]
        eeg_spectogram = np.load(file_path)
        all_eegs[eeg_id] = eeg_spectogram
else:
    all_eegs = np.load(config.PRE_LOADED_EEGS, allow_pickle=True).item()
tok = time.perf_counter()
print(f"Read EEG spectograms time: {tok-tik:.2f} seconds")

label_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

class CustomDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, config,
        augment: bool = False, mode: str = 'train',
        specs: Dict[int, np.ndarray] = all_spectrograms,
        eeg_specs: Dict[int, np.ndarray] = all_eegs
    ):
        self.df = df
        self.config = config
        self.batch_size = self.config.BATCH_SIZE_TRAIN
        self.augment = augment
        self.mode = mode
        self.spectograms = all_spectrograms
        self.eeg_spectograms = eeg_specs

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __data_generation(self, index):
        """
        Generates data containing batch_size samples.
        """
        X = np.zeros((128, 256, 8), dtype='float32')
        y = np.zeros(6, dtype='float32')
        img = np.ones((128,256), dtype='float32')
        row = self.df.iloc[index]
        if self.mode=='test':
            r = 0
        else:
            r = int((row['min'] + row['max']) // 4)

        for region in range(4):
            img = self.spectograms[row.spectogram_id][r:r+300, region*100:(region+1)*100].T

            # Log transform spectogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # Standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img-mu)/(std+ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0
            img = self.eeg_spectograms[row.eeg_id]
            X[:, :, 4:] = img

            if self.mode != 'test':
                y = row[label_cols].values.astype(np.float32)

        return X, y

    def __transform(self, img):
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
        ])
        return transforms(image=img)['image']

if __name__=="__main__":
    preprocesed_path = config.PREPROCESSED_TRAIN
    df = pd.read_csv(preprocesed_path)
    print(f"Train cataframe shape is: {df.shape}")
    tik = time.perf_counter()
    train_dataset = CustomDataset(df, config, mode="train")
    tok = time.perf_counter()
    print(f"Dataset creation time: {tok-tik:.2f} seconds")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    X, y = train_dataset[0]
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    xb, yb = next(iter(train_loader))
    print(f"X batch shape: {xb.shape}")
    print(f"y batch shape: {yb.shape}")
