import pandas as pd
import config
from glob import glob

df = pd.read_csv(config.TRAIN_CSV)
label_cols = df.columns[-6:]
#print(f"Train dataframe shape is: {df.shape}")
#print(f"Labels: {list(label_cols)}")
df.head()

# Read Train Spectrogram
READ_SPEC_FILES = False

paths_spectograms = glob(config.TRAIN_SPECTOGRAMS + "*.parquet")
#print(f'There are {len(paths_spectograms)} spectrogram parquets')

import random
idx = random.randint(0,len(paths_spectograms))
aux = pd.read_parquet(paths_spectograms[idx])
print(f"Spectrogram shape of index {idx} is: {aux.shape}")
print(f"spectrogram shape used in dataset is {aux.iloc[:,1:].values.shape}")

path_eegs = glob(config.TRAIN_EEGS + "*.parquet")
print(f'There are {len(path_eegs)} EEG spectograms')
eeg = pd.read_parquet(path_eegs[idx])
print(f"EEG spectogram shape is: {eeg.shape}")


#if READ_SPEC_FILES:
#    all_spectrograms = {}
#    for file_path in tqdm(paths_spectograms):
#        aux = pd.read_parquet(file_path)
#        name = int(file_path.split("/")[-1].split('.')[0])
#        all_spectrograms[name] = aux.iloc[:,1:].values
#        del aux
#else:
#    all_spectrograms = np.load(config.PRE_LOADED_SPECTOGRAMS, allow_pickle=True).item()
#
#if config.VISUALIZE:
#    idx = np.random.randint(0,len(paths_spectograms))
#    spectrogram_path = paths_spectograms[idx]
#    plot_spectrogram(spectrogram_path)
