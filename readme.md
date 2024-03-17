# HMS - Harmful Brain Activity Classification

## Introduction
The goal of this competition is to detect and classify seizures and other types of harmful brain activity. You will develop a model trained on electroencephalography (EEG) signals recorded from critically ill hospital patients.

**Data Description**
1. train_eegs(directory): Contain lots of .parquet files.
2. test_eegs(directory): Contain 1 .parquet file.
3. train_spectrograms(directory): Contain lots of .parquet files.
4. test_spectrograms(directory): Contain 1 .parquet file.
5. train.csv(file): Metadata for the training data.
6. sample_submission.csv(file):
7. test.csv(file): Metadata for the test data.
8. example_figures:

**What is EEG and Spectrogram?**

## Experiment 1: Using EfficientNetB0: LB: 0.57
## Preparing the dataset
1. dataset is a class whick take input as df, config, all_spectogram, all_eeg_spectogram
2. all_spectogram is a dict which contain all the spectogram files.
    1. key: file name
    2. value: spectogram file
3. Getitem function call date_generate function for the index
4. Inside date_generate function
    1. Create an zero tensor of shape (128, 256, 8) called X
    2. select r = (df[index][min]+df[index][max]) // 4
    3. Crop 4 img of shape (300, 100) from the spectogram file while shifting the window by 1
    4. Put the cropped img in X for the first 4 channels
    5. Put the eeg spectogram in the last 4 channels
5. Return X, y of shape (128, 256, 8), (6)

## Model
1. Model is a custom efficientnet b0 model with 8 input channels.
2. With the config you can change whether you want to use the eeg spectogram or not.
3. Model return the logits of shape (bs, 6)


## Experiment 2: Using Resnet34d: LB: 0.47

## To Do: 17032024
**Compare submissions and see what is the difference.**

1. For the first submission, I used CATBoost starter which use tabular information only.: LB: 0.81
2. For the second submission, CATBoost with minor optimization: LB: 0.67
3. For the third submission, use efficientnetb0 with only one spectogram: LB: 0.57
4. For the fourth submission, used resnet34d with only one spectogram: LB: 0.47
5. For the fifth submission, used resnet34d with ensemble for 6 models: LB: 0.45
6. Tried fastai with resnet34: LB: 1.14

