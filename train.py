# imports
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from tqdm.auto import tqdm
import wandb
from dataset import AudioDataset
from model import ConvTasNet
import warnings
warnings.filterwarnings("ignore")

# set config
config = {
    'metadata_dir': '/kaggle/input/heart-lung-source-separation/data/metadata', # path to folder containing training, validation, & testing metadata files
    'audio_dir': '/kaggle/input/heart-lung-source-separation/data/audio', # path to folder containing audio files
    'train_samples_per_epoch': 1024, # number of training samples to generate (on-the-fly) per epoch
    'val_samples_per_epoch': 512, # number of validation samples to generate (on-the-fly) per epoch
    'target_sample_rate': 16000,
    'num_samples': 80000,
    'batch_size': 32,
    'epochs': 25,
    'N': 128, # number of filters in autoencoder
    'L': 40, # length of the filters (in samples)
    'B': 128, # number of channels in the bottleneck and residual paths' 1x1-conv blocks
    'H': 256, # number of channels in conv blocks
    'Sc': 128, # number of channels in skip-connections paths' 1x1-conv blocks
    'P': 3, # kernel size in conv blocks
    'R': 2, # number of repeats
    'X': 7, # number of conv blocks in each repeat
    'lr': 3e-4, # learning rate
    'weight_decay': 0.01, # weight decay for AdamW
    'max_norm': float('inf'), # max norm for gradient clipping
    'dropout': 0.0, # dropout probability
    'device': 'cuda' if torch.cuda.is_available() else 'cpu', # set device
    'seed': 123
}
