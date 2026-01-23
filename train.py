# imports
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from tqdm import tqdm
from pathlib import Path
from dataset import AudioDataset
from model import ConvTasNet
import warnings

# set config
config = {
    'metadata_dir': Path('/kaggle/input/heart-lung-source-separation/data/metadata'), # path to folder containing all metadata files
    'audio_dir': Path('/kaggle/input/heart-lung-source-separation/data/audio'), # path to folder containing audio files
    'train_samples_per_epoch': 1024, # number of training samples to generate (on-the-fly) per epoch
    'val_samples_per_epoch': 512, # number of validation samples to generate (on-the-fly) per epoch
    'target_sample_rate': 16000,
    'num_samples': 80000,
    'batch_size': 32,
    'epochs': 1,
    'N': 128, # number of filters in autoencoder
    'L': 40, # length of the filters (in samples)
    'B': 128, # number of channels in the bottleneck and residual paths' 1x1-conv blocks
    'H': 256, # number of channels in conv blocks
    'Sc': 128, # number of channels in skip-connections paths' 1x1-conv blocks
    'P': 3, # kernel size in conv blocks
    'R': 2, # number of repeats
    'X': 7, # number of conv blocks in each repeat
    'lr': 3e-4, # learning rate
    'dropout': 0.0, # dropout probability
    'weight_decay': 0.01, # weight decay for AdamW
    'max_norm': float('inf'), # max norm for gradient clipping
    'device': 'cuda' if torch.cuda.is_available() else 'cpu', # set device
    'seed': 123
}

def get_ds(config):
    # datasets
    training_set = AudioDataset(config['metadata_dir'] / 'train.csv', config['audio_dir'], config['train_samples_per_epoch'], config['target_sample_rate'], config['num_samples'])
    validation_set = AudioDataset(config['metadata_dir'] / 'val.csv', config['audio_dir'], config['val_samples_per_epoch'], config['target_sample_rate'], config['num_samples'], deterministic = True)

    # dataloaders
    train_loader = DataLoader(training_set, config['batch_size'], shuffle = True)
    val_loader = DataLoader(validation_set, config['batch_size'], shuffle = False)

    return train_loader, val_loader

def get_model(config):
    model = ConvTasNet(N = config['N'], L = config['L'], B = config['B'], H = config['H'], Sc = config['Sc'], P = config['P'], R = config['R'], X = config['X'], dropout = config['dropout'])
    return model

def train_and_validate(config):
    # set device
    device = config['device']

    # get dataloaders
    train_loader, val_loader = get_ds(config)

    # get model and move to device
    model = get_model(config).to(device)

    # set loss function and move to device
    criterion = ScaleInvariantSignalNoiseRatio().to(device)

    # set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

    # iterate over epochs
    for epoch in tqdm(range(config['epochs'])):
        
        # initialize training and validation losses
        train_loss = 0.0
        val_loss = 0.0
        
        # training loop
        model.train()
        for data in tqdm(train_loader, desc = 'Training...', leave = False):
            # move data to device
            heart_signal, lung_signal, mixed_signal = data['target_heart'], data['target_lung'], data['mixture']
            heart_signal, lung_signal, mixed_signal = heart_signal.to(device), lung_signal.to(device), mixed_signal.to(device)

            # clear gradients
            optimizer.zero_grad()

            # get predictions
            preds = model(mixed_signal) # (batch_size, 2, L)
            preds = preds.view(-1, heart_signal.size(-1)) # (batch_size * 2, L)

            # targets
            targets = torch.concat([heart_signal, lung_signal], dim = 1).view(-1, preds.size(-1))
            
            # compute loss (negate to minimize)
            loss = -criterion(preds, targets)
    
            # backprop 
            loss.backward()
    
            # compute total gradient norm across all parameters
            # total_norm = 0.0
            # for param in model.parameters():
            #     if param.grad is not None:
            #         total_norm += param.grad.data.norm(2).item()**2
            # total_norm = total_norm**0.5
    
            # update total epoch grad norm
            # epoch_grad_norm += total_norm
            
            # update weights
            optimizer.step()
    
            # update total loss over this epoch
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)

        # validation loop
        model.eval()
        # avoid gradient computations
        with torch.no_grad():
    
            # iterate over batches
            for data in tqdm(val_loader, desc = 'Validating...', leave = False):
    
                # read in and move data to device
                heart_signal, lung_signal, mixed_signal = data['target_heart'], data['target_lung'], data['mixture']
                heart_signal, lung_signal, mixed_signal = heart_signal.to(device), lung_signal.to(device), mixed_signal.to(device)
                
                # get predictions
                preds = model(mixed_signal)
                preds = preds.view(-1, heart_signal.size(-1)) # (batch_size * 2, L)
                
                # targets
                targets = torch.concat([heart_signal, lung_signal], dim = 1).view(-1, preds.size(-1))
                
                # compute average loss over batch
                batch_loss = -criterion(preds, targets)
    
                # update total epoch loss
                val_loss += batch_loss.item()
            
            # compute avg loss for this epoch
            val_loss = val_loss / len(val_loader)
    
            # compute si-snr
            si_snr = -val_loss

        print(f'Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')

# if __name__ == '__main__':
#     warnings.filterwarnings('ignore')
#     train_and_validate(config)