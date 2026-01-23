from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as F

class AudioDataset(Dataset):
    """
    PyTorch Dataset for audio source separation of heart and lung sounds.
    
    This dataset creates mixed audio samples by combining heart sounds and lung sounds
    from different recordings. It handles audio loading, resampling, normalization,
    and mixing with random signal-to-noise (SNR) ratios for robust model training.
    
    Args:
        metadata_file (str): Path to CSV file containing audio metadata with columns
            'Heart Sound ID' and 'Lung Sound ID'.
        audio_dir (str): Directory path containing audio files in WAV format.
        samples_per_epoch (int): Number of samples to generate per epoch (controls
            dataset length).
        target_sample_rate (int, optional): Target sampling rate in Hz for all audio.
            Defaults to 16000.
        num_samples (int, optional): Number of audio samples (time steps) per clip.
            Defaults to 80000 (5 seconds at 16kHz).
        deterministic (bool, optional): If True, uses index-based seeding for
            reproducible validation sets. Defaults to False.
        eps (float, optional): Small constant for numerical stability in normalization.
            Defaults to 1e-8.
    
    Returns:
        dict: Dictionary containing:
            - mixture: Mixed audio signal (heart + lung)
            - target_heart: Isolated heart sound component
            - target_lung: Isolated lung sound component
            - scaling_factor: Maximum value used for final normalization
            - ids: List of [heart_sound_id, lung_sound_id]
    """
    
    def __init__(
        self, 
        metadata_file,
        audio_dir,
        samples_per_epoch,
        target_sample_rate=16000,
        num_samples=80000,
        deterministic=False,
        eps=1e-8
    ):
        # Load metadata containing audio file IDs
        self.metadata = pd.read_csv(metadata_file)
        
        # Convert audio directory to Path object for robust file handling
        self.audio_dir = Path(audio_dir)
        
        # Number of samples to generate per epoch (dataset length)
        self.samples_per_epoch = samples_per_epoch
        
        # Target sampling rate for all audio (Hz)
        self.target_sample_rate = target_sample_rate
        
        # Number of time samples per audio clip
        self.num_samples = num_samples
        
        # Whether to use deterministic sampling for validation
        self.deterministic = deterministic
        
        # Small epsilon for numerical stability
        self.eps = eps
    
    def __len__(self):
        """
        Returns the number of samples in an epoch.
        
        Returns:
            int: Number of samples per epoch.
        """
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        """
        Generates a single mixed audio sample with heart and lung sound components.
        
        This method:
        1. Selects a heart sound based on the index
        2. Randomly selects a different lung sound
        3. Loads and preprocesses both audio files (resample, mix to mono)
        4. Applies random cropping if signals are longer than num_samples
        5. Normalizes each signal to unit energy
        6. Applies random SNR mixing
        7. Combines signals and applies final scaling
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: Dictionary containing mixed and separated audio components plus metadata.
        """
        # Set random seed based on index for deterministic validation sets
        if self.deterministic:
            np.random.seed(idx)
        
        # Cycle through metadata indices to ensure all samples are used
        idx = idx % len(self.metadata)
        
        # Get heart sound ID from metadata at the current index
        hs_id = self.metadata.loc[idx, 'Heart Sound ID']
        
        # Select a random lung sound ID that is different from the heart sound index
        possible_indices = torch.arange(len(self.metadata))  # All possible indices
        mask = torch.ones(possible_indices.size(0), dtype=torch.bool)  # Create boolean mask
        mask[idx] = False  # Exclude current index to ensure different sounds
        possible_indices = possible_indices[mask]  # Filter to valid indices
        lung_idx = np.random.choice(possible_indices)  # Randomly select lung sound index
        ls_id = self.metadata.loc[lung_idx, 'Lung Sound ID']  # Get lung sound ID
        
        # Load audio files and their original sampling rates
        heart_signal, heart_sr = torchaudio.load(self.audio_dir / f'{hs_id}.wav')
        lung_signal, lung_sr = torchaudio.load(self.audio_dir / f'{ls_id}.wav')
        
        # Resample heart signal to target sample rate if necessary
        if heart_sr != self.target_sample_rate:
            heart_signal = F.resample(heart_signal, heart_sr, self.target_sample_rate)
        
        # Resample lung signal to target sample rate if necessary
        if lung_sr != self.target_sample_rate:
            lung_signal = F.resample(lung_signal, lung_sr, self.target_sample_rate)
        
        # Convert stereo to mono by averaging channels if necessary
        if heart_signal.size(0) > 1:
            heart_signal = torch.mean(heart_signal, dim=0, keepdim=True)
        
        if lung_signal.size(0) > 1:
            lung_signal = torch.mean(lung_signal, dim=0, keepdim=True)
        
        # Apply random cropping to heart signal if longer than target length
        if heart_signal.size(1) >= self.num_samples:
            start_idx = np.random.randint(0, heart_signal.size(1) - self.num_samples + 1)
            heart_signal = heart_signal[:, start_idx: start_idx + self.num_samples]
        
        # Apply random cropping to lung signal if longer than target length
        if lung_signal.size(1) >= self.num_samples:
            start_idx = np.random.randint(0, lung_signal.size(1) - self.num_samples + 1)
            lung_signal = lung_signal[:, start_idx: start_idx + self.num_samples]
        
        # Normalize each signal to unit energy (L2 norm = 1)
        heart_signal /= (torch.linalg.norm(heart_signal) + self.eps)
        lung_signal /= (torch.linalg.norm(lung_signal) + self.eps)
        
        # Apply random SNR (Signal-to-Noise Ratio) between -5 and 5 dB
        snr_db = np.random.randint(-5, 5)
        scale = 10 ** (snr_db / 20)  # Convert dB to linear scale
        heart_signal *= scale
        
        # Mix the heart and lung signals
        mixed_signal = heart_signal + lung_signal
        
        # Scale all signals to prevent clipping (normalize by max absolute value)
        max_val = torch.abs(mixed_signal).max()
        mixed_signal /= max_val
        heart_signal /= max_val
        lung_signal /= max_val
        
        # Return dictionary with all components
        return {
            'mixture': mixed_signal, # Mixed audio (heart + lung)
            'target_heart': heart_signal, # Isolated heart sound
            'target_lung': lung_signal, # Isolated lung sound
            'scaling_factor': max_val, # Scaling factor for potential reconstruction
            'ids': [hs_id, ls_id] # Original audio IDs for tracking
        }