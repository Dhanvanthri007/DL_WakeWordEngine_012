import torch
import torchaudio

import soundfile as sf
import numpy as np

# Completely bypass torchaudio.load due to torchcodec bug
def _custom_load(filepath, *args, **kwargs):
    data, sr = sf.read(filepath)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    else:
        data = data.T
    return torch.from_numpy(data).float(), sr
torchaudio.load = _custom_load
import librosa
import numpy as np

def pad_sequence(batch):
    """
    Pad sequences to same length
    """
    # Find max length
    max_len = max([x.shape[2] for x in batch])
    
    padded_batch = []
    for x in batch:
        pad_amount = max_len - x.shape[2]
        padded_x = torch.nn.functional.pad(x, (0, pad_amount))
        padded_batch.append(padded_x)
        
    return torch.stack(padded_batch)

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, n_mfcc=40):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_mels": self.n_mfcc, "hop_length": 512}
        )

    def extract_features(self, waveform):
        """
        Extract MFCC features from an audio waveform
        Args:
            waveform: Tensor of shape (1, seq_length)
        Returns:
            mfcc: Tensor of shape (1, n_mfcc, time_steps)
        """
        mfcc = self.transform(waveform)
        return mfcc

    def process_file(self, filepath):
        """
        Load audio file and extract features
        """
        waveform, sr = torchaudio.load(filepath)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        # Ensure single channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        return self.extract_features(waveform)

    def collate_fn(self, batch):
        """
        Collate function for DataLoader
        """
        tensors, targets = [], []
        
        for waveform, _, label, *_ in batch:
            mfcc = self.extract_features(waveform)
            tensors.append(mfcc)
            targets.append(label)
            
        tensors = pad_sequence(tensors)
        targets = torch.tensor(targets)
        
        return tensors, targets
