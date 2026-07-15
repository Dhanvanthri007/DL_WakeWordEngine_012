import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
from preprocess import AudioPreprocessor
from model import WakeWordCNN

# Configuration
WAKE_WORD = "marvin"
DATASET_PATH = "./datasets"
MODEL_SAVE_PATH = "../models/best_model.pth"
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001

class SpeechCommandsDataset(Dataset):
    def __init__(self, torchaudio_dataset, wake_word=WAKE_WORD, max_samples=None):
        self.dataset = torchaudio_dataset
        self.wake_word = wake_word
        self.max_samples = max_samples
        
    def __len__(self):
        if self.max_samples:
            return min(len(self.dataset), self.max_samples)
        return len(self.dataset)
        
    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[idx]
        
        # Label is 1 if it's the wake word, 0 otherwise
        target = 1 if label == self.wake_word else 0
        return waveform, sample_rate, target, speaker_id, utterance_number

def train():
    print("Downloading and preparing dataset...")
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # Download dataset
    train_dataset = torchaudio.datasets.SPEECHCOMMANDS(
        DATASET_PATH, 
        url='speech_commands_v0.02', 
        folder_in_archive='SpeechCommands', 
        download=True, 
        subset='training'
    )
    val_dataset = torchaudio.datasets.SPEECHCOMMANDS(
        DATASET_PATH, 
        url='speech_commands_v0.02', 
        folder_in_archive='SpeechCommands', 
        download=True, 
        subset='validation'
    )
    
    print("Limiting to a subset for quick demonstration...")
    # Limiting the dataset to 2000 samples to avoid excessive training time
    train_subset = SpeechCommandsDataset(train_dataset, wake_word=WAKE_WORD, max_samples=2000)
    val_subset = SpeechCommandsDataset(val_dataset, wake_word=WAKE_WORD, max_samples=400)
    
    preprocessor = AudioPreprocessor()
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=preprocessor.collate_fn
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=preprocessor.collate_fn
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = WakeWordCNN(num_classes=2, in_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("=> Model saved!")
            
    print("Training complete.")

if __name__ == "__main__":
    train()
