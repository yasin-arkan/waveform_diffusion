import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import pandas as pd 
import random 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


N_FFT = 256       
HOP_LENGTH = 64  
WIN_LENGTH = N_FFT 
WINDOW_TENSOR = torch.hann_window(WIN_LENGTH, periodic=True)

def load_data(path, n_fft, hop_length, win_length ,window):
    data = pd.read_csv(path)

    cond_cols = ['EventLat', 'EventLon', 'StationLat', 'StationLon']
    cond_vars = data[cond_cols]

    cond_data = []


    for cvar in cond_cols:
        cv = cond_vars[cvar].to_numpy()
        cv = cv.reshape(cv.shape[0], 1)
        cond_data.append(cv)
    
    
    wfs = data.iloc[:, 16:].to_numpy()
    orig_wfs = torch.from_numpy(wfs).float()

    wfs = torch.stft(orig_wfs, n_fft, hop_length, win_length, window, return_complex=True)

    wfs = torch.abs(wfs) # Magnitude 

    # Our goal here is to pad the shape into [1183, 128, 128]
    print("Before padding:", wfs.shape) # Right now it is [1183, 129, 110]
    wfs = wfs[:, :128, :] # Discarding the last frequency to make it 128 

    current_time_dim = wfs.shape[2] # Should be 110, we will pad it to 128
    padding_needed = 128 - current_time_dim 

    time_padding = (0, padding_needed)
    wfs = F.pad(wfs, time_padding, mode='constant', value=wfs.mean())

    print("After padding:" ,wfs.shape) # Now it is [1183, 128, 128]
    

    # We get the length for now and reshape the wfs to squeeze last 2 dimensions,
    # so we can normalize them 
    length, x, y = wfs.shape

    wfs = wfs.reshape((length, -1)) # wfs = [1183, 16384]

    wfs += 1e-10 
    wfs = torch.log10(wfs)

    wfs_min, wfs_max = wfs.min(), wfs.max()
    scaler = MinMaxScaler()
    wfs = scaler.fit_transform(wfs)

    wfs = wfs.reshape((length, x, y))
    wfs = torch.from_numpy(wfs).float()


    cond_var = np.concatenate(cond_data, axis=1) 
    cond_var = torch.from_numpy(cond_var)

    dataset = WaveformSTFTDataset(wfs, cond_var)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    print("Waveform magnitudes:", dataset.wfs.shape)
    print("Conditional variables:", dataset.cond_var.shape)

    return dataset, dataloader, length, wfs_min, wfs_max

class WaveformSTFTDataset(Dataset):
    def __init__(self, wfs, cond_var):
        # Ensure data is already scaled and in tensor format
        self.wfs= wfs
        self.cond_var = cond_var

    def __len__(self):
        return self.wfs.shape[0]

    def __getitem__(self, idx):
        return self.wfs[idx], self.cond_var[idx]

# dataset, dataloader, length, wfs_min, wfs_max = load_data("data/timeseries_EW.csv", N_FFT, HOP_LENGTH, WIN_LENGTH, WINDOW_TENSOR)




# Plot the spectrograms
def plot_sample_stft(n):
  num_samples_to_plot = n
  sample_indices = random.sample(range(length), num_samples_to_plot)
  for i in sample_indices:
      spectrogram = dataset.wfs[i]

      # Reshape if necessary
      spectrogram = spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1])
      spectrogram = spectrogram.cpu().detach().numpy()

      plt.figure(figsize=(10, 6))
      plt.imshow(spectrogram, origin='lower', aspect='auto', cmap='viridis')
      plt.colorbar(label='Magnitude')
      plt.xlabel('Time Frame')
      plt.ylabel('Frequency Bin')
      plt.title(f"Spectrogram for Sample {i}")
      plt.show()


# plot_sample_stft(5)