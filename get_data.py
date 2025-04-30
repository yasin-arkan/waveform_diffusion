import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


N_FFT = 256       
HOP_LENGTH = 64  
WIN_LENGTH = N_FFT 
WINDOW_TENSOR = torch.hann_window(WIN_LENGTH, periodic=True)

def load_data(path, n_fft, hop_length, win_length ,window):
    data = pd.read_csv(path)

    cond_cols = ['EventLat', 'EventLon', 'StationLat', 'StationLon', 'Depth', 'Magnitude', 'RuptureDist_km']
    cond_vars = data[cond_cols]

    cond_data = []


    for cvar in cond_cols:
        cv = cond_vars[cvar].to_numpy()
        cv = cv.reshape(cv.shape[0], 1)
        scaler = MinMaxScaler()
        cv = scaler.fit_transform(cv)
        cond_data.append(cv)
    
    
    wfs = data.iloc[:, 16:].to_numpy()
    orig_wfs = torch.from_numpy(wfs).float()

    wfs = torch.stft(orig_wfs, n_fft, hop_length, win_length, window, return_complex=True)

    print("Before padding:", wfs.shape)

    wfs = wfs[:, :128, :] # Discarding the last frequency to make it 128 

    current_time_dim = wfs.shape[2] # Should be 110, we will pad it to 128
    target_time_dim = 128
    padding_needed = target_time_dim - current_time_dim # Should be 18

    time_padding = (0, padding_needed)
    wfs = F.pad(wfs, time_padding, mode='constant', value=0)

    print("After padding:" ,wfs.shape)
    true_phase, wfs = get_phase_mag(wfs)

    length, x, y = wfs.shape

    wfs = wfs.reshape((length, -1))

    wfs_min, wfs_max = wfs.min(), wfs.max()

    print(wfs_min, wfs_max)

    scaler = MinMaxScaler()
    wfs = scaler.fit_transform(wfs)

    wfs = wfs.reshape((length, x, y))
    wfs = torch.from_numpy(wfs).float()

    cond_var = np.concatenate(cond_data, axis=1) 
    cond_var = torch.from_numpy(cond_var)

    dataset = WaveformSTFTDataset(wfs, cond_var, true_phase, orig_wfs)

    print("Waveform magnitudes:", dataset.wfs.shape)
    print("Conditional variables:", dataset.cond_var.shape)
    print("True phase of waveform:",dataset.true_phase.shape)
    print("Original waveform:", dataset.orig_wfs.shape)

    return dataset, length, wfs_min, wfs_max


def get_phase_mag(wfs):
    phase = torch.angle(wfs)
    magnitude = torch.abs(wfs)
    return phase, magnitude

class WaveformSTFTDataset(Dataset):
    def __init__(self, wfs, cond_var, true_phase, orig_wfs):
        # Ensure data is already scaled and in tensor format
        self.wfs= wfs
        self.cond_var = cond_var
        self.true_phase = true_phase
        self.orig_wfs = orig_wfs

    def __len__(self):
        return self.wfs.shape[0]

    def __getitem__(self, idx):
        return self.wfs[idx], self.cond_var[idx], self.true_phase[idx], self.orig_wfs[idx]

load_data("data/timeseries_EW.csv", N_FFT, HOP_LENGTH, WIN_LENGTH, WINDOW_TENSOR)

