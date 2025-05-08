import torch
import torchaudio
import torch.nn.functional as F
import obspy
import numpy as np
import pandas as pd 
import random 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import librosa 


N_FFT = 256      
HOP_LENGTH = 64  
WIN_LENGTH = N_FFT 

file_path = "data/timeseries_EW.csv"

def load_data(path, n_fft, hop_length, win_length, batch_size=32):
    data = pd.read_csv(path)

    cond_cols = ['EventLat', 'EventLon', 'StationLat', 'StationLon', 'Depth', 'Magnitude', 'RuptureDist_km']
    cond_vars = data[cond_cols]

    cond_data = []
    norm_dict = {}


    for cvar in cond_cols:
        cv = cond_vars[cvar].to_numpy()
        cv = cv.reshape(cv.shape[0], 1)
        
        cv_mean, cv_std = cv.mean(), cv.std()
        norm_dict[cvar] = [cv_mean, cv_std]

        cv = (cv - cv_mean) / cv_std

        cond_data.append(cv)
    
    
    wfs = data.iloc[:, 16:].to_numpy()

    wfs = librosa.stft(wfs, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    wfs = np.abs(wfs)
    wfs = torch.from_numpy(wfs).float()
    # Magnitude with shape [1183, 129, 110]

    wfs = wfs[:, :128, :] # Discarding the last frequency to make it 128

    current_time_dim = wfs.shape[2] # Should be 110, we will pad it to 128
    padding_needed = 128 - current_time_dim 
    time_padding = (0, padding_needed)
    wfs = F.pad(wfs, time_padding, mode='constant', value=wfs.median())
    print("After padding:" ,wfs.shape) # Now it is [1183, 128, 128]
    

    # We get the length for now and reshape the wfs to squeeze last 2 dimensions,
    # so we can normalize them 
    length, x, y = wfs.shape

    wfs = wfs.reshape((length, -1)) # wfs = [1183, 16384]
 
    # wfs += 1e-10
    # wfs = torch.log10(wfs)

    wfs_mean, wfs_std = wfs.mean(), wfs.std() # we need these values later 
    wfs = (wfs - wfs_mean) / wfs_std
    wfs = wfs.reshape((length, x, y))

    

    cond_var = np.concatenate(cond_data, axis=1) 
    cond_var = torch.from_numpy(cond_var)

    dataset = WaveformSTFTDataset(wfs, cond_var)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Waveform shape:", dataset.wfs.shape)
    print("Cond shape:", dataset.cond_var.shape)
    
    return dataset, dataloader, length, wfs_mean, wfs_std, norm_dict




class WaveformSTFTDataset(Dataset):
    def __init__(self, wfs, cond_var):
        # Ensure data is already scaled and in tensor format
        self.wfs = wfs
        self.cond_var = cond_var

    def __len__(self):
        return self.wfs.shape[0]

    def __getitem__(self, idx):
        return self.wfs[idx], self.cond_var[idx]

if __name__ == "__main__":

  dataset, dataloader, length, wfs_mean, wfs_std, norm_dict = load_data(file_path,
                                                                        N_FFT,
                                                                        HOP_LENGTH,
                                                                        WIN_LENGTH,
                                                                        batch_size=32)
  # Plot the spectrograms 
  def plot_sample_stft():

    index = random.randint(0, 1000)
    spectrogram = dataset.wfs[index].cpu().numpy()
    print(spectrogram.shape)

    # Reshape if necessary
    # spectrogram = spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1])
    # spectrogram = spectrogram.cpu().detach().numpy()
    tf = librosa.griffinlim(spectrogram, n_iter=256)

    # tf = torchaudio.transforms.GriffinLim(N_FFT, 256, WIN_LENGTH, HOP_LENGTH, power=1)
    # waveform = tf(spectrogram)
    # wf = waveform.numpy()

    trace = obspy.Trace(data=tf)
    trace.stats.sampling_rate = 100

    
        
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram, origin='lower', aspect='auto', cmap='viridis')
    plt.xlabel('Time Frame')
    plt.ylabel('Frequency Bin')
    plt.title(f"Spectrogram for Sample {index}")

    stream = obspy.Stream([trace])
    stream.plot()


  plot_sample_stft()