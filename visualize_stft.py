import torch
import torchaudio
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt 

data = pd.read_csv("data/timeseries_EW.csv")

# This file is for visualizing a single waveform's STFT before any of the normalization and training


WAVEFORM_INDEX = 0

info_cols = data.columns[:16]
print(info_cols)
info = data[info_cols] 
'''First 16 columns: RecordID, StationID, EventID, EventLat, EventLon,
                        StationLat, StationLon, Depth, Magnitude, RuptureDist_km,
                        SamplingRate, NumTimeSteps, DeltaT, MinFreq, MaxFreq, NumFreqSteps'''

wfs = data.iloc[:, 16:].to_numpy()

orig_wfs = torch.from_numpy(wfs).float()

waveform_to_plot = orig_wfs[WAVEFORM_INDEX, :]

SAMPLING_RATE = 100  # Hz
WAVEFORM_LENGTH = waveform_to_plot.shape[0] # Should be 7000

N_FFT = 256       # Number of FFT points
HOP_LENGTH = 64   # Number of samples between successive frames
WIN_LENGTH = N_FFT # Window size (often same as N_FFT)
# Use periodic=True for standard Hanning window
WINDOW_TENSOR = torch.hann_window(WIN_LENGTH, periodic=True)


wf_spec_single = torch.stft(waveform_to_plot,
                            n_fft=N_FFT,
                            hop_length=HOP_LENGTH,
                            win_length=WIN_LENGTH,
                            window=WINDOW_TENSOR,
                            return_complex=True)

print(f"Shape of STFT for single waveform: {wf_spec_single.shape}")

frequencies = np.linspace(0, SAMPLING_RATE / 2, N_FFT // 2 + 1)
num_time_frames = wf_spec_single.shape[1]
times = np.arange(num_time_frames) * HOP_LENGTH / SAMPLING_RATE
spectrogram_magnitude = torch.abs(wf_spec_single)


# --- Plotting ---
plt.figure(figsize=(12, 6)) # Adjusted figure size for better aspect ratio

# Use imshow with extent to correctly label axes
plt.imshow(spectrogram_magnitude,
           extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
           aspect='auto',          # Prevent stretching
           origin='lower',         # Put frequency=0 at the bottom
           cmap='viridis')         # Viridis is a perceptually uniform colormap

plt.colorbar(label='Magnitude') # Label the color bar
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('STFT Spectrogram')
plt.grid(True, which='both', axis='both', linestyle='--', color='gray', alpha=0.2) # Add a faint grid
plt.tight_layout()
plt.show()


def plot_stft(wf_stft):
    pass





'''
class TimeSpecConverter:
    def __init__(self, n_fft, w_len, h_len, power, device, n_iter=50):
        self.n_fft = n_fft
        self.w_len = w_len
        self.h_len = h_len
        self.power = power
        self.n_iter = n_iter
        self.device = device
        self.griffinlim = torchaudio.transforms.GriffinLim(n_fft=self.n_fft, n_iter=1000, win_length=self.w_len, 
                                                            hop_length=self.h_len, power=self.power).to(self.device)
            
    def time_to_spec(self, wfs):
        return torch.stft(wfs, n_fft=self.n_fft, hop_length=self.h_len, win_length=self.w_len, return_complex=True)
        
    def spec_to_time(self, wfs):
        return torch.istft(wfs, n_fft=self.n_fft, hop_length=self.h_len, win_length=self.w_len)
        
    def griffinlim(self, wfs):
        return self.griffinlim(wfs)

'''