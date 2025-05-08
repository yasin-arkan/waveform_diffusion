import torch 
import torchaudio
import logging
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from model import SimpleUnet
from cond_model import SimpleNN
from get_data import load_data

import obspy 
import librosa

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

N_FFT = 256      
HOP_LENGTH = 64  
WIN_LENGTH = N_FFT 

dataset, dataloader, length, wfs_mean, wfs_std, norm_dict = load_data("data/timeseries_EW.csv", N_FFT, HOP_LENGTH, WIN_LENGTH)

MODEL_CHECKPOINT_PATH = "checkpoints/model_epoch_100.pth"
COND_CHECKPOINT_PATH = "checkpoints/cond_model_epoch_2000.pth"

model_main = SimpleUnet(cond_dim=7).to(device)
model_cond = SimpleNN().to(device)


checkpoint_cond = torch.load(COND_CHECKPOINT_PATH, map_location=device)
model_cond.load_state_dict(checkpoint_cond['model_state_dict'], strict=False)
model_cond.eval()


EventLat = 40.4328
EventLon = 29.1212

StationLat = 40.5683
StationLon = 28.8660 

EventLat = (EventLat - norm_dict['EventLat'][0]) - norm_dict['EventLat'][1]
EventLon = (EventLon - norm_dict['EventLon'][0]) - norm_dict['EventLon'][1]
StationLat = (StationLat - norm_dict['StationLat'][0]) - norm_dict['StationLat'][1]
StationLon = (StationLon - norm_dict['StationLon'][0]) - norm_dict['StationLon'][1]

source_loc = [EventLat, EventLon]
station_loc = [StationLat, StationLon]


loc = np.array(source_loc + station_loc)

loc = torch.Tensor(loc).float().to(device)

c = model_cond(loc) # model predicts depth, magnitude and rupture distance 


condition_vec = torch.cat((loc, c)).to(device)


print(condition_vec)


try:
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device, weights_only=False)
    # Strict=False might be needed if checkpoint has extra keys (e.g., optimizer state)
    model_main.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model_main.eval() 
    logging.info(f"Model loaded successfully from {MODEL_CHECKPOINT_PATH}")
except FileNotFoundError:
    logging.error(f"Model checkpoint not found at {MODEL_CHECKPOINT_PATH}. Cannot perform sampling.")
    exit(1)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)








@torch.no_grad()
def sample_plot_image():
    # Sample noise
    spec_noise = torch.randn((1, 1, 128, 128), device=device)
    timesteps = torch.LongTensor([1000]).to(device)

    pred = model_main(spec_noise, timesteps, condition_vec)

    spec = pred.squeeze()
    # spec = (spec * wfs_std) + wfs_mean
    
    spec = spec.cpu().numpy()

    wf = librosa.griffinlim(spec)

    wf = wf.flatten()

    trace = obspy.Trace(data=wf)
    trace.stats.sampling_rate = 100

    stream = obspy.Stream([trace])
    stream.plot()

    plt.figure(figsize=(9, 6))
    plt.imshow(spec, origin='lower', aspect='auto', cmap='viridis')
    plt.xlabel('Time Frame')
    plt.ylabel('Frequency Bin')
    plt.title(f"Generated spectrogram")
            
    plt.show()   
           

sample_plot_image()   