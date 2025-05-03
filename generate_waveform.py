import torch 
import torchaudio
import logging
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from model import SimpleUnet

import obspy 

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

N_FFT = 256      
HOP_LENGTH = 64  
WIN_LENGTH = N_FFT 
WINDOW_TENSOR = torch.hann_window(window_length=WIN_LENGTH, periodic=True)

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


MODEL_CHECKPOINT_PATH = "checkpoints/model_epoch_50.pth"
T = 300 
betas = linear_beta_schedule(300)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)




model = SimpleUnet(cond_dim=4).to(device)

condition_vec = np.array([40.4328, 29.1212, 40.5683, 28.8660])
condition_vec = torch.from_numpy(condition_vec).float().to(device)


try:
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
    # Strict=False might be needed if checkpoint has extra keys (e.g., optimizer state)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval() # Set model to evaluation mode (important!)
    logging.info(f"Model loaded successfully from {MODEL_CHECKPOINT_PATH}")
except FileNotFoundError:
    logging.error(f"Model checkpoint not found at {MODEL_CHECKPOINT_PATH}. Cannot perform sampling.")
    exit(1)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """

    betas_t = get_index_from_list(betas, t, x.shape).to(device)
    
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape).to(device)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, condition_vec) / sqrt_one_minus_alphas_cumprod_t
    ).to(device)

    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape).to(device)
  

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img_size = 128
    spec_noise = torch.randn((1, 1, img_size, img_size), device=device)
    num_images = 1
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        spec = sample_timestep(spec_noise, t)
        spec = spec.squeeze()

        tf = torchaudio.transforms.GriffinLim(255, 255, 255, HOP_LENGTH, window_fn=torch.hann_window ,power=1).to(device)
        
        if i % stepsize == 0:
            waveform = tf(spec)
            wf = waveform.cpu().numpy()

            wf = wf.flatten()

            trace = obspy.Trace(data=wf)
            trace.stats.sampling_rate = 100

            stream = obspy.Stream([trace])
            stream.plot()

            spec = spec.cpu()

            plt.figure(figsize=(9, 6))
            plt.imshow(spec, origin='lower', aspect='auto', cmap='viridis')
            plt.xlabel('Time Frame')
            plt.ylabel('Frequency Bin')
            plt.title(f"Spectrogram for Sample {i}")
            
    plt.show()   
           

sample_plot_image()   