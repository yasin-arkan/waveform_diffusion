import os 
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
import numpy as np
# Assuming you are in an environment like Jupyter for audio display
from IPython.display import Audio, display

# Assuming these modules contain the necessary classes/functions
from model import SimpleUnet # Contains the SimpleUnet class definition
from noise_linear import linear_beta_schedule # Assumed function to get schedules
from get_data import load_data

# --- 1. Configuration (Should match training) ---
N_FFT = 256
HOP_LENGTH = 64
WIN_LENGTH = N_FFT
WINDOW_TENSOR = torch.hann_window(WIN_LENGTH, periodic=True)
TARGET_SHAPE = (128, 128) # H, W of the spectrogram
COND_DIM = 7
TOTAL_DIFFUSION_STEPS = 1000
NOISE_SCHEDULE_TYPE = 'linear'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 7000 # IMPORTANT: Specify the sample rate of your original audio

conditional_values = [40.4328, 29.1212, 40.5683, 28.8660, 6.6, 2.3, 26.3507] # These are values of station and source latitude and longitudes, plus phase, magnitude and rupture. 


# --- 2. Load Trained Model ---
logging.info("Loading trained model...")
MODEL_CHECKPOINT_PATH = "checkpoints/model_epoch_100.pth" # <<< CHANGE TO YOUR CHECKPOINT PATH

dataset, dataset_length, wf_min, wf_max = load_data(
            "data/timeseries_EW.csv", N_FFT, HOP_LENGTH, WIN_LENGTH, WINDOW_TENSOR
        )

model = SimpleUnet(cond_dim=COND_DIM).to(DEVICE)

try:
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
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

# --- 3. Prepare Noise Schedules ---
# Recalculate schedules needed for sampling (betas, alphas, alphas_cumprod)
betas = linear_beta_schedule(TOTAL_DIFFUSION_STEPS).to(DEVICE)
alphas = (1. - betas).to(DEVICE)
alphas_cumprod = torch.cumprod(alphas, dim=0).to(DEVICE)
# Schedule needed for the reverse formula's noise term coefficient
sqrt_one_minus_alpha_bar = torch.sqrt(1. - alphas_cumprod).to(DEVICE)
# Variance term for DDPM sampling step (simplest choice)
posterior_variance = betas * (1. - torch.cat([torch.tensor([0.0], device=DEVICE), alphas_cumprod[:-1]])) / (1. - alphas_cumprod)
# Avoid division by zero at step 0 - use beta_tilde (clipped beta)
posterior_log_variance_clipped = torch.log(torch.maximum(posterior_variance, torch.tensor(1e-20, device=DEVICE)))


# --- 4. Sampling Function (Reverse Diffusion) ---

@torch.no_grad() # Disable gradient calculations for inference
def sample(model, num_samples, condition_vectors, schedules, device, target_shape):
    """Generates samples using the DDPM reverse process."""
    (betas, alphas, alphas_cumprod, sqrt_one_minus_alpha_bar, posterior_log_variance_clipped) = schedules
    img_channels = 1 # Should match model's out_dim / input dim channel
    img_height, img_width = target_shape

    # Start with pure Gaussian noise
    x_t = torch.randn(num_samples, img_channels, img_height, img_width, device=device)
    condition_vectors = condition_vectors.to(device).float() # Ensure condition is on device

    logging.info(f"Starting sampling loop for {num_samples} samples...")
    for t_int in reversed(range(1, TOTAL_DIFFUSION_STEPS + 1)):
        t = torch.full((num_samples,), t_int - 1, device=device, dtype=torch.long) # t-1 for 0-indexing

        # Predict noise using the model
        predicted_noise = model(x_t, t, condition_vectors)

        # Get schedule parameters for current t
        beta_t = betas[t].view(num_samples, 1, 1, 1)
        alpha_t = alphas[t].view(num_samples, 1, 1, 1)
        alpha_bar_t = alphas_cumprod[t].view(num_samples, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar[t].view(num_samples, 1, 1, 1)
        log_variance_t = posterior_log_variance_clipped[t].view(num_samples, 1, 1, 1)

        # Calculate x_{t-1} mean
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise)

        # Add noise term for stochasticity (DDPM step)
        if t_int > 1:
            noise = torch.randn_like(x_t)
            variance = torch.exp(0.5 * log_variance_t) # sigma_t
            x_t_prev = mean + variance * noise
        else:
            # Last step, no noise added
            x_t_prev = mean

        x_t = x_t_prev # Update for next iteration

        # Optional: Log progress
        # if t_int % 100 == 0:
        #     logging.info(f"  Sampling step {t_int}/{TOTAL_DIFFUSION_STEPS}")

    logging.info("Sampling complete.")
    # x_t now holds the predicted x_0 (clean data)
    # Output might be scaled [-1, 1] or [0, 1] depending on training data scaling.
    # Assume output is the linear magnitude spectrogram for now.
    # Remove channel dimension for STFT processing: [B, 1, H, W] -> [B, H, W]
    return x_t.squeeze(1)


# --- 5. Waveform Reconstruction (Griffin-Lim) ---

def reconstruct_waveform(magnitude_spectrogram, n_fft, hop_length, win_length, window):
    """Reconstructs waveform from magnitude spectrogram using Griffin-Lim."""
    logging.info("Reconstructing waveform using Griffin-Lim...")
    # Ensure spectrogram is on CPU and detach if needed
    magnitude_spectrogram = magnitude_spectrogram.cpu().detach()

    # Griffin-Lim expects power, but often works okay with magnitude directly.
    # For better results, power = magnitude**2, but let's try magnitude first.
    # power_spectrogram = magnitude_spectrogram.pow(2)

    griffin_lim_transform = T.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=1.0, # Power to raise magnitude before GL (1.0 = use magnitude)
        n_iter=32  # Number of iterations
    ).cpu()

    waveform = griffin_lim_transform(magnitude_spectrogram)
    logging.info("Waveform reconstruction complete.")
    return waveform


# --- 6. Visualization ---

def visualize_results(spectrogram, waveform, sr):
    """Displays the spectrogram and waveform, plays audio."""
    # Ensure data is on CPU and converted to numpy for plotting
    spec_np = spectrogram.cpu().detach().numpy()
    wf_np = waveform.cpu().detach().numpy()

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot Spectrogram
    axs[0].set_title("Generated STFT Magnitude Spectrogram")
    im = axs[0].imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, ax=axs[0], format="%+2.0f dB") # Assuming magnitude needs interpretation
    axs[0].set_ylabel("Frequency Bins")
    axs[0].set_xlabel("Time Frames")

    # Plot Waveform
    axs[1].set_title("Reconstructed Waveform")
    time_axis = np.linspace(0, len(wf_np), num=len(wf_np))
    axs[1].plot(time_axis, wf_np)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

    # Play Audio (in compatible environments like Jupyter)
    print("Generated Audio:")
    display(Audio(wf_np, rate=sr))

def save_waveform_as_png(waveform, sr, file_path):
    """Plots the waveform and saves it as a PNG file."""
    logging.info(f"Saving waveform plot to {file_path}...")
    wf_np = waveform.cpu().detach().numpy()

    plt.figure(figsize=(12, 4)) # Create a new figure
    time_axis = np.linspace(0, len(wf_np) / sr, num=len(wf_np))
    plt.plot(time_axis, wf_np)
    plt.title("Generated Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.ylim(-1, 1) # Optional: Set consistent y-axis limits
    plt.grid(True)
    plt.savefig(file_path, format='png', bbox_inches='tight', dpi=150) # Save the figure
    plt.close() # Close the figure to free memory
    logging.info("Waveform plot saved.")


# --- 7. Main Sampling Execution ---

if __name__ == "__main__":
    print("Start")
    NUM_SAMPLES_TO_GENERATE = 1 # Generate one sample

    # --- Prepare Conditioning Data ---
    # Create or load your condition vector(s) here
    # Example: Using a random vector
    # Ensure it has shape [NUM_SAMPLES_TO_GENERATE, COND_DIM]
    example_condition = np.array([conditional_values])
    example_condition = torch.from_numpy(example_condition)
    logging.info(f"Using example condition vector shape: {example_condition.shape}")

    # --- Collect Schedules ---
    schedules = (betas, alphas, alphas_cumprod, sqrt_one_minus_alpha_bar, posterior_log_variance_clipped)

    # --- Generate Sample(s) ---
    generated_spectrograms = sample(
        model,
        NUM_SAMPLES_TO_GENERATE,
        example_condition,
        schedules,
        DEVICE,
        TARGET_SHAPE
    ) # Output shape: [num_samples, H, W]

    output_dir = "generated_waveforms" # Directory to save plots
    os.makedirs(output_dir, exist_ok=True) 

    # --- Reconstruct and Visualize each sample ---
    for i in range(NUM_SAMPLES_TO_GENERATE):
        logging.info(f"--- Processing Sample {i+1}/{NUM_SAMPLES_TO_GENERATE} ---")
        spec = generated_spectrograms[i] # Shape [H, W]

        padding = (0, 0, 0, 1) # (pad time_left, pad time_right, pad freq_top, pad freq_bottom)
        spec_padded = F.pad(spec, padding, mode='constant', value=0)
        # spec_padded shape is now [129, 128]
        logging.info(f"Padded spectrogram shape for ISTFT: {spec_padded.shape}")

        spec_padded = spec_padded * (wf_max - wf_min) + wf_min

        waveform = reconstruct_waveform(
            spec_padded, N_FFT, HOP_LENGTH, WIN_LENGTH, torch.hann_window
        ) # Output shape: [num_audio_samples]

        output_filename = os.path.join(output_dir, f"waveform_sample_{i+1}.png")
        save_waveform_as_png(waveform, SAMPLE_RATE, output_filename)

        visualize_results(spec, waveform, SAMPLE_RATE)