import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging # Added for better logging
import os      # Added for potential future model saving

# Assuming these modules contain the necessary classes/functions
from model import SimpleUnet # Contains the SimpleUnet class definition
from noise_linear import linear_beta_schedule # Assumed function to get schedules
from get_data import load_data # Assumed function that returns Dataset, length

# --- 1. Configuration & Parameters ---
# Data/STFT parameters
N_FFT = 256
HOP_LENGTH = 64 # Check if this gives the expected time frames (e.g., 128)
WIN_LENGTH = N_FFT
WINDOW_TENSOR = torch.hann_window(WIN_LENGTH, periodic=True) # Usually not needed directly in training script
DATA_PATH = "data/timeseries_EW.csv"

# Diffusion parameters
TOTAL_DIFFUSION_STEPS = 1000
NOISE_SCHEDULE_TYPE = 'linear' # Example: 'linear', 'cosine'

# Model parameters
COND_DIM = 7 # Dimension of the conditioning vector

# Training parameters
NUM_EPOCHS = 100
BATCH_SIZE = 16 
LEARNING_RATE = 1e-5 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. Utility Functions (Optional but good practice) ---

def setup_noise_schedule(total_steps, schedule_type, device):
    """Calculates and returns noise schedule tensors on the specified device."""
    logging.info(f"Setting up {schedule_type} noise schedule for {total_steps} steps...")
    # Assuming get_noise_schedule returns betas
    betas = linear_beta_schedule(total_steps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    sqrt_alpha_bar = torch.sqrt(alphas_cumprod).to(device).float()
    sqrt_one_minus_alpha_bar = torch.sqrt(1. - alphas_cumprod).to(device).float()
    logging.info("Noise schedule setup complete.")
    return sqrt_alpha_bar, sqrt_one_minus_alpha_bar

def prepare_batch_input(batch_X_0, device):
    """Ensures batch data is on the correct device and has the channel dimension."""
    batch_X_0 = batch_X_0.to(device).float()
    # Ensure input has channel dim [B, 1, H, W]
    if batch_X_0.ndim == 3:
        batch_X_0 = batch_X_0.unsqueeze(1) # Add channel dim
    elif batch_X_0.ndim != 4 or batch_X_0.shape[1] != 1:
        raise ValueError(f"Unexpected batch shape received: {batch_X_0.shape}. Expected [B, H, W] or [B, 1, H, W].")
    return batch_X_0

# --- 3. Training Epoch Function ---

def train_epoch(model, dataloader, optimizer, criterion, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, total_steps, device):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, data_batch in enumerate(dataloader):
        # --- Data Preparation ---
        # Adjust unpacking based on what your dataloader actually yields
        # Assuming format: (processed_magnitude_data, condition_vector)
        batch_X_0_raw, batch_c, true, original = data_batch
        
        # Ensure data has correct shape [B, 1, H, W] and is on device
        batch_X_0 = prepare_batch_input(batch_X_0_raw, device)
        batch_c = batch_c.to(device).float() # Assuming batch_c is [B, COND_DIM]
        
        current_batch_size = batch_X_0.shape[0]

        # --- Diffusion Forward Process ---
        # 1. Sample time steps
        t = torch.randint(1, total_steps + 1, (current_batch_size,), device=device).long()

        # 2. Get schedule values for t (already on device)
        sqrt_alpha_bar_t = sqrt_alpha_bar[t - 1]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar[t - 1]

        # 3. Generate noise (matching the target shape [B, 1, H, W])
        noise = torch.randn_like(batch_X_0, device=device)

        # 4. Compute noisy data X_t
        # Reshape schedules for broadcasting: [B,] -> [B, 1, 1, 1]
        sqrt_alpha_bar_t_r = sqrt_alpha_bar_t.view(current_batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t_r = sqrt_one_minus_alpha_bar_t.view(current_batch_size, 1, 1, 1)
        batch_X_t = sqrt_alpha_bar_t_r * batch_X_0 + sqrt_one_minus_alpha_bar_t_r * noise

        # --- Model Prediction & Loss ---
        # Note: Removed unsqueeze for batch_X_t as prepare_batch_input handles shape
        predicted_noise = model(batch_X_t, t, batch_c)

        # Ensure noise and predicted_noise have the same shape ([B, 1, H, W])
        # Note: noise generated via randn_like(batch_X_0) already has the correct shape
        loss = criterion(predicted_noise, noise)

        # --- Backpropagation ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Optional: Log batch loss periodically
        # if batch_idx % 50 == 0:
        #     logging.info(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# --- 4. Main Execution ---

if __name__ == "__main__":
    logging.info(f"Using device: {DEVICE}")

    # --- Data Loading ---
    # Modify load_data if needed to return Dataset object and accept target_shape
    logging.info("Loading data...")
    try:
        # Assuming load_data now processes STFT, gets magnitude, handles shapes (128, 128), and returns (Dataset, condition_data_tensor)
        # Modify based on your actual load_data implementation
        dataset, dataset_length, wf_min, wf_max = load_data(
            DATA_PATH, N_FFT, HOP_LENGTH, WIN_LENGTH, WINDOW_TENSOR
        )
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if DEVICE=='cuda' else False)
        logging.info(f"Data loaded. Dataset length: {dataset_length}, Batches: {len(dataloader)}")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        exit(1)


    # --- Model Setup ---
    logging.info("Initializing model...")
    model = SimpleUnet(cond_dim=COND_DIM).to(DEVICE)
    # Optional: Print model summary or parameter count
    # print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {num_params:,}")


    # --- Optimizer & Loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()


    # --- Noise Schedule ---
    sqrt_alpha_bar, sqrt_one_minus_alpha_bar = setup_noise_schedule(
        TOTAL_DIFFUSION_STEPS, NOISE_SCHEDULE_TYPE, DEVICE
    )


    # --- Training ---
    logging.info("Starting training...")
    for epoch in range(NUM_EPOCHS):
        avg_epoch_loss = train_epoch(
            model, dataloader, optimizer, criterion,
            sqrt_alpha_bar, sqrt_one_minus_alpha_bar,
            TOTAL_DIFFUSION_STEPS, DEVICE
        )
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_epoch_loss:.4f}")

        # --- Optional: Add evaluation/validation step here ---
        # model.eval()
        # with torch.no_grad():
        #    # Run validation loop
        #    val_loss = evaluate_epoch(...)
        #    logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {val_loss:.4f}")

        # --- Optional: Save model checkpoint periodically ---
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")

    logging.info("Training complete.")