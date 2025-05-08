import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging 
import os      

from model import SimpleUnet 
from get_data import load_data 
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# --- 1. Configuration & Parameters ---
N_FFT = 256
HOP_LENGTH = 64
WIN_LENGTH = N_FFT
WINDOW_TENSOR = torch.hann_window(WIN_LENGTH, periodic=True)
DATA_PATH = "data/timeseries_EW.csv"

# Diffusion parameters
TOTAL_DIFFUSION_STEPS = 1000
COND_DIM = 7

# Training parameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_epoch(model, dataloader, optimizer, criterion, total_steps, device):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, data_batch in enumerate(dataloader):
        
        batch_X_0, batch_c = data_batch
        
        batch_X_0 = batch_X_0.to(device)
        batch_c = batch_c.to(device).float() 

        # print("STFT SHAPE",batch_X_0.shape) # [batch_size, 128, 128]
        # print("CONDITIONALS SHAPE",batch_c.shape) # [batch_size, cond_dim]
        
        current_batch_size = batch_X_0.shape[0]

        noise = torch.randn(batch_X_0.shape).to(device)
        timesteps = torch.LongTensor([50]).to(device)
        noisy_stft = noise_scheduler.add_noise(batch_X_0, noise, timesteps)
        

        # --- Model Prediction & Loss ---
        noisy_stft = noisy_stft.unsqueeze(1) # [batch_size, 1, 128, 128]
        # print(noisy_stft.shape)
        predicted_noise = model(noisy_stft, timesteps, batch_c)
        predicted_noise = predicted_noise.squeeze()

        loss = criterion(predicted_noise, noise)

        # --- Backpropagation ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# --- 4. Main Execution ---

if __name__ == "__main__":
    logging.info(f"Using device: {device}")
    logging.info("Loading data...")

    try:
        dataset, dataloader, dataset_length, wf_min, wf_max, norm_dict = load_data(
            DATA_PATH, N_FFT, HOP_LENGTH, WIN_LENGTH, BATCH_SIZE
        )
        logging.info(f"Data loaded. Dataset length: {dataset_length}, Batches: {len(dataloader)}")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        exit(1)


    # --- Model Setup ---
    logging.info("Initializing model...")
    model = SimpleUnet(cond_dim=COND_DIM).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {num_params:,}")


    # --- Optimizer & Loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()


    # --- Training ---
    logging.info("Starting training...")
    for epoch in range(NUM_EPOCHS):
        avg_epoch_loss = train_epoch(
            model, dataloader, optimizer, criterion,
            TOTAL_DIFFUSION_STEPS, device
        )
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_epoch_loss:.4f}")

        # --- Optional: Add evaluation/validation step here ---
        # model.eval()
        # with torch.no_grad():
        #    # Run validation loop  
        #    val_loss = evaluate_epoch(...)
        #    logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {val_loss:.4f}")

        # Save model checkpoint periodically 
        if (epoch + 1) % (NUM_EPOCHS // 10) == 0:
            checkpoint_path = f"checkpoints/model_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'n_fft': N_FFT,
                'diff_steps': TOTAL_DIFFUSION_STEPS,
                'cond_vars': COND_DIM,
                'num_epochs': NUM_EPOCHS,
                'batch_size': BATCH_SIZE,
                'lr': LEARNING_RATE,
                'norm_dict': norm_dict     
            }, checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")

    logging.info("Training complete.")