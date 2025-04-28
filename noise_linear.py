import torch
import numpy as np

from get_data import load_data
import matplotlib.pyplot as plt 
import model 

# --- Linear Noise Scheduler Implementation ---

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, timesteps)
    return betas

# --- Configuration ---
TOTAL_DIFFUSION_STEPS = 1000 # T - Choose your total number of diffusion steps
LINEAR_BETA_START = 0.0001    # beta_start - Typical starting beta
LINEAR_BETA_END = 0.02        # beta_end - Typical ending beta


