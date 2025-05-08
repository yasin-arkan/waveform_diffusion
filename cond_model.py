from matplotlib.pyplot import xlabel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from get_data import load_data

from sklearn.model_selection import train_test_split

INPUT_DIM = 4  
OUTPUT_DIM = 3 
HIDDEN_LAYER_SIZE = 64 
EPOCHS = 2000       
LEARNING_RATE = 0.001


class SimpleNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3):
        super(SimpleNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x




if __name__ == "__main__":

  dataset, dataloader, length, wfs_mean, wfs_std, norm_dict = load_data("data/timeseries_EW.csv",
                                                           256, 64,
                                                            256, 
                                                            batch_size=32)


  print("SHAPE:", dataset.cond_var.shape)

  X = dataset.cond_var[:, :4]
  y = dataset.cond_var[:, 4:]

  length = X.shape[0]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  X_train = X_train.float()
  X_test = X_test.float()
  y_train = y_train.float()
  y_test = y_test.float()


  model = SimpleNN(INPUT_DIM, HIDDEN_LAYER_SIZE, OUTPUT_DIM)

  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

  model.train()

  for epoch in range(EPOCHS):
      outputs = model(X_train)

      loss = criterion(outputs, y_train)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if (epoch + 1) % 100 == 0:
          print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
      
            
  checkpoint_path = f"checkpoints/cond_model_epoch_{epoch + 1}.pth"

  torch.save({
              'epoch': epoch + 1,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),  
              }, checkpoint_path)
  print("Training finished.")



# model.eval()

# x = [40.4328, 29.1212, 40.5683, 28.8660]

# x = torch.Tensor(x).float()

# with torch.no_grad():
#     predictions = model(x)
#     pred = model(X_test)

# print(predictions)

# final_loss = criterion(pred, y_test)
# print(final_loss)


