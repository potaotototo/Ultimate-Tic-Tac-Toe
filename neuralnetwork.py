import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import load_data  # Ensure this is available

# Load Data
data = load_data()

# Convert State objects into NumPy arrays (flattened board)
X = np.array([state.board.flatten() for state, utility in data], dtype=np.float32)
y = np.array([utility for state, utility in data], dtype=np.float32).reshape(-1, 1)  # Reshape for PyTorch

# Convert to PyTorch tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# Define Neural Network
class TicTacToeNN(nn.Module):
    def __init__(self):
        super(TicTacToeNN, self).__init__()
        self.fc1 = nn.Linear(81, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss, and optimizer
model = TicTacToeNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "tictactoe_nn.pth")
print("Model trained and saved!") # Epoch [999/1000], Loss: 0.1274
