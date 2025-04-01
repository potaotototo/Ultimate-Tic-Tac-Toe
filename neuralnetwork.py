import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle  # To load data
from utils import load_data

### Feature Extraction Using NumPy ###
def extract_features_numpy(boards, current_players):
    """
    Extracts features from multiple super Tic-Tac-Toe boards using NumPy.
    
    Args:
        boards (numpy array of shape (N,3,3,3,3)): The batch of game states.
        current_players (numpy array of shape (N,)): Players whose turn it is (1 or -1).
    
    Returns:
        numpy array of shape (N, 91): Extracted features.
    """
    N = boards.shape[0]  # Number of board states
    
    # Flatten local board features: shape (N, 81)
    local_features = boards.reshape(N, -1)

    # Compute global board status (9 values per board)
    wins_p1 = np.all(boards == 1, axis=(-2, -1))  # Player 1 wins sub-board
    wins_p2 = np.all(boards == -1, axis=(-2, -1))  # Player 2 wins sub-board
    global_features = (wins_p1.astype(int) - wins_p2.astype(int)).reshape(N, -1)  # (N, 9)

    # Expand current player turn feature to shape (N, 1)
    turn_features = current_players[:, np.newaxis]  

    # Concatenate all features (shape: N × 91)
    return np.hstack([local_features, global_features, turn_features])


### Neural Network Model ###
class TicTacToeNN(nn.Module):
    def __init__(self):
        super(TicTacToeNN, self).__init__()
        self.fc1 = nn.Linear(91, 256)  # Input: 91 features (81 local, 9 global, 1 turn)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)   # Output: Utility value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


### Train the Neural Network ###
def train_model(features, utilities, epochs=1000, lr=0.001):
    model = TicTacToeNN()
    criterion = nn.MSELoss()  # Mean squared error for regression
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert to PyTorch tensors
    X_train = torch.tensor(features, dtype=torch.float32)
    y_train = torch.tensor(utilities, dtype=torch.float32).view(-1, 1)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    print("Model trained!")
    return model


### Load Data and Train ###
if __name__ == "__main__":
    data = load_data()

    # Extract raw board states and corresponding utilities
    boards = np.array([state.board for state, utility in data])  # Shape: (80000,3,3,3,3)
    current_players = np.array([state.fill_num for state, utility in data])  # Shape: (80000,)
    utilities = np.array([utility for state, utility in data])  # Shape: (80000,)

    # Convert game states into features
    features = extract_features_numpy(boards, current_players)

    # Train the model using extracted features
    model = train_model(features, utilities)
