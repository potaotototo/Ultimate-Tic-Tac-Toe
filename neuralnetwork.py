import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import load_data
from sklearn.model_selection import train_test_split

def extract_features_numpy(boards, current_players):
    N = boards.shape[0]
    local_features = boards.reshape(N, -1)
    wins_p1 = np.all(boards == 1, axis=(-2, -1))
    wins_p2 = np.all(boards == -1, axis=(-2, -1))
    global_features = (wins_p1.astype(int) - wins_p2.astype(int)).reshape(N, -1)
    turn_features = current_players[:, np.newaxis]
    return np.hstack([local_features, global_features, turn_features])

class TicTacToeNN(nn.Module):
    def __init__(self):
        super(TicTacToeNN, self).__init__()
        self.fc1 = nn.Linear(91, 384)
        self.fc2 = nn.Linear(384, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

def train_model(features, utilities, epochs=1000, lr=0.001, patience=200):
    model = TicTacToeNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(utilities, dtype=torch.float32).view(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)

    best_loss = float('inf')
    best_model = None
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

        if test_loss.item() < best_loss - 2e-5:
            best_loss = test_loss.item()
            best_model = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch} with best test loss: {best_loss:.4f}")
                break

    if best_model:
        model.load_state_dict(best_model)

    print("Model trained!")

    # Dump trained weights for hardcoding
    print("\n# ==== Weights and Biases ====")
    for name, param in model.named_parameters():
        array = param.detach().numpy()
        print(f"{name.replace('.', '_')} = np.{repr(array)}\n")

    return model

if __name__ == "__main__":
    data = load_data()
    boards = np.array([state.board for state, utility in data])
    current_players = np.array([state.fill_num for state, utility in data])
    utilities = np.array([utility for state, utility in data])

    features = extract_features_numpy(boards, current_players)
    model = train_model(features, utilities)
