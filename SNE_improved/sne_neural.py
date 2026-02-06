import torch
import torch.nn as nn
import torch.nn.functional as F

class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class UncertaintyEstimator(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class CuriosityModule(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> tuple:
        pred_next_state = self.forward_model(torch.cat([state, action], dim=1))
        pred_action = self.inverse_model(torch.cat([state, next_state], dim=1))
        
        forward_loss = F.mse_loss(pred_next_state, next_state)
        inverse_loss = F.mse_loss(pred_action, action)
        
        return forward_loss, inverse_loss