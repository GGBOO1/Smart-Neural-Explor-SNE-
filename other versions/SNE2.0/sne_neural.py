import torch
import torch.nn as nn
import torch.nn.functional as F

class StateEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class UncertaintyEstimator(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class CuriosityModule(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
        
    def compute_curiosity(self, state: torch.Tensor, action: torch.Tensor, 
                         next_state: torch.Tensor) -> torch.Tensor:
        predicted_next_state = self.forward_model(torch.cat([state, action], dim=-1))
        prediction_error = F.mse_loss(predicted_next_state, next_state, reduction='none')
        return prediction_error.mean(dim=-1, keepdim=True)