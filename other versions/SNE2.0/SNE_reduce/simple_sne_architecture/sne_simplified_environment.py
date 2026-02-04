import numpy as np
from typing import Dict, Tuple, Any

class SimpleDemonstrationEnvironment:
    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = np.zeros(state_dim)
        self.steps = 0
        self.max_steps = 100
    
    def reset(self) -> np.ndarray:
        self.state = np.random.uniform(-1, 1, self.state_dim)
        self.steps = 0
        return self.state.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.steps += 1
        
        action = np.clip(action, -1, 1)
        
        reward = -np.linalg.norm(self.state[:len(action)] - action)
        
        noise = np.random.randn(self.state_dim) * 0.1
        state_update = np.zeros(self.state_dim)
        state_update[:len(action)] = action
        self.state = self.state * 0.9 + state_update * 0.1 + noise
        self.state = np.clip(self.state, -1, 1)
        
        done = self.steps >= self.max_steps
        
        info = {
            'steps': self.steps,
            'state_norm': np.linalg.norm(self.state),
            'action_norm': np.linalg.norm(action)
        }
        
        return self.state.copy(), reward, done, info
    
    def render(self, agent_info: Dict[str, Any] = None):
        print(f"Step: {self.steps}, State: {self.state[:2]}, Info: {agent_info}")
