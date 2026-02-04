import numpy as np

class StateEncoder:
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16]):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = hidden_dims[-1]
        self.weights = []
        self.biases = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.weights.append(np.random.randn(prev_dim, hidden_dim) * 0.1)
            self.biases.append(np.zeros(hidden_dim))
            prev_dim = hidden_dim
        
    def forward(self, state: np.ndarray) -> np.ndarray:
        x = state
        for i in range(len(self.weights)):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = np.maximum(0, x)
        return x

class UncertaintyEstimator:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.weights = [
            np.random.randn(state_dim + action_dim, 128) * 0.1,
            np.random.randn(128, 64) * 0.1,
            np.random.randn(64, 1) * 0.1
        ]
        
        self.biases = [
            np.zeros(128),
            np.zeros(64),
            np.zeros(1)
        ]
    
    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = np.concatenate([state, action], axis=-1)
        x = np.dot(x, self.weights[0]) + self.biases[0]
        x = np.maximum(0, x)
        x = np.dot(x, self.weights[1]) + self.biases[1]
        x = np.maximum(0, x)
        x = np.dot(x, self.weights[2]) + self.biases[2]
        x = np.log(1 + np.exp(x))
        return x

class CuriosityModule:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.inverse_weights = [
            np.random.randn(state_dim * 2, 128) * 0.1,
            np.random.randn(128, action_dim) * 0.1
        ]
        
        self.inverse_biases = [
            np.zeros(128),
            np.zeros(action_dim)
        ]
        
        self.forward_weights = [
            np.random.randn(state_dim + action_dim, 128) * 0.1,
            np.random.randn(128, state_dim) * 0.1
        ]
        
        self.forward_biases = [
            np.zeros(128),
            np.zeros(state_dim)
        ]
        
    def compute_curiosity(self, state: np.ndarray, action: np.ndarray, 
                         next_state: np.ndarray) -> np.ndarray:
        x = np.concatenate([state, action], axis=-1)
        x = np.dot(x, self.forward_weights[0]) + self.forward_biases[0]
        x = np.maximum(0, x)
        predicted_next_state = np.dot(x, self.forward_weights[1]) + self.forward_biases[1]
        prediction_error = np.mean((predicted_next_state - next_state) ** 2, axis=-1, keepdims=True)
        return prediction_error