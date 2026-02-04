import numpy as np

class SimpleStateEncoder:
    def __init__(self, input_dim: int, hidden_dims: list = [32, 16]):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = hidden_dims[-1]
        self.weights = []
        self.biases = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.weights.append(np.random.randn(prev_dim, hidden_dim) * 0.01)
            self.biases.append(np.zeros(hidden_dim))
            prev_dim = hidden_dim
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        x = state
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = np.dot(x, w) + b
            x = np.maximum(0, x)  # ReLU
            if i < len(self.weights) - 1:
                x *= (np.random.rand(*x.shape) > 0.1)  # Dropout
        return x

class SimpleUncertaintyEstimator:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weights = [
            np.random.randn(state_dim + action_dim, 64) * 0.01,
            np.random.randn(64, 1) * 0.01
        ]
        self.biases = [np.zeros(64), np.zeros(1)]
    
    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = np.concatenate([state, action], axis=-1)
        x = np.dot(x, self.weights[0]) + self.biases[0]
        x = np.maximum(0, x)  # ReLU
        x = np.dot(x, self.weights[1]) + self.biases[1]
        x = np.log(1 + np.exp(x))  # Softplus
        return x

class SimpleCuriosityModule:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.forward_model_weights = [
            np.random.randn(state_dim + action_dim, 64) * 0.01,
            np.random.randn(64, state_dim) * 0.01
        ]
        self.forward_model_biases = [np.zeros(64), np.zeros(state_dim)]
    
    def compute_curiosity(self, state: np.ndarray, action: np.ndarray, 
                         next_state: np.ndarray) -> np.ndarray:
        x = np.concatenate([state, action], axis=-1)
        x = np.dot(x, self.forward_model_weights[0]) + self.forward_model_biases[0]
        x = np.maximum(0, x)  # ReLU
        predicted_next_state = np.dot(x, self.forward_model_weights[1]) + self.forward_model_biases[1]
        prediction_error = np.mean((predicted_next_state - next_state) ** 2)
        return np.array([prediction_error])
