import numpy as np
from typing import Tuple, Dict, Any, List

class MultiModalEnvironment:
    def __init__(self, state_dim: int = 4, action_dim: int = 2, max_steps: int = 100, modalities: List[str] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.current_step = 0
        self.target_state = np.random.uniform(-1, 1, state_dim)
        
        if modalities is None:
            self.modalities = ['visual', 'auditory', 'tactile', 'proprioceptive']
        else:
            self.modalities = modalities
        
        self.modality_dims = {
            'visual': state_dim,
            'auditory': state_dim // 2,
            'tactile': state_dim // 2,
            'proprioceptive': state_dim // 3
        }
        
        self.environment_dynamics = {
            'visual_noise': 0.1,
            'auditory_noise': 0.2,
            'tactile_noise': 0.15,
            'proprioceptive_noise': 0.05,
            'target_moving_speed': 0.01
        }
        
        self.contextual_factors = {
            'lighting': np.random.uniform(0.5, 1.0),
            'background_noise': np.random.uniform(0, 0.5),
            'temperature': np.random.uniform(0.7, 1.3),
            'fatigue': 0.0
        }
    
    def reset(self) -> Dict[str, np.ndarray]:
        self.current_step = 0
        self.target_state = np.random.uniform(-1, 1, self.state_dim)
        
        for key in self.contextual_factors:
            if key == 'lighting':
                self.contextual_factors[key] = np.random.uniform(0.5, 1.0)
            elif key == 'background_noise':
                self.contextual_factors[key] = np.random.uniform(0, 0.5)
            elif key == 'temperature':
                self.contextual_factors[key] = np.random.uniform(0.7, 1.3)
            elif key == 'fatigue':
                self.contextual_factors[key] = 0.0
        
        return self._generate_multi_modal_state()
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        self.current_step += 1
        
        self._update_target_state()
        self._update_contextual_factors()
        
        multi_modal_state = self._generate_multi_modal_state()
        unified_state = self._unify_modalities(multi_modal_state)
        
        distance = np.linalg.norm(unified_state - self.target_state)
        reward = self._calculate_reward(distance, multi_modal_state)
        
        done = self.current_step >= self.max_steps
        
        info = {
            'distance': distance,
            'target_state': self.target_state.copy(),
            'step': self.current_step,
            'contextual_factors': self.contextual_factors.copy(),
            'unified_state': unified_state
        }
        
        return multi_modal_state, reward, done, info
    
    def _generate_multi_modal_state(self) -> Dict[str, np.ndarray]:
        multi_modal_state = {}
        base_state = np.random.uniform(-1, 1, self.state_dim)
        
        for modality in self.modalities:
            if modality == 'visual':
                noise = np.random.randn(self.modality_dims[modality]) * self.environment_dynamics['visual_noise']
                noise *= self.contextual_factors['lighting']
                state = base_state + noise
                state = np.clip(state, -1, 1)
            elif modality == 'auditory':
                noise = np.random.randn(self.modality_dims[modality]) * self.environment_dynamics['auditory_noise']
                noise += self.contextual_factors['background_noise']
                state = base_state[:self.modality_dims[modality]] + noise
                state = np.clip(state, -1, 1)
            elif modality == 'tactile':
                noise = np.random.randn(self.modality_dims[modality]) * self.environment_dynamics['tactile_noise']
                noise *= self.contextual_factors['temperature']
                state = base_state[:self.modality_dims[modality]] + noise
                state = np.clip(state, -1, 1)
            elif modality == 'proprioceptive':
                noise = np.random.randn(self.modality_dims[modality]) * self.environment_dynamics['proprioceptive_noise']
                noise *= (1 + self.contextual_factors['fatigue'])
                state = base_state[:self.modality_dims[modality]] + noise
                state = np.clip(state, -1, 1)
            else:
                state = np.random.uniform(-1, 1, self.modality_dims.get(modality, self.state_dim))
            
            multi_modal_state[modality] = state
        
        return multi_modal_state
    
    def _unify_modalities(self, multi_modal_state: Dict[str, np.ndarray]) -> np.ndarray:
        states = []
        for modality, state in multi_modal_state.items():
            states.extend(state[:self.state_dim // len(self.modalities)])
        
        while len(states) < self.state_dim:
            states.append(0.0)
        
        return np.array(states[:self.state_dim])
    
    def _calculate_reward(self, distance: float, multi_modal_state: Dict[str, np.ndarray]) -> float:
        base_reward = np.exp(-distance) * 10
        
        modality_bonus = 0.0
        for modality, state in multi_modal_state.items():
            if modality == 'visual':
                clarity = 1.0 - np.std(state) * 0.5
                modality_bonus += clarity * 0.5
            elif modality == 'auditory':
                signal_to_noise = 1.0 - self.contextual_factors['background_noise']
                modality_bonus += signal_to_noise * 0.3
            elif modality == 'tactile':
                stability = 1.0 - np.std(state) * 0.3
                modality_bonus += stability * 0.4
            elif modality == 'proprioceptive':
                precision = 1.0 - self.contextual_factors['fatigue']
                modality_bonus += precision * 0.6
        
        reward = base_reward * (1 + modality_bonus * 0.1)
        
        if np.random.random() < 0.1:
            reward *= 1.5
        
        if distance < 0.2:
            reward += 5.0
        
        return reward
    
    def _update_target_state(self):
        movement = np.random.randn(self.state_dim) * self.environment_dynamics['target_moving_speed']
        self.target_state = self.target_state + movement
        self.target_state = np.clip(self.target_state, -1, 1)
    
    def _update_contextual_factors(self):
        for key in self.contextual_factors:
            if key == 'lighting':
                self.contextual_factors[key] += np.random.uniform(-0.05, 0.05)
                self.contextual_factors[key] = np.clip(self.contextual_factors[key], 0.1, 1.5)
            elif key == 'background_noise':
                self.contextual_factors[key] += np.random.uniform(-0.02, 0.02)
                self.contextual_factors[key] = np.clip(self.contextual_factors[key], 0, 1.0)
            elif key == 'temperature':
                self.contextual_factors[key] += np.random.uniform(-0.01, 0.01)
                self.contextual_factors[key] = np.clip(self.contextual_factors[key], 0.5, 1.5)
            elif key == 'fatigue':
                self.contextual_factors[key] += 0.01
                self.contextual_factors[key] = min(self.contextual_factors[key], 1.0)
    
    def get_state_dim(self) -> int:
        return self.state_dim
    
    def get_action_dim(self) -> int:
        return self.action_dim
    
    def get_modalities(self) -> List[str]:
        return self.modalities
    
    def get_modality_dim(self, modality: str) -> int:
        return self.modality_dims.get(modality, self.state_dim)
    
    def get_unified_state_dim(self) -> int:
        return self.state_dim