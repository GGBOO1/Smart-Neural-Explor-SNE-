from collections import deque, defaultdict
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sne_simplified_basics import (
    AttentionGranularity, IntentionType, CognitiveState, ExplorationPhase,
    Experience, AttentionFocus, SimpleIntention, PriorityLevel, EmergencyType
)
from sne_simplified_neural import SimpleStateEncoder, SimpleUncertaintyEstimator, SimpleCuriosityModule
from sne_simplified_core import SimpleWorkingMemory, SimpleAttentionManager, SimpleIntentionGenerator, SimpleMetaCognitiveMonitor, SimplePriorityManager

class SimpleSmartNeuralExplorer:
    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.step_count = 0
        
        self.state_encoder = SimpleStateEncoder(state_dim)
        self.uncertainty_estimator = SimpleUncertaintyEstimator(state_dim, action_dim)
        self.curiosity_module = SimpleCuriosityModule(state_dim, action_dim)
        
        self.attention_manager = SimpleAttentionManager(state_dim)
        self.intention_generator = SimpleIntentionGenerator(action_dim)
        self.meta_monitor = SimpleMetaCognitiveMonitor()
        self.priority_manager = SimplePriorityManager(state_dim, action_dim)
        
        self.experience_buffer = deque(maxlen=500)
        self.best_action = None
        self.best_reward = -float('inf')
        
        self.exploration_phase = ExplorationPhase.EARLY
        self.exploration_rate = 1.0
        
        self.stats = {
            'exploration_types': defaultdict(int),
            'steps': 0,
            'emergency_triggers': 0,
            'priority_actions': 0
        }
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        self.step_count += 1
        self.stats['steps'] += 1
        
        self._update_exploration_phase()
        
        uncertainty = self._estimate_uncertainty(state)
        
        novelty = self._calculate_novelty(state)
        
        cognitive_state = self.meta_monitor.update_state(
            reward=0,
            uncertainty=uncertainty.mean(),
            novelty=novelty
        )
        
        attention_focus = self.attention_manager.select_focus(
            state=state,
            uncertainty=uncertainty,
            novelty=novelty,
            cognitive_state=cognitive_state
        )
        
        intentions = self.intention_generator.generate_intentions(
            state=state,
            focus=attention_focus,
            best_action=self.best_action,
            num_intentions=3
        )
        
        recent_reward = 0
        if self.experience_buffer:
            recent_reward = self.experience_buffer[-1].reward
        
        for intention in intentions:
            self.priority_manager.add_intention(
                intention=intention,
                cognitive_state=cognitive_state,
                exploration_phase=self.exploration_phase,
                reward=recent_reward,
                timestamp=self.step_count
            )
        
        emergency_triggered = self.priority_manager.check_and_activate_emergency(
            reward=recent_reward,
            uncertainty=uncertainty.mean(),
            novelty=novelty,
            cognitive_state=cognitive_state,
            timestamp=self.step_count
        )
        
        if emergency_triggered:
            self.stats['emergency_triggers'] += 1
        
        selected_intention = self.priority_manager.get_highest_priority_intention()
        
        if selected_intention is None:
            action = np.random.uniform(-1, 1, self.action_dim)
            intention_type = IntentionType.RANDOM
        else:
            self.stats['priority_actions'] += 1
            action = selected_intention.action
            intention_type = selected_intention.intention_type
        
        action = self._add_exploration_noise(action)
        
        self.stats['exploration_types'][intention_type.value] += 1
        
        return action
    
    def _select_best_intention(self, intentions: List[SimpleIntention]) -> Optional[SimpleIntention]:
        if not intentions:
            return None
        
        return max(intentions, key=lambda x: x.confidence)
    
    def _add_exploration_noise(self, action: np.ndarray) -> np.ndarray:
        if self.exploration_phase == ExplorationPhase.EARLY:
            noise_level = 0.2
        else:
            noise_level = 0.05
        
        noise = np.random.randn(*action.shape) * noise_level
        return np.clip(action + noise, -1.0, 1.0)
    
    def _estimate_uncertainty(self, state: np.ndarray) -> np.ndarray:
        if len(self.experience_buffer) < 5:
            return np.ones(self.state_dim) * 0.8
        
        uncertainties = []
        for _ in range(3):
            random_action = np.random.randn(self.action_dim)
            uncertainty = self.uncertainty_estimator.forward(state, random_action)
            uncertainties.append(uncertainty.item())
        
        avg_uncertainty = np.mean(uncertainties)
        return np.ones(self.state_dim) * avg_uncertainty
    
    def _calculate_novelty(self, state: np.ndarray) -> float:
        if len(self.experience_buffer) < 3:
            return 1.0
        
        states = np.array([exp.state for exp in list(self.experience_buffer)[-20:]])
        distances = np.linalg.norm(states - state, axis=1)
        min_distance = np.min(distances) if len(distances) > 0 else 1.0
        
        novelty = np.exp(-min_distance)
        return float(novelty)
    
    def _update_exploration_phase(self):
        if self.step_count < 200:
            self.exploration_phase = ExplorationPhase.EARLY
            self.exploration_rate = 1.0
        else:
            self.exploration_phase = ExplorationPhase.LATE
            self.exploration_rate = 0.3
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        novelty = self._calculate_novelty(state)
        uncertainty = self._estimate_uncertainty(state).mean()
        
        experience = Experience(
            state=state.copy(),
            action=action.copy(),
            reward=reward,
            next_state=next_state.copy(),
            done=done
        )
        
        self.experience_buffer.append(experience)
        
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_action = action.copy()
        
        self.meta_monitor.update_state(
            reward=reward,
            uncertainty=uncertainty,
            novelty=novelty
        )
    
    def get_agent_info(self) -> Dict[str, Any]:
        meta_metrics = self.meta_monitor.get_metrics()
        
        info = {
            'step': self.step_count,
            'exploration_phase': self.exploration_phase.value,
            'exploration_rate': self.exploration_rate,
            'experience_buffer_size': len(self.experience_buffer),
            'best_reward': self.best_reward,
            'stats': dict(self.stats),
            'meta_metrics': meta_metrics,
            'priority_stats': self.priority_manager.get_priority_stats(),
            'emergency_status': self.priority_manager.get_emergency_status()
        }
        
        return info