from collections import deque, defaultdict
import numpy as np
import torch
import random
from typing import List, Dict, Tuple, Optional, Any
from sne_basics import (
    AttentionGranularity, IntentionType, CognitiveState, ExplorationPhase,
    Experience, AttentionFocus, MetaIntention, EmotionalState, ContextFrame
)
from sne_neural import StateEncoder, UncertaintyEstimator, CuriosityModule
from sne_core import WorkingMemory, DynamicAttentionManager, CreativeIntentionGenerator, MetaCognitiveMonitor

class SmartNeuralExplorer:
    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.step_count = 0
        
        self.state_encoder = StateEncoder(state_dim)
        self.uncertainty_estimator = UncertaintyEstimator(state_dim, action_dim)
        self.curiosity_module = CuriosityModule(state_dim, action_dim)
        
        self.attention_manager = DynamicAttentionManager(state_dim, action_dim)
        self.intention_generator = CreativeIntentionGenerator(action_dim, state_dim)
        self.meta_monitor = MetaCognitiveMonitor()
        
        self.experience_buffer = deque(maxlen=1000)
        self.best_action = None
        self.best_reward = -float('inf')
        
        self.exploration_phase = ExplorationPhase.EARLY
        self.exploration_rate = 1.0
        self.cognitive_load = 0.0
        
        self.emotional_state = EmotionalState.NEUTRAL
        self.cognitive_flexibility = 1.0
        self.learning_styles = ['analytical', 'holistic', 'pragmatic', 'creative']
        self.current_learning_style = 'analytical'
        self.style_switch_counter = 0
        self.style_switch_threshold = 20
        
        self.stats = {
            'exploration_types': defaultdict(int),
            'attention_changes': 0,
            'intentions_generated': 0,
            'meta_state_changes': 0,
            'emotional_states': defaultdict(int),
            'learning_styles': defaultdict(int)
        }
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        self.step_count += 1
        
        self._update_exploration_phase()
        self._update_emotional_state()
        self._update_learning_style()
        
        if self._should_use_random_action():
            action = np.random.uniform(-1, 1, self.action_dim)
            self.stats['exploration_types']['random_adaptive'] += 1
            return action
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        uncertainty = self._estimate_uncertainty(state_tensor)
        
        novelty = self._calculate_novelty(state)
        
        cognitive_state = self.meta_monitor.update_state(
            reward=0,
            uncertainty=uncertainty.mean(),
            novelty=novelty,
            step=self.step_count
        )
        
        self.cognitive_load = self._calculate_cognitive_load()
        
        associative_strength = self._get_associative_strength(state)
        attention_focus = self.attention_manager.select_focus(
            state=state,
            uncertainty=uncertainty,
            novelty=novelty,
            associative_strength=associative_strength,
            cognitive_state=cognitive_state,
            exploration_phase=self.exploration_phase
        )
        
        intentions = self.intention_generator.generate_intentions(
            state=state,
            focus=attention_focus,
            best_action=self.best_action,
            experiences=list(self.experience_buffer),
            num_intentions=5
        )
        
        self.stats['intentions_generated'] += len(intentions)
        
        selected_intention = self.attention_manager.gated_selection(
            intentions=intentions,
            cognitive_load=self.cognitive_load
        )
        
        if selected_intention is None:
            action = np.random.uniform(-1, 1, self.action_dim)
            intention_type = IntentionType.RANDOM
        else:
            action = self._intention_to_action(selected_intention, attention_focus)
            intention_type = selected_intention.intention_type
        
        action = self._add_exploration_noise(action)
        
        self.stats['exploration_types'][intention_type.value] += 1
        self.stats['emotional_states'][self.emotional_state.value] += 1
        self.stats['learning_styles'][self.current_learning_style] += 1
        
        return action
    
    def _intention_to_action(self, intention: MetaIntention, 
                            focus: AttentionFocus) -> np.ndarray:
        base_action = intention.base_action.copy()
        
        if intention.intention_type == IntentionType.MODIFICATION:
            noise = np.random.randn(*base_action.shape) * intention.fuzziness
            action = base_action + noise
            
        elif intention.intention_type == IntentionType.ANALOGICAL:
            alpha = np.random.uniform(0.3, 0.7)
            action = alpha * base_action + (1 - alpha) * np.random.randn(*base_action.shape)
            
        elif intention.intention_type == IntentionType.COMBINATION:
            action = base_action.copy()
            
        elif intention.intention_type == IntentionType.COUNTERFACTUAL:
            action = np.tanh(base_action)
            
        else:
            action = base_action.copy()
        
        if focus.granularity == AttentionGranularity.FINE:
            action = action * 0.9 + np.random.randn(*action.shape) * 0.1
        elif focus.granularity == AttentionGranularity.COARSE:
            pass
        
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def _add_exploration_noise(self, action: np.ndarray) -> np.ndarray:
        if self.exploration_phase == ExplorationPhase.EARLY:
            noise_level = 0.3
        elif self.exploration_phase == ExplorationPhase.MIDDLE:
            noise_level = 0.1
        else:
            noise_level = 0.05
        
        noise = np.random.randn(*action.shape) * noise_level
        return np.clip(action + noise, -1.0, 1.0)
    
    def _estimate_uncertainty(self, state: torch.Tensor) -> np.ndarray:
        if len(self.experience_buffer) < 10:
            return np.ones(self.state_dim) * 0.8
        
        uncertainties = []
        for _ in range(5):
            random_action = torch.randn(1, self.action_dim)
            with torch.no_grad():
                uncertainty = self.uncertainty_estimator(state, random_action)
            uncertainties.append(uncertainty.item())
        
        avg_uncertainty = np.mean(uncertainties)
        return np.ones(self.state_dim) * avg_uncertainty
    
    def _calculate_novelty(self, state: np.ndarray) -> float:
        if len(self.experience_buffer) < 5:
            return 1.0
        
        states = np.array([exp.state for exp in list(self.experience_buffer)[-50:]])
        distances = np.linalg.norm(states - state, axis=1)
        min_distance = np.min(distances) if len(distances) > 0 else 1.0
        
        novelty = np.exp(-min_distance)
        return float(novelty)
    
    def _get_associative_strength(self, state: np.ndarray) -> np.ndarray:
        return np.random.rand(self.state_dim) * 0.5
    
    def _calculate_cognitive_load(self) -> float:
        base_load = 0.3
        
        if self.exploration_phase == ExplorationPhase.EARLY:
            base_load += 0.2
        
        if self.meta_monitor.current_state == CognitiveState.CONFUSED:
            base_load += 0.3
        
        base_load += np.random.uniform(-0.1, 0.1)
        
        return min(1.0, max(0.0, base_load))
    
    def _update_exploration_phase(self):
        if self.step_count < 100:
            self.exploration_phase = ExplorationPhase.EARLY
            self.exploration_rate = 1.0
        elif self.step_count < 500:
            self.exploration_phase = ExplorationPhase.MIDDLE
            self.exploration_rate = 0.5
        else:
            self.exploration_phase = ExplorationPhase.LATE
            self.exploration_rate = 0.2
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        novelty = self._calculate_novelty(state)
        uncertainty = self._estimate_uncertainty(torch.FloatTensor(state).unsqueeze(0)).mean()
        
        experience = Experience(
            state=state.copy(),
            action=action.copy(),
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            novelty=novelty,
            uncertainty=uncertainty,
            timestamp=self.step_count
        )
        
        self.experience_buffer.append(experience)
        
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_action = action.copy()
        
        self.meta_monitor.update_state(
            reward=reward,
            uncertainty=uncertainty,
            novelty=novelty,
            step=self.step_count
        )
    
    def get_agent_info(self) -> Dict[str, Any]:
        meta_metrics = self.meta_monitor.get_metrics()
        
        info = {
            'step': self.step_count,
            'exploration_phase': self.exploration_phase.value,
            'exploration_rate': self.exploration_rate,
            'cognitive_load': self.cognitive_load,
            'emotional_state': self.emotional_state.value,
            'cognitive_flexibility': self.cognitive_flexibility,
            'current_learning_style': self.current_learning_style,
            'experience_buffer_size': len(self.experience_buffer),
            'best_reward': self.best_reward,
            'stats': dict(self.stats),
            'meta_metrics': meta_metrics
        }
        
        if self.attention_manager.current_focus:
            info['current_focus'] = str(self.attention_manager.current_focus)
        
        return info
    
    def _update_emotional_state(self):
        if not self.experience_buffer:
            return
        
        recent_experiences = list(self.experience_buffer)[-5:]
        avg_reward = np.mean([exp.reward for exp in recent_experiences])
        
        if avg_reward > 2:
            self.emotional_state = EmotionalState.POSITIVE
        elif avg_reward < -1:
            self.emotional_state = EmotionalState.NEGATIVE
        elif avg_reward > 0.5:
            self.emotional_state = EmotionalState.EXCITED
        elif avg_reward < 0:
            self.emotional_state = EmotionalState.FRUSTRATED
        else:
            self.emotional_state = EmotionalState.NEUTRAL
    
    def _update_learning_style(self):
        self.style_switch_counter += 1
        
        if self.style_switch_counter >= self.style_switch_threshold:
            self.style_switch_counter = 0
            self.current_learning_style = random.choice(self.learning_styles)
            self.cognitive_flexibility = np.random.uniform(0.8, 1.2)
    
    def _should_use_random_action(self):
        base_probability = 0.3
        
        if self.emotional_state == EmotionalState.FRUSTRATED:
            base_probability = 0.6
        elif self.emotional_state == EmotionalState.CURIOUS:
            base_probability = 0.5
        elif self.emotional_state == EmotionalState.POSITIVE:
            base_probability = 0.2
        
        if self.current_learning_style == 'creative':
            base_probability *= 1.5
        elif self.current_learning_style == 'analytical':
            base_probability *= 0.8
        
        base_probability *= self.cognitive_flexibility
        
        return np.random.random() < min(base_probability, 0.8)
    
    def _calculate_cognitive_load(self) -> float:
        base_load = 0.3
        
        if self.exploration_phase == ExplorationPhase.EARLY:
            base_load += 0.2
        
        if self.meta_monitor.current_state == CognitiveState.CONFUSED:
            base_load += 0.3
        
        if self.emotional_state == EmotionalState.FRUSTRATED:
            base_load += 0.2
        
        base_load += np.random.uniform(-0.1, 0.1)
        
        return min(1.0, max(0.0, base_load))
    
    def _get_associative_strength(self, state: np.ndarray) -> np.ndarray:
        strengths = np.random.rand(self.state_dim) * 0.5
        
        if hasattr(self.intention_generator, 'semantic_network'):
            semantic_strengths = self.intention_generator.semantic_network.get_associative_strengths(
                [f"state_dim_{i}_{state[i]:.2f}" for i in range(len(state))]
            )
            if semantic_strengths:
                avg_strength = np.mean(list(semantic_strengths.values()))
                strengths = strengths * 0.5 + avg_strength * 0.5
        
        return strengths