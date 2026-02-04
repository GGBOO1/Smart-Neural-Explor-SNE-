from collections import deque, defaultdict
import numpy as np
import random
from typing import List, Dict, Optional
from sne_simplified_basics import (
    AttentionGranularity, IntentionType, CognitiveState, ExplorationPhase,
    Experience, AttentionFocus, SimpleIntention
)

class SimpleWorkingMemory:
    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def add(self, item: Any):
        self.memory.append(item)
        
    def get_contents(self) -> List[Any]:
        return list(self.memory)
    
    def clear(self):
        self.memory.clear()

class SimpleAttentionManager:
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.working_memory = SimpleWorkingMemory(capacity=5)
        self.current_focus = None
        
    def select_focus(self, state: np.ndarray, uncertainty: np.ndarray, 
                    novelty: float, cognitive_state: CognitiveState) -> AttentionFocus:
        attention_scores = np.zeros(self.state_dim)
        
        for i in range(self.state_dim):
            score = uncertainty[i] + novelty * 0.5 + random.uniform(-0.1, 0.1)
            attention_scores[i] = score
        
        top_k = 2 if cognitive_state == CognitiveState.EXPLORATORY else 1
        focus_dims = np.argsort(attention_scores)[-top_k:].tolist()
        
        granularity = AttentionGranularity.FINE if cognitive_state == CognitiveState.FOCUSED else AttentionGranularity.COARSE
        
        focus = AttentionFocus(
            state_dimensions=focus_dims,
            granularity=granularity
        )
        
        self.working_memory.add(focus)
        self.current_focus = focus
        
        return focus

class SimpleIntentionGenerator:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.intention_history = defaultdict(int)
        
    def generate_intentions(self, state: np.ndarray, focus: AttentionFocus, 
                           best_action: Optional[np.ndarray], 
                           num_intentions: int = 3) -> List[SimpleIntention]:
        intentions = []
        
        if best_action is not None:
            modify_intent = self._generate_modify_intention(best_action, focus)
            intentions.append(modify_intent)
            
            direct_intent = SimpleIntention(
                intention_type=IntentionType.DIRECT,
                action=best_action.copy(),
                confidence=0.8
            )
            intentions.append(direct_intent)
        
        random_intent = self._generate_random_intention(focus)
        intentions.append(random_intent)
        
        scored_intentions = []
        for intent in intentions:
            frequency = self._get_intention_frequency(intent)
            novelty_bonus = 1.0 - frequency
            
            score = intent.confidence + novelty_bonus * 0.3 + random.uniform(0, 0.1)
            scored_intentions.append((score, intent))
        
        scored_intentions.sort(key=lambda x: x[0], reverse=True)
        selected = [intent for _, intent in scored_intentions[:num_intentions]]
        
        for intent in selected:
            self.intention_history[self._intention_signature(intent)] += 1
        
        return selected
    
    def _generate_modify_intention(self, best_action: np.ndarray, 
                                  focus: AttentionFocus) -> SimpleIntention:
        magnitude = 0.2 if focus.granularity == AttentionGranularity.FINE else 0.4
        modified_action = best_action + np.random.randn(*best_action.shape) * magnitude
        modified_action = np.clip(modified_action, -1, 1)
        
        return SimpleIntention(
            intention_type=IntentionType.MODIFY,
            action=modified_action,
            confidence=0.6
        )
    
    def _generate_random_intention(self, focus: AttentionFocus) -> SimpleIntention:
        random_action = np.random.uniform(-1, 1, self.action_dim)
        
        return SimpleIntention(
            intention_type=IntentionType.RANDOM,
            action=random_action,
            confidence=0.2
        )
    
    def _get_intention_frequency(self, intention: SimpleIntention) -> float:
        signature = self._intention_signature(intention)
        frequency = self.intention_history[signature]
        return min(1.0, frequency / 5.0)
    
    def _intention_signature(self, intention: SimpleIntention) -> str:
        return f"{intention.intention_type.value}"

class SimpleMetaCognitiveMonitor:
    def __init__(self):
        self.performance_history = deque(maxlen=50)
        self.current_state = CognitiveState.FOCUSED
        
    def update_state(self, reward: float, uncertainty: float, 
                    novelty: float) -> CognitiveState:
        self.performance_history.append(reward)
        
        if uncertainty > 0.6 or novelty > 0.7:
            self.current_state = CognitiveState.EXPLORATORY
        else:
            self.current_state = CognitiveState.FOCUSED
        
        return self.current_state
    
    def get_metrics(self) -> Dict[str, float]:
        return {
            'cognitive_state': self.current_state.value,
            'avg_recent_reward': np.mean(self.performance_history) if self.performance_history else 0
        }
