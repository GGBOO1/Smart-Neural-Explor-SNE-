from collections import deque, defaultdict
import numpy as np
import random
from typing import List, Dict, Optional, Any
from sne_simplified_basics import (
    AttentionGranularity, IntentionType, CognitiveState, ExplorationPhase,
    Experience, AttentionFocus, SimpleIntention, PriorityLevel, EmergencyType,
    PriorityItem, EmergencyChannel
)
import heapq

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

class SimplePriorityManager:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.priority_queue = []
        self.emergency_channel = EmergencyChannel()
        self.priority_history = deque(maxlen=50)
        self.emergency_history = deque(maxlen=20)
        
        self.params = {
            'priority_weights': {
                'confidence': 0.5,
                'urgency': 0.3,
                'cognitive_state': 0.2
            },
            'emergency_thresholds': {
                'performance_drop': -0.5,
                'uncertainty': 0.7,
                'novelty': 0.8
            }
        }
    
    def calculate_priority_score(self, intention: SimpleIntention, cognitive_state: CognitiveState, 
                               exploration_phase: ExplorationPhase, reward: float) -> float:
        base_score = {
            PriorityLevel.LOW: 0.2,
            PriorityLevel.MEDIUM: 0.5,
            PriorityLevel.HIGH: 0.8,
            PriorityLevel.CRITICAL: 1.0
        }[intention.priority]
        
        cognitive_bonus = {
            CognitiveState.FOCUSED: 0.1,
            CognitiveState.EXPLORATORY: -0.1
        }[cognitive_state]
        
        priority_score = (
            base_score +
            self.params['priority_weights']['confidence'] * intention.confidence +
            self.params['priority_weights']['urgency'] * intention.urgency +
            self.params['priority_weights']['cognitive_state'] * cognitive_bonus +
            random.uniform(-0.05, 0.05)
        )
        
        priority_score = np.clip(priority_score, 0.0, 1.0)
        return priority_score
    
    def add_intention(self, intention: SimpleIntention, cognitive_state: CognitiveState, 
                     exploration_phase: ExplorationPhase, reward: float, timestamp: int):
        priority_score = self.calculate_priority_score(intention, cognitive_state, 
                                                     exploration_phase, reward)
        
        priority_item = PriorityItem(
            intention=intention,
            priority_score=priority_score,
            timestamp=timestamp,
            emergency_type=EmergencyType.NONE
        )
        
        heapq.heappush(self.priority_queue, (-priority_score, timestamp, priority_item))
        self.priority_history.append((timestamp, priority_score, intention.intention_type.value))
    
    def get_highest_priority_intention(self) -> Optional[SimpleIntention]:
        if self.emergency_channel.is_active:
            self.emergency_history.append((self.emergency_channel.emergency_type, 
                                          self.emergency_channel.severity))
            return self._create_emergency_intention()
        
        if not self.priority_queue:
            return None
        
        _, _, priority_item = heapq.heappop(self.priority_queue)
        return priority_item.intention
    
    def check_and_activate_emergency(self, reward: float, uncertainty: float, 
                                    novelty: float, cognitive_state: CognitiveState, 
                                    timestamp: int) -> bool:
        if reward < self.params['emergency_thresholds']['performance_drop']:
            self.emergency_channel.activate(
                EmergencyType.PERFORMANCE_DROP,
                severity=abs(reward),
                recommended_action=np.random.uniform(-1, 1, self.action_dim),
                timestamp=timestamp
            )
            return True
        
        if uncertainty > self.params['emergency_thresholds']['uncertainty']:
            self.emergency_channel.activate(
                EmergencyType.HIGH_UNCERTAINTY,
                severity=uncertainty,
                recommended_action=np.random.uniform(-1, 1, self.action_dim),
                timestamp=timestamp
            )
            return True
        
        if novelty > self.params['emergency_thresholds']['novelty']:
            self.emergency_channel.activate(
                EmergencyType.NOVEL_SITUATION,
                severity=novelty,
                recommended_action=np.random.uniform(-1, 1, self.action_dim),
                timestamp=timestamp
            )
            return True
        
        if self.emergency_channel.is_active:
            self.emergency_channel.deactivate()
        
        return False
    
    def _create_emergency_intention(self) -> SimpleIntention:
        emergency_type = self.emergency_channel.emergency_type
        severity = self.emergency_channel.severity
        action = self.emergency_channel.recommended_action
        
        if action is None:
            action = np.random.uniform(-1, 1, self.action_dim)
        
        intention_type = {
            EmergencyType.PERFORMANCE_DROP: IntentionType.MODIFY,
            EmergencyType.HIGH_UNCERTAINTY: IntentionType.RANDOM,
            EmergencyType.NOVEL_SITUATION: IntentionType.RANDOM
        }.get(emergency_type, IntentionType.RANDOM)
        
        emergency_intention = SimpleIntention(
            intention_type=intention_type,
            action=action,
            confidence=0.6,
            priority=PriorityLevel.CRITICAL,
            urgency=1.0
        )
        
        return emergency_intention
    
    def clear_queue(self):
        self.priority_queue.clear()
    
    def get_queue_size(self) -> int:
        return len(self.priority_queue)
    
    def get_emergency_status(self) -> Dict[str, Any]:
        return {
            'is_active': self.emergency_channel.is_active,
            'emergency_type': self.emergency_channel.emergency_type.value,
            'severity': self.emergency_channel.severity,
            'trigger_timestamp': self.emergency_channel.trigger_timestamp
        }
    
    def get_priority_stats(self) -> Dict[str, Any]:
        if not self.priority_history:
            return {
                'avg_priority': 0.0,
                'queue_size': len(self.priority_queue),
                'emergency_active': self.emergency_channel.is_active
            }
        
        avg_priority = np.mean([score for _, score, _ in self.priority_history])
        return {
            'avg_priority': avg_priority,
            'queue_size': len(self.priority_queue),
            'emergency_active': self.emergency_channel.is_active,
            'emergency_count': len([e for e in self.emergency_history if e[0] != EmergencyType.NONE])
        }
    
    def batch_add_intentions(self, intentions: List[SimpleIntention], cognitive_state: CognitiveState, 
                            exploration_phase: ExplorationPhase, reward: float, timestamp: int):
        batch_items = []
        
        for intention in intentions:
            priority_score = self.calculate_priority_score(intention, cognitive_state, 
                                                         exploration_phase, reward)
            
            priority_item = PriorityItem(
                intention=intention,
                priority_score=priority_score,
                timestamp=timestamp,
                emergency_type=EmergencyType.NONE
            )
            
            batch_items.append((-priority_score, timestamp, priority_item))
            self.priority_history.append((timestamp, priority_score, intention.intention_type.value))
        
        for item in batch_items:
            heapq.heappush(self.priority_queue, item)
    
    def task_sorting_channel(self, sort_by: str = 'priority') -> List[SimpleIntention]:
        if not self.priority_queue:
            return []
        
        items = []
        while self.priority_queue:
            items.append(heapq.heappop(self.priority_queue))
        
        if sort_by == 'priority':
            items.sort(key=lambda x: x[0])
        elif sort_by == 'timestamp':
            items.sort(key=lambda x: x[1])
        elif sort_by == 'confidence':
            items.sort(key=lambda x: x[2].intention.confidence, reverse=True)
        
        sorted_intentions = [item[2].intention for item in items]
        
        for item in items:
            heapq.heappush(self.priority_queue, item)
        
        return sorted_intentions
    
    def idea_sorting_channel(self, ideas: List[SimpleIntention], cognitive_state: CognitiveState, 
                            exploration_phase: ExplorationPhase, reward: float) -> List[SimpleIntention]:
        scored_ideas = []
        
        for idea in ideas:
            priority_score = self.calculate_priority_score(idea, cognitive_state, 
                                                         exploration_phase, reward)
            scored_ideas.append((-priority_score, idea))
        
        scored_ideas.sort(key=lambda x: x[0])
        sorted_ideas = [idea for _, idea in scored_ideas]
        
        return sorted_ideas
    
    def emergency_sort_override(self) -> List[SimpleIntention]:
        emergency_items = []
        regular_items = []
        
        while self.priority_queue:
            item = heapq.heappop(self.priority_queue)
            if item[2].intention.priority == PriorityLevel.CRITICAL:
                emergency_items.append(item)
            else:
                regular_items.append(item)
        
        emergency_items.sort(key=lambda x: x[0])
        regular_items.sort(key=lambda x: x[0])
        
        all_items = emergency_items + regular_items
        sorted_intentions = [item[2].intention for item in all_items]
        
        for item in all_items:
            heapq.heappush(self.priority_queue, item)
        
        return sorted_intentions