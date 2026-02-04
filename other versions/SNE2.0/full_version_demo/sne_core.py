from collections import deque, defaultdict
import numpy as np
import random
import heapq
from typing import List, Dict, Tuple, Optional, Any
from sne_basics import (
    AttentionGranularity, IntentionType, CognitiveState, ExplorationPhase,
    Experience, AttentionFocus, MetaIntention, PriorityLevel, EmergencyType,
    PriorityItem, EmergencyChannel
)

class WorkingMemory:
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.access_counts = defaultdict(int)
        
    def add(self, item: Any, importance: float = 1.0):
        if len(self.memory) >= self.capacity:
            self.memory.popleft()
        self.memory.append((item, importance))
        
    def get_contents(self) -> List[Any]:
        return [item for item, _ in self.memory]
    
    def clear(self):
        self.memory.clear()

class DynamicAttentionManager:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.working_memory = WorkingMemory(capacity=7)
        self.attention_history = deque(maxlen=100)
        
        self.params = {
            'uncertainty_weight': 0.4,
            'novelty_weight': 0.3,
            'associative_weight': 0.2,
            'reward_weight': 0.1,
            'fatigue_decay': 0.95,
            'persistence_base': 0.7
        }
        
        self.cognitive_fatigue = 0.0
        self.current_focus = None
        
    def select_focus(self, state: np.ndarray, uncertainty: np.ndarray,
                    novelty: float, associative_strength: np.ndarray,
                    cognitive_state: CognitiveState, exploration_phase: ExplorationPhase) -> AttentionFocus:
        attention_scores = np.zeros(self.state_dim)
        
        for i in range(self.state_dim):
            score = (
                self.params['uncertainty_weight'] * uncertainty[i] +
                self.params['novelty_weight'] * novelty +
                self.params['associative_weight'] * associative_strength[i] +
                random.uniform(-0.05, 0.05)
            )
            attention_scores[i] = score
        
        top_k = 2 if exploration_phase == ExplorationPhase.EARLY else 1
        focus_dims = np.argsort(attention_scores)[-top_k:].tolist()
        
        granularity = self._determine_granularity(cognitive_state, exploration_phase, uncertainty.mean())
        
        persistence = self._calculate_persistence(cognitive_state, granularity)
        curiosity = self._calculate_curiosity(cognitive_state, novelty)
        
        self._update_fatigue(granularity)
        
        focus = AttentionFocus(
            state_dimensions=focus_dims,
            action_features=['all'],
            granularity=granularity,
            persistence=persistence,
            curiosity=curiosity,
            fatigue=self.cognitive_fatigue
        )
        
        self.working_memory.add(focus, importance=persistence)
        self.attention_history.append(focus)
        self.current_focus = focus
        
        return focus
    
    def _determine_granularity(self, cognitive_state: CognitiveState, 
                              exploration_phase: ExplorationPhase, 
                              uncertainty: float) -> AttentionGranularity:
        if cognitive_state == CognitiveState.CONFUSED or uncertainty > 0.7:
            return AttentionGranularity.COARSE
        
        if cognitive_state == CognitiveState.INSIGHT:
            return AttentionGranularity.FINE
        
        if exploration_phase == ExplorationPhase.EARLY:
            return random.choice([AttentionGranularity.COARSE, AttentionGranularity.MIXED])
        elif exploration_phase == ExplorationPhase.LATE:
            return AttentionGranularity.FINE
        else:
            return AttentionGranularity.MIXED
    
    def _calculate_persistence(self, cognitive_state: CognitiveState, 
                              granularity: AttentionGranularity) -> float:
        base_persistence = self.params['persistence_base']
        
        if cognitive_state == CognitiveState.FOCUSED:
            return base_persistence * 1.3
        elif cognitive_state == CognitiveState.DIVERGENT:
            return base_persistence * 0.7
        elif cognitive_state == CognitiveState.CONFUSED:
            return base_persistence * 0.5
        else:
            return base_persistence * 1.5
    
    def _calculate_curiosity(self, cognitive_state: CognitiveState, 
                            novelty: float) -> float:
        if cognitive_state == CognitiveState.CONFUSED:
            return min(1.0, novelty * 1.5)
        elif cognitive_state == CognitiveState.DIVERGENT:
            return min(1.0, novelty * 1.2)
        else:
            return novelty
    
    def _update_fatigue(self, granularity: AttentionGranularity):
        fatigue_gain = {
            AttentionGranularity.FINE: 0.15,
            AttentionGranularity.MIXED: 0.08,
            AttentionGranularity.COARSE: 0.03
        }
        
        self.cognitive_fatigue += fatigue_gain[granularity]
        self.cognitive_fatigue *= self.params['fatigue_decay']
    
    def gated_selection(self, intentions: List[MetaIntention], 
                       cognitive_load: float) -> Optional[MetaIntention]:
        if not intentions:
            return None
        
        effective_capacity = max(1, int(3 * (1 - cognitive_load)))
        
        if len(intentions) <= effective_capacity:
            candidates = intentions
        else:
            scores = []
            for intention in intentions:
                score = (
                    0.4 * intention.confidence +
                    0.3 * intention.novelty +
                    0.2 * (1 - intention.fuzziness) +
                    0.1 * random.random()
                )
                scores.append(score)
            
            top_indices = np.argsort(scores)[-effective_capacity:]
            candidates = [intentions[i] for i in top_indices]
        
        if len(candidates) > 1:
            confidences = np.array([c.confidence for c in candidates])
            probabilities = np.exp(confidences) / np.exp(confidences).sum()
            selected_idx = np.random.choice(len(candidates), p=probabilities)
            return candidates[selected_idx]
        else:
            return candidates[0]

class CreativeIntentionGenerator:
    def __init__(self, action_dim: int, state_dim: int):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.intention_history = defaultdict(int)
        self.success_patterns = deque(maxlen=50)
        
    def generate_intentions(self, state: np.ndarray, focus: AttentionFocus,
                           best_action: Optional[np.ndarray], 
                           experiences: List[Experience], 
                           num_intentions: int = 5) -> List[MetaIntention]:
        intentions = []
        
        if best_action is not None:
            exploitation_intents = self._generate_exploitation_intentions(
                best_action, focus
            )
            intentions.extend(exploitation_intents)
        
        analogical_intents = self._generate_analogical_intentions(
            state, focus, experiences
        )
        intentions.extend(analogical_intents)
        
        combination_intents = self._generate_combination_intentions(
            focus, experiences
        )
        intentions.extend(combination_intents)
        
        counterfactual_intents = self._generate_counterfactual_intentions(
            experiences
        )
        intentions.extend(counterfactual_intents)
        
        random_intent = self._generate_random_intention(focus)
        intentions.append(random_intent)
        
        scored_intentions = []
        for intent in intentions:
            frequency = self._get_intention_frequency(intent)
            novelty_bonus = 1.0 - frequency
            
            score = (
                0.35 * intent.confidence +
                0.35 * intent.novelty * novelty_bonus +
                0.20 * (1 - intent.fuzziness) +
                0.10 * random.uniform(0, 1)
            )
            
            scored_intentions.append((score, intent))
        
        scored_intentions.sort(key=lambda x: x[0], reverse=True)
        selected = [intent for _, intent in scored_intentions[:num_intentions]]
        
        for intent in selected:
            self.intention_history[self._intention_signature(intent)] += 1
        
        return selected
    
    def _generate_exploitation_intentions(self, best_action: np.ndarray,
                                         focus: AttentionFocus) -> List[MetaIntention]:
        intentions = []
        
        if focus.granularity == AttentionGranularity.FINE:
            magnitudes = [0.05, 0.1, 0.15]
        elif focus.granularity == AttentionGranularity.COARSE:
            magnitudes = [0.3, 0.5, 0.7]
        else:
            magnitudes = [0.1, 0.2, 0.3]
        
        for magnitude in magnitudes:
            modified_action = best_action + np.random.randn(*best_action.shape) * magnitude
            modified_action = np.clip(modified_action, -1, 1)
            
            intent = MetaIntention(
                intention_type=IntentionType.MODIFICATION,
                base_action=best_action.copy(),
                target_feature="all",
                confidence=0.7 - magnitude,
                novelty=0.2 + magnitude * 0.5,
                fuzziness=magnitude,
                expected_outcome=modified_action,
                explanation_confidence=0.6,
                causal_hypothesis=f"Modify best action by ±{magnitude:.2f}",
                abandonment_threshold=0.4
            )
            intentions.append(intent)
        
        return intentions
    
    def _generate_analogical_intentions(self, state: np.ndarray, focus: AttentionFocus,
                                       experiences: List[Experience]) -> List[MetaIntention]:
        if len(experiences) < 3:
            return []
        
        intentions = []
        
        recent_experiences = experiences[-20:]
        if not recent_experiences:
            return []
        
        states = np.array([exp.state for exp in recent_experiences])
        rewards = np.array([exp.reward for exp in recent_experiences])
        actions = np.array([exp.action for exp in recent_experiences])
        
        distances = np.linalg.norm(states - state, axis=1)
        similar_idx = np.argmin(distances)
        
        if distances[similar_idx] < 1.0:
            analogical_action = actions[similar_idx]
            analogical_reward = rewards[similar_idx]
            
            intent = MetaIntention(
                intention_type=IntentionType.ANALOGICAL,
                base_action=analogical_action.copy(),
                target_feature="all",
                confidence=0.6 * (1 - distances[similar_idx]),
                novelty=0.3,
                fuzziness=distances[similar_idx],
                expected_outcome=state * 0.5 + np.pad(analogical_action, (0, len(state) - len(analogical_action))) * 0.5,
                explanation_confidence=0.5,
                causal_hypothesis=f"Analogous to state with reward {analogical_reward:.2f}",
                abandonment_threshold=0.5
            )
            intentions.append(intent)
        
        return intentions
    
    def _generate_combination_intentions(self, focus: AttentionFocus,
                                       experiences: List[Experience]) -> List[MetaIntention]:
        if len(experiences) < 5:
            return []
        
        intentions = []
        
        successful_exps = [exp for exp in experiences if exp.reward > 0]
        if len(successful_exps) >= 2:
            exp1, exp2 = random.sample(successful_exps, 2)
            
            alpha = random.uniform(0.3, 0.7)
            combined_action = alpha * exp1.action + (1 - alpha) * exp2.action
            
            intent = MetaIntention(
                intention_type=IntentionType.COMBINATION,
                base_action=combined_action.copy(),
                target_feature="all",
                confidence=0.5,
                novelty=0.7,
                fuzziness=0.4,
                expected_outcome=combined_action,
                explanation_confidence=0.4,
                causal_hypothesis=f"Combine {alpha:.2f}:{(1-alpha):.2f} of two successful actions",
                abandonment_threshold=0.6
            )
            intentions.append(intent)
        
        return intentions
    
    def _generate_counterfactual_intentions(self, experiences: List[Experience]) -> List[MetaIntention]:
        recent_experiences = experiences[-10:]
        failures = [exp for exp in recent_experiences if exp.reward < 0]
        
        if not failures:
            return []
        
        failed_exp = random.choice(failures)
        counterfactual_action = -failed_exp.action * random.uniform(0.5, 1.5)
        
        intent = MetaIntention(
            intention_type=IntentionType.COUNTERFACTUAL,
            base_action=counterfactual_action.copy(),
            target_feature="all",
            confidence=0.4,
            novelty=0.8,
            fuzziness=0.6,
            expected_outcome=counterfactual_action,
            explanation_confidence=0.3,
            causal_hypothesis=f"Opposite of failed action (reward: {failed_exp.reward:.2f})",
            abandonment_threshold=0.7
        )
        
        return [intent]
    
    def _generate_random_intention(self, focus: AttentionFocus) -> MetaIntention:
        random_action = np.random.uniform(-1, 1, self.action_dim)
        
        if focus.granularity == AttentionGranularity.FINE:
            confidence = 0.1
            novelty = 0.9
        elif focus.granularity == AttentionGranularity.COARSE:
            confidence = 0.05
            novelty = 0.95
        else:
            confidence = 0.15
            novelty = 0.85
        
        return MetaIntention(
            intention_type=IntentionType.RANDOM,
            base_action=random_action.copy(),
            target_feature="all",
            confidence=confidence,
            novelty=novelty,
            fuzziness=0.9,
            expected_outcome=random_action,
            explanation_confidence=0.1,
            causal_hypothesis="Random exploration for diversity",
            abandonment_threshold=0.2
        )
    
    def _get_intention_frequency(self, intention: MetaIntention) -> float:
        signature = self._intention_signature(intention)
        frequency = self.intention_history[signature]
        return min(1.0, frequency / 10.0)
    
    def _intention_signature(self, intention: MetaIntention) -> str:
        return f"{intention.intention_type.value}_{intention.target_feature}"

class MetaCognitiveMonitor:
    def __init__(self):
        self.performance_history = deque(maxlen=100)
        self.uncertainty_history = deque(maxlen=100)
        self.novelty_history = deque(maxlen=100)
        self.cognitive_state_history = deque(maxlen=50)
        
        self.current_state = CognitiveState.FOCUSED
        self.insight_count = 0
        self.confusion_count = 0
        self.learning_progress = 0.0
        
        self.confusion_threshold = 0.7
        self.insight_threshold = 0.8
        self.plateau_window = 20
    
    def update_state(self, reward: float, uncertainty: float, 
                    novelty: float, step: int) -> CognitiveState:
        self.performance_history.append(reward)
        self.uncertainty_history.append(uncertainty)
        self.novelty_history.append(novelty)
        
        if len(self.performance_history) >= 10:
            recent_perf = list(self.performance_history)[-10:]
            old_perf = list(self.performance_history)[-20:-10] if len(self.performance_history) >= 20 else recent_perf
            self.learning_progress = np.mean(recent_perf) - np.mean(old_perf)
        
        if uncertainty > self.confusion_threshold and novelty > 0.6:
            self.current_state = CognitiveState.CONFUSED
            self.confusion_count += 1
        elif self.learning_progress > 0.1 and uncertainty < 0.3:
            self.current_state = CognitiveState.INSIGHT
            self.insight_count += 1
        elif novelty > 0.7:
            self.current_state = CognitiveState.DIVERGENT
        else:
            self.current_state = CognitiveState.FOCUSED
        
        self.cognitive_state_history.append(self.current_state)
        
        return self.current_state
    
    def detect_plateau(self) -> bool:
        if len(self.performance_history) < self.plateau_window:
            return False
        
        recent_perf = list(self.performance_history)[-self.plateau_window:]
        if len(recent_perf) < 10:
            return False
        
        variance = np.var(recent_perf)
        if len(recent_perf) >= 2:
            x = np.arange(len(recent_perf))
            slope, _ = np.polyfit(x, recent_perf, 1)
        else:
            slope = 0
        
        return variance < 0.05 and abs(slope) < 0.01
    
    def get_metrics(self) -> Dict[str, float]:
        return {
            'cognitive_state': self.current_state.value,
            'learning_progress': self.learning_progress,
            'avg_recent_reward': np.mean(list(self.performance_history)[-10:]) if self.performance_history else 0,
            'avg_uncertainty': np.mean(self.uncertainty_history) if self.uncertainty_history else 0,
            'avg_novelty': np.mean(self.novelty_history) if self.novelty_history else 0,
            'insight_count': self.insight_count,
            'confusion_count': self.confusion_count
        }

class PriorityManager:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.priority_queue = []
        self.emergency_channel = EmergencyChannel()
        self.priority_history = deque(maxlen=100)
        self.emergency_history = deque(maxlen=50)
        
        self.params = {
            'priority_weights': {
                'confidence': 0.3,
                'novelty': 0.2,
                'urgency': 0.25,
                'cognitive_state': 0.15,
                'exploration_phase': 0.1
            },
            'emergency_thresholds': {
                'performance_drop': -0.5,
                'uncertainty': 0.8,
                'novelty': 0.9,
                'confusion': 0.7
            }
        }
    
    def calculate_priority_score(self, intention: MetaIntention, cognitive_state: CognitiveState, 
                               exploration_phase: ExplorationPhase, reward: float) -> float:
        base_score = {
            PriorityLevel.LOW: 0.2,
            PriorityLevel.MEDIUM: 0.5,
            PriorityLevel.HIGH: 0.8,
            PriorityLevel.CRITICAL: 1.0
        }[intention.priority]
        
        cognitive_bonus = {
            CognitiveState.INSIGHT: 0.2,
            CognitiveState.FOCUSED: 0.1,
            CognitiveState.DIVERGENT: -0.1,
            CognitiveState.CONFUSED: -0.2
        }[cognitive_state]
        
        exploration_bonus = {
            ExplorationPhase.EARLY: 0.1,
            ExplorationPhase.MIDDLE: 0.0,
            ExplorationPhase.LATE: -0.1
        }[exploration_phase]
        
        priority_score = (
            base_score +
            self.params['priority_weights']['confidence'] * intention.confidence +
            self.params['priority_weights']['novelty'] * intention.novelty +
            self.params['priority_weights']['urgency'] * intention.urgency +
            self.params['priority_weights']['cognitive_state'] * cognitive_bonus +
            self.params['priority_weights']['exploration_phase'] * exploration_bonus +
            random.uniform(-0.05, 0.05)
        )
        
        priority_score = np.clip(priority_score, 0.0, 1.0)
        return priority_score
    
    def add_intention(self, intention: MetaIntention, cognitive_state: CognitiveState, 
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
    
    def get_highest_priority_intention(self) -> Optional[MetaIntention]:
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
        if cognitive_state == CognitiveState.CONFUSED and uncertainty > self.params['emergency_thresholds']['uncertainty']:
            self.emergency_channel.activate(
                EmergencyType.CONFUSION_STATE,
                severity=uncertainty,
                recommended_action=np.zeros(self.action_dim),
                timestamp=timestamp
            )
            return True
        
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
    
    def _create_emergency_intention(self) -> MetaIntention:
        emergency_type = self.emergency_channel.emergency_type
        severity = self.emergency_channel.severity
        action = self.emergency_channel.recommended_action
        
        if action is None:
            action = np.random.uniform(-1, 1, self.action_dim)
        
        intention_type = {
            EmergencyType.PERFORMANCE_DROP: IntentionType.COUNTERFACTUAL,
            EmergencyType.HIGH_UNCERTAINTY: IntentionType.RANDOM,
            EmergencyType.NOVEL_SITUATION: IntentionType.ANALOGICAL,
            EmergencyType.CONFUSION_STATE: IntentionType.MODIFICATION
        }.get(emergency_type, IntentionType.RANDOM)
        
        emergency_intention = MetaIntention(
            intention_type=intention_type,
            base_action=action,
            target_feature="all",
            confidence=0.6,
            novelty=0.8,
            fuzziness=0.3,
            priority=PriorityLevel.CRITICAL,
            urgency=1.0,
            expected_outcome=action,
            explanation_confidence=0.5,
            causal_hypothesis=f"Emergency response to {emergency_type.value}",
            abandonment_threshold=0.2
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
    
    def batch_add_intentions(self, intentions: List[MetaIntention], cognitive_state: CognitiveState, 
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
    
    def task_sorting_channel(self, sort_by: str = 'priority') -> List[MetaIntention]:
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
        elif sort_by == 'novelty':
            items.sort(key=lambda x: x[2].intention.novelty, reverse=True)
        
        sorted_intentions = [item[2].intention for item in items]
        
        for item in items:
            heapq.heappush(self.priority_queue, item)
        
        return sorted_intentions
    
    def idea_sorting_channel(self, ideas: List[MetaIntention], cognitive_state: CognitiveState, 
                            exploration_phase: ExplorationPhase, reward: float) -> List[MetaIntention]:
        scored_ideas = []
        
        for idea in ideas:
            priority_score = self.calculate_priority_score(idea, cognitive_state, 
                                                         exploration_phase, reward)
            scored_ideas.append((-priority_score, idea))
        
        scored_ideas.sort(key=lambda x: x[0])
        sorted_ideas = [idea for _, idea in scored_ideas]
        
        return sorted_ideas
    
    def emergency_sort_override(self) -> List[MetaIntention]:
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