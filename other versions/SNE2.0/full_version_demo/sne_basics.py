from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

class AttentionGranularity(Enum):
    COARSE = "coarse"
    FINE = "fine"
    MIXED = "mixed"

class IntentionType(Enum):
    DIRECTIONAL = "directional"
    MODIFICATION = "modification"
    COMBINATION = "combination"
    ANALOGICAL = "analogical"
    RANDOM = "random"
    COUNTERFACTUAL = "counterfactual"

class CognitiveState(Enum):
    FOCUSED = "focused"
    DIVERGENT = "divergent"
    CONFUSED = "confused"
    INSIGHT = "insight"

class ExplorationPhase(Enum):
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"

class PriorityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EmergencyType(Enum):
    NONE = "none"
    PERFORMANCE_DROP = "performance_drop"
    HIGH_UNCERTAINTY = "high_uncertainty"
    NOVEL_SITUATION = "novel_situation"
    CONFUSION_STATE = "confusion_state"

@dataclass
class Experience:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    novelty: float = 0.0
    uncertainty: float = 0.0
    timestamp: int = 0

@dataclass
class AttentionFocus:
    state_dimensions: List[int]
    action_features: List[str]
    granularity: AttentionGranularity
    persistence: float = 0.5
    curiosity: float = 0.3
    fatigue: float = 0.0
    duration: int = 0
    
    def __str__(self):
        return (f"AttentionFocus(dims={self.state_dimensions[:2]}, "
                f"granularity={self.granularity.value}, "
                f"persistence={self.persistence:.2f}, "
                f"curiosity={self.curiosity:.2f})")

@dataclass
class FuzzyIntention:
    intention_type: IntentionType
    base_action: np.ndarray
    target_feature: str
    confidence: float
    novelty: float
    fuzziness: float
    
    def __str__(self):
        return (f"{self.intention_type.value.capitalize()}Intention("
                f"conf={self.confidence:.2f}, "
                f"nov={self.novelty:.2f}, "
                f"fuzz={self.fuzziness:.2f})")

@dataclass 
class MetaIntention(FuzzyIntention):
    expected_outcome: np.ndarray = None
    explanation_confidence: float = 0.0
    causal_hypothesis: str = ""
    alternatives: List = None
    abandonment_threshold: float = 0.3
    priority: PriorityLevel = PriorityLevel.MEDIUM
    urgency: float = 0.0

@dataclass
class PriorityItem:
    intention: MetaIntention
    priority_score: float
    timestamp: int
    emergency_type: EmergencyType = EmergencyType.NONE

    def __lt__(self, other):
        return self.priority_score < other.priority_score

@dataclass
class EmergencyChannel:
    is_active: bool = False
    emergency_type: EmergencyType = EmergencyType.NONE
    severity: float = 0.0
    recommended_action: Optional[np.ndarray] = None
    trigger_timestamp: int = 0

    def activate(self, emergency_type: EmergencyType, severity: float, 
                recommended_action: Optional[np.ndarray], timestamp: int):
        self.is_active = True
        self.emergency_type = emergency_type
        self.severity = severity
        self.recommended_action = recommended_action
        self.trigger_timestamp = timestamp

    def deactivate(self):
        self.is_active = False
        self.emergency_type = EmergencyType.NONE
        self.severity = 0.0
        self.recommended_action = None
        self.trigger_timestamp = 0