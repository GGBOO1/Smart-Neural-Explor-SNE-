from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

class AttentionGranularity(Enum):
    COARSE = "coarse"
    FINE = "fine"

class IntentionType(Enum):
    DIRECT = "direct"
    MODIFY = "modify"
    RANDOM = "random"

class CognitiveState(Enum):
    FOCUSED = "focused"
    EXPLORATORY = "exploratory"

class ExplorationPhase(Enum):
    EARLY = "early"
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

class Language(Enum):
    ENGLISH = "english"
    CHINESE = "chinese"

@dataclass
class Experience:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool

@dataclass
class AttentionFocus:
    state_dimensions: List[int]
    granularity: AttentionGranularity

@dataclass
class SimpleIntention:
    intention_type: IntentionType
    action: np.ndarray
    confidence: float
    priority: PriorityLevel = PriorityLevel.MEDIUM
    urgency: float = 0.0

@dataclass
class PriorityItem:
    intention: SimpleIntention
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