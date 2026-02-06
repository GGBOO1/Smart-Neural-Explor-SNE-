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

class EmotionalState(Enum):
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CURIOUS = "curious"
    FRUSTRATED = "frustrated"
    EXCITED = "excited"

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
    emotional_state: EmotionalState = EmotionalState.NEUTRAL
    context: Dict[str, Any] = None

@dataclass
class LongTermMemory:
    content: Any
    importance: float
    timestamp: int
    associations: List[str] = None
    emotional_tag: EmotionalState = EmotionalState.NEUTRAL
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.associations is None:
            self.associations = []
        if self.context is None:
            self.context = {}

@dataclass
class SemanticNode:
    concept: str
    activation: float
    neighbors: Dict[str, float] = None
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = {}

@dataclass
class ContextFrame:
    situation: str
    goals: List[str]
    constraints: List[str]
    emotional_valence: float = 0.0
    time_pressure: float = 0.0

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