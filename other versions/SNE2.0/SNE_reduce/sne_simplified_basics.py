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
