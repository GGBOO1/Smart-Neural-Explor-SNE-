from collections import defaultdict, deque
import numpy as np
from typing import List, Dict, Optional, Any
from sne_basics import SemanticNode, LongTermMemory, EmotionalState, ContextFrame

class SemanticNetwork:
    def __init__(self, decay_rate: float = 0.95, activation_threshold: float = 0.1):
        self.nodes = defaultdict(SemanticNode)
        self.decay_rate = decay_rate
        self.activation_threshold = activation_threshold
        self.association_strengths = defaultdict(float)
        self.long_term_memory = []
        
    def add_concept(self, concept: str, initial_activation: float = 0.0):
        if concept not in self.nodes:
            self.nodes[concept] = SemanticNode(concept=concept, activation=initial_activation)
    
    def add_association(self, concept1: str, concept2: str, strength: float):
        self.add_concept(concept1)
        self.add_concept(concept2)
        
        self.nodes[concept1].neighbors[concept2] = strength
        self.nodes[concept2].neighbors[concept1] = strength
        
        key = tuple(sorted([concept1, concept2]))
        self.association_strengths[key] = strength
    
    def activate_concept(self, concept: str, activation: float):
        if concept in self.nodes:
            self.nodes[concept].activation = max(self.nodes[concept].activation, activation)
    
    def spread_activation(self, steps: int = 3, max_activations: int = 10):
        activated_nodes = []
        for concept, node in self.nodes.items():
            if node.activation > self.activation_threshold:
                activated_nodes.append((concept, node.activation))
        
        for _ in range(steps):
            new_activations = {}
            for concept, activation in activated_nodes:
                if concept not in self.nodes:
                    continue
                
                node = self.nodes[concept]
                for neighbor, strength in node.neighbors.items():
                    spread_activation = activation * strength * 0.5
                    if spread_activation > self.activation_threshold:
                        if neighbor not in new_activations or spread_activation > new_activations[neighbor]:
                            new_activations[neighbor] = spread_activation
            
            for concept, activation in new_activations.items():
                self.activate_concept(concept, activation)
            
            activated_nodes = list(new_activations.items())[:max_activations]
        
        self.decay_activations()
    
    def decay_activations(self):
        for concept, node in self.nodes.items():
            node.activation *= self.decay_rate
            if node.activation < self.activation_threshold:
                node.activation = 0.0
    
    def get_active_concepts(self, threshold: float = 0.2) -> List[str]:
        active_concepts = []
        for concept, node in self.nodes.items():
            if node.activation > threshold:
                active_concepts.append(concept)
        return active_concepts
    
    def calculate_similarity(self, concept1: str, concept2: str) -> float:
        if concept1 not in self.nodes or concept2 not in self.nodes:
            return 0.0
        
        key = tuple(sorted([concept1, concept2]))
        if key in self.association_strengths:
            return self.association_strengths[key]
        
        node1 = self.nodes[concept1]
        node2 = self.nodes[concept2]
        
        common_neighbors = set(node1.neighbors.keys()) & set(node2.neighbors.keys())
        if not common_neighbors:
            return 0.0
        
        similarity = 0.0
        for neighbor in common_neighbors:
            strength1 = node1.neighbors.get(neighbor, 0.0)
            strength2 = node2.neighbors.get(neighbor, 0.0)
            similarity += min(strength1, strength2)
        
        return similarity / len(common_neighbors)
    
    def store_in_long_term_memory(self, content: Any, importance: float, timestamp: int, 
                                associations: List[str] = None, 
                                emotional_tag: EmotionalState = EmotionalState.NEUTRAL, 
                                context: Dict[str, Any] = None):
        memory = LongTermMemory(
            content=content,
            importance=importance,
            timestamp=timestamp,
            associations=associations,
            emotional_tag=emotional_tag,
            context=context
        )
        self.long_term_memory.append(memory)
        
        if len(associations) >= 2:
            for i in range(len(associations)):
                for j in range(i + 1, len(associations)):
                    self.add_association(associations[i], associations[j], importance * 0.5)
    
    def retrieve_from_long_term_memory(self, cue: str, context: ContextFrame = None, 
                                     max_results: int = 5) -> List[LongTermMemory]:
        scores = []
        for memory in self.long_term_memory:
            score = 0.0
            
            if memory.associations:
                for association in memory.associations:
                    similarity = self.calculate_similarity(cue, association)
                    score += similarity
            
            if context:
                if memory.context:
                    for key, value in memory.context.items():
                        if key in context.__dict__:
                            if str(value) == str(context.__dict__[key]):
                                score += 0.2
            
            score *= memory.importance
            if score > 0:
                scores.append((score, memory))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scores[:max_results]]
    
    def get_associative_strengths(self, concepts: List[str]) -> Dict[str, float]:
        strengths = {}
        for concept in concepts:
            if concept in self.nodes:
                total_strength = 0.0
                for neighbor, strength in self.nodes[concept].neighbors.items():
                    total_strength += strength
                strengths[concept] = total_strength / len(self.nodes[concept].neighbors) if self.nodes[concept].neighbors else 0.0
        return strengths
    
    def update_associations_from_experience(self, state: np.ndarray, action: np.ndarray, 
                                          reward: float, context: Dict[str, Any] = None):
        state_concepts = [f"state_dim_{i}_{state[i]:.2f}" for i in range(len(state))]
        action_concepts = [f"action_dim_{i}_{action[i]:.2f}" for i in range(len(action))]
        
        for concept in state_concepts + action_concepts:
            self.add_concept(concept)
        
        for i, state_concept in enumerate(state_concepts):
            for j, action_concept in enumerate(action_concepts):
                strength = abs(reward) * 0.1 if reward != 0 else 0.01
                self.add_association(state_concept, action_concept, strength)
        
        if reward > 0:
            emotional_tag = EmotionalState.POSITIVE
        elif reward < 0:
            emotional_tag = EmotionalState.NEGATIVE
        else:
            emotional_tag = EmotionalState.NEUTRAL
        
        self.store_in_long_term_memory(
            content={"state": state, "action": action, "reward": reward},
            importance=abs(reward) if reward != 0 else 0.1,
            timestamp=int(np.random.rand() * 1000000),
            associations=state_concepts + action_concepts,
            emotional_tag=emotional_tag,
            context=context
        )
    
    def get_semantic_map(self) -> Dict[str, List[str]]:
        semantic_map = {}
        for concept, node in self.nodes.items():
            neighbors = list(node.neighbors.keys())
            if neighbors:
                semantic_map[concept] = neighbors
        return semantic_map
    
    def clear_activations(self):
        for concept, node in self.nodes.items():
            node.activation = 0.0
    
    def prune_network(self, threshold: float = 0.05):
        concepts_to_remove = []
        for concept, node in self.nodes.items():
            if node.activation == 0.0 and len(node.neighbors) == 0:
                concepts_to_remove.append(concept)
        
        for concept in concepts_to_remove:
            del self.nodes[concept]
        
        associations_to_remove = []
        for key, strength in self.association_strengths.items():
            if strength < threshold:
                associations_to_remove.append(key)
        
        for key in associations_to_remove:
            del self.association_strengths[key]