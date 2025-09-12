from enum import Enum

class SelectionPolicy(str, Enum):
    GREEDY = "greedy"
    IMPORTANCE_SAMPLING = "importance_sampling"
    PAIRWISE_IMPORTANCE_SAMPLING = "pairwise_importance_sampling"

class InitialStrategy(str, Enum):
    ZERO_SHOT = "zero_shot"
    DUMMY_ANSWER = "dummy_answer"
