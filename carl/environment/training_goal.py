from enum import Enum


class TrainingGoal(Enum):
    POLICY = 'policy'
    VALUE = 'value'
    CLLP = 'cllp'
    GENERATOR = 'generator'
