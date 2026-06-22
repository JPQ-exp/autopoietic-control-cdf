"""
compiler/dsl.py — Universal, domain-agnostic declarative eDSL primitives.
"""
import numpy as np

class SpatialTarget:
    """Primitive 1: Represents an environmental landmark or potential field."""
    def __init__(self, name, coords, behavior="ATTRACTIVE", influence_radius=3.0, gain=2.0, physical_dock=None):
        self.name = name
        self.coords = np.array(coords, dtype=float)
        # Decouples pathfinding/steering coords (which may shift in queues) from physical interaction docks
        self.physical_dock = np.array(physical_dock, dtype=float) if physical_dock is not None else self.coords
        self.behavior = behavior                 # "ATTRACTIVE" or "REPULSIVE"
        self.influence_radius = influence_radius # Sigma in the RBF equation
        self.gain = gain                         # Speed/force scale factor


class Drive:
    """Primitive 2: Connects a SpatialTarget to a decision-making channel."""
    def __init__(self, name, target: SpatialTarget, gating_type="COGNITIVE", default_priority=4.0):
        self.name = name
        self.target = target
        self.gating_type = gating_type           # "COGNITIVE" or "REACTIVE"
        self.default_priority = default_priority # Baseline priority (S)


class FeedbackLoop:
    """Primitive 3: Couples continuous internal pools to behavioral drives."""
    def __init__(self, source_pool_name, target_drive: Drive, threshold, gain, max_bias, response_direction="INVERSE"):
        self.source_pool_name = source_pool_name # String identifier of any custom internal variable
        self.target_drive = target_drive
        self.threshold = threshold               # Sigmoidal midpoint
        self.gain = gain                         # Sigmoidal steepness
        self.max_bias = max_bias                 # Max priority boost
        self.response_direction = response_direction # "INVERSE" (depletion triggers) or "DIRECT" (accumulation triggers)


class LambdaInstruction:
    """Primitive 4: Dynamic rules for functional state/coordinate modifications."""
    def __init__(self, trigger_fn, modifier_fn, name="lambda_rule"):
        self.name = name
        self.trigger_fn = trigger_fn             # bool function: (pos, engine) -> bool
        self.modifier_fn = modifier_fn           # callback: (pos, v_actual, engine, dt) -> (new_pos, new_v_actual)


class State:
    """Declarative node in the System 2 State Machine."""
    def __init__(self, drive: Drive, on_reach=None, lambda_rule: LambdaInstruction = None):
        self.drive = drive                       # The primary Drive active in this state
        self.on_reach = on_reach                 # Name of the next State string
        self.lambda_rule = lambda_rule           # State-specific custom safety constraints


class StateMachine:
    """The high-level mission controller (System 2)."""
    def __init__(self, start_state: str):
        self.states = {}                         
        self.current_state_name = start_state

    def add_state(self, name: str, state_node: State):
        self.states[name] = state_node