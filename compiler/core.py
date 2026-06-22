"""
compiler/core.py — Universal core CDF Compiler Engine with Priority-Driven Resting Weights.
"""
import numpy as np
from compiler.gating import competitive_gate

class CDFCompilerEngine:
    def __init__(self, mission, initial_internal_states=None, metabolism_fn=None, bounds=None):
        self.mission = mission
        
        # Generic state variable storage
        self.internal_states = initial_internal_states if initial_internal_states is not None else {}
        
        # User-supplied integration function callback
        self.metabolism_fn = metabolism_fn 
        self.bounds = bounds # None or (min_bounds, max_bounds)
        
        self.global_drives = []                  
        self.feedback_loops = []
        self.lambdas = []
        
        self.all_drives = []
        self.w = None                            

    def register_background_drive(self, drive):
        self.global_drives.append(drive)

    def register_feedback_loop(self, loop):
        self.feedback_loops.append(loop)

    def register_lambda(self, instruction):
        self.lambdas.append(instruction)

    def compile_environment(self):
        """Compiles registered primitives into statically sized continuous matrices."""
        self.all_drives = list(self.global_drives)
        for state_node in self.mission.states.values():
            if state_node.drive not in self.all_drives:
                self.all_drives.append(state_node.drive)
                
        # Initialize weights to a neutral starting point
        self.w = np.full(len(self.all_drives), -1.0)

    def step(self, pos, dt):
        """Processes one complete, domain-agnostic step cycle of System 1 & 2."""
        current_state_node = self.mission.states[self.mission.current_state_name]
        active_state_drive = current_state_node.drive
        
        # 1. Evaluate Target Distances
        distances = {d.name: float(np.linalg.norm(pos - d.target.coords)) for d in self.all_drives}

        # 2. Check System 2 Mission Milestones
        target_name = active_state_drive.name
        if distances[target_name] < 0.6 and current_state_node.on_reach:
            next_state = current_state_node.on_reach
            if next_state != "SUCCESS":
                self.mission.current_state_name = next_state
                current_state_node = self.mission.states[self.mission.current_state_name]
                active_state_drive = current_state_node.drive

        # 3. Construct Priority Vector (S)
        S = np.full(len(self.all_drives), -2.0)
        for idx, drive in enumerate(self.all_drives):
            if drive == active_state_drive:
                dist_to_active = distances[drive.name]
                S[idx] = 8.5 if dist_to_active < 1.5 else drive.default_priority
            elif drive in self.global_drives:
                S[idx] = drive.default_priority

        # 4. Process Homeostatic Feedback Loops
        for loop in self.feedback_loops:
            val = self.internal_states.get(loop.source_pool_name, 0.0)
            if loop.response_direction == "INVERSE":
                bias = loop.max_bias / (1.0 + np.exp(loop.gain * (val - loop.threshold)))
            else:
                bias = loop.max_bias / (1.0 + np.exp(-loop.gain * (val - loop.threshold)))
            idx = self.all_drives.index(loop.target_drive)
            S[idx] += bias 

        # 5. Process State-Specific Lambdas
        max_speed_scale = 1.0
        if current_state_node.lambda_rule:
            rule = current_state_node.lambda_rule
            if rule.trigger_fn is None or rule.trigger_fn(pos, self):
                max_speed_scale = rule.modifier_fn(self)

        # 6. Calculate Lateral Inhibition Decision Gating (psi) with resting weights
        P = np.array([np.exp(-distances[d.name]**2 / d.target.influence_radius) for d in self.all_drives])
        
        # Dynamic Resting Weights: Cognitive drives rest at their current priority S, 
        # while reactive drives decay to -5.0 to fully suppress far-away peer repulsions.
        w_rest = np.array([-5.0 if d.gating_type == "REACTIVE" else S[idx] for idx, d in enumerate(self.all_drives)])
        
        f_w = S * P - 0.1 * (self.w - w_rest)
        self.w = self.w + f_w * dt
        
        reactive_mask = np.array([d.gating_type == "REACTIVE" for d in self.all_drives])
        psi = competitive_gate(self.w, reactive_mask, gamma=2.5)

        # 7. Generate Raw Control Vectors (u)
        u = []
        for d in self.all_drives:
            v_dir = d.target.coords - pos
            norm = np.linalg.norm(v_dir)
            if d.target.behavior == "ATTRACTIVE":
                u_vec = d.target.gain * v_dir / max(norm, 1e-3)
            else: # REPULSIVE
                # Regularized denominator prevents division-by-zero explosions
                u_vec = d.target.gain * (pos - d.target.coords) / (norm**2 + 0.5)
            u.append(u_vec)
            
        u_s = sum(p * ui for p, ui in zip(psi, u))
        if np.linalg.norm(u_s) > 2.0:
            u_s = 2.0 * u_s / np.linalg.norm(u_s)

        # 8. Couple Physical Dynamics
        v_actual = u_s * max_speed_scale
        
        # Execute Global Lambdas
        for lam in self.lambdas:
            if lam.trigger_fn(pos, self):
                pos, v_actual = lam.modifier_fn(pos, v_actual, self, dt)

        # Move Physical Entity
        pos = pos + v_actual * dt

        # Watertight Safety Sandbox boundary enforcement
        if self.bounds is not None:
            min_bounds, max_bounds = self.bounds
            pos = np.clip(pos, min_bounds, max_bounds)

        # 9. Integrate Homeostatic Core Updates via Strategy Callback
        if self.metabolism_fn is not None:
            self.metabolism_fn(self, kinetic_cost=float(v_actual @ v_actual), 
                               effort_cost=float(f_w @ f_w), 
                               dt=dt)

        return pos, psi