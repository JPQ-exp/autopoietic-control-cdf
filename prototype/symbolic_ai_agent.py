"""
prototype/symbolic_ai_agent.py — Corrected Hybrid Symbolic-Reactive Warehouse Robot.
"""

import sys
import os
import time
import numpy as np

# Resolve imports from the compiler directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from compiler.dsl import SpatialTarget, Drive, FeedbackLoop, LambdaInstruction, State, StateMachine
from compiler.core import CDFCompilerEngine
from compiler.integrator import lambert_w0, exact_linear_step

# =====================================================================
# FACTORY COORDINATES & CONSTANTS
# =====================================================================

STATIONS = {
    "charger": np.array([5.0, 1.0]),
    "pickup_a": np.array([2.0, 15.0]),
    "pickup_b": np.array([18.0, 15.0]),
    "assembly": np.array([10.0, 10.0]),
    "delivery": np.array([15.0, 2.0]),
    "obstacle": np.array([10.0, 6.0])
}

# =====================================================================
# SYSTEM 2: RULE-BASED SYMBOLIC AI INFERENCE LAYER
# =====================================================================

class SymbolicCognitivePlanner:
    def __init__(self):
        # Symbolic Belief-Desire-Intention State
        self.beliefs = {
            "battery_level": 100.0,
            "inventory": [],
            "pending_orders": ["ORDER_A", "ORDER_B"],
            "current_order": None,
            "active_intention": "IDLE"
        }
        self.last_intention = "IDLE"

    def execute_reasoning_cycle(self, pos, battery):
        """
        Translates continuous physical observations into symbolic beliefs,
        applies inference rules, and returns the next chosen high-level Intention.
        """
        self.beliefs["battery_level"] = battery

        # 1. Grounding: Map continuous physical spaces to discrete logical assertions
        at_charger  = np.linalg.norm(pos - STATIONS["charger"])  < 1.0
        at_pickup_a = np.linalg.norm(pos - STATIONS["pickup_a"]) < 1.0
        at_pickup_b = np.linalg.norm(pos - STATIONS["pickup_b"]) < 1.0
        at_assembly = np.linalg.norm(pos - STATIONS["assembly"]) < 1.0
        at_delivery = np.linalg.norm(pos - STATIONS["delivery"]) < 1.0

        battery_low      = battery < 35.0
        battery_critical = battery < 20.0
        battery_full     = battery > 95.0

        current_intention = self.beliefs["active_intention"]

        # 2. Forward-Chaining Inference Rules
        
        # RULE 1: Emergency low battery takes absolute precedence
        if battery_critical and current_intention != "RECHARGE":
            self.last_intention = current_intention
            self.beliefs["active_intention"] = "RECHARGE"
            print(f"  [Symbolic Inference]: Battery Critical ({battery:.1f}%). Interrupting to RECHARGE.")
            return

        # RULE 2: Preemptive recharge when low and idle
        if battery_low and current_intention == "IDLE":
            self.beliefs["active_intention"] = "RECHARGE"
            print("  [Symbolic Inference]: Low Battery. Preemptively heading to Charger.")
            return

        # RULE 3: Resume work once charging is complete
        if current_intention == "RECHARGE" and battery_full:
            if self.beliefs["current_order"] is not None:
                self.beliefs["active_intention"] = self.last_intention
                print(f"  [Symbolic Inference]: Battery Replete. Resuming task: {self.beliefs['active_intention']}.")
            else:
                self.beliefs["active_intention"] = "IDLE"
                print("  [Symbolic Inference]: Battery Replete. Returning to IDLE.")
            return

        # RULE 4: Select a new order if idle and orders are available
        if current_intention == "IDLE" and self.beliefs["pending_orders"]:
            next_order = self.beliefs["pending_orders"].pop(0)
            self.beliefs["current_order"] = next_order
            if next_order == "ORDER_A":
                self.beliefs["active_intention"] = "PICK_A"
            else:
                self.beliefs["active_intention"] = "PICK_B"
            print(f"  [Symbolic Inference]: Selected new order {next_order}. Intention set to {self.beliefs['active_intention']}.")
            return

        # RULE 5: Pick up parts when arriving at pickup stations
        if current_intention == "PICK_A" and at_pickup_a:
            self.beliefs["inventory"].append("raw_parts_a")
            self.beliefs["active_intention"] = "GO_ASSEMBLY"
            print("  [Symbolic Inference]: Parts A Picked Up. Heading to Assembly.")
            return
            
        if current_intention == "PICK_B" and at_pickup_b:
            self.beliefs["inventory"].append("raw_parts_b")
            self.beliefs["active_intention"] = "GO_ASSEMBLY"
            print("  [Symbolic Inference]: Parts B Picked Up. Heading to Assembly.")
            return

        # RULE 6: Assemble product at assembly station
        if current_intention == "GO_ASSEMBLY" and at_assembly:
            if "raw_parts_a" in self.beliefs["inventory"]:
                self.beliefs["inventory"].remove("raw_parts_a")
            if "raw_parts_b" in self.beliefs["inventory"]:
                self.beliefs["inventory"].remove("raw_parts_b")
            self.beliefs["inventory"].append("assembled_product")
            self.beliefs["active_intention"] = "GO_DELIVERY"
            print("  [Symbolic Inference]: Assembly completed. Heading to Delivery.")
            return

        # RULE 7: Complete delivery
        if current_intention == "GO_DELIVERY" and at_delivery:
            self.beliefs["inventory"].remove("assembled_product")
            completed_order = self.beliefs["current_order"]
            self.beliefs["current_order"] = None
            self.beliefs["active_intention"] = "IDLE"
            print(f"  [Symbolic Inference]: Order {completed_order} successfully DELIVERED. Returning to IDLE.")
            return

# =====================================================================
# METABOLIC ENERGETICS
# =====================================================================

def evaluate_tripartite_metabolism(atp, structure, precursor, kinetic_cost, effort_cost, is_charging, dt):
    atp_max = 10.0
    structure_max = 100.0
    precursor_max = 100.0
    k_m = 50.0

    # FIX: Apply numerical safety clamps to prevent boundary division-by-zero
    atp = float(np.clip(atp, 1e-9, atp_max - 1e-9))
    structure = float(np.clip(structure, 1e-9, structure_max - 1e-9))
    precursor = float(np.clip(precursor, 0.0, precursor_max))

    # 1. Compute Autopoietic Metabolic Rates
    chi = 1.0 / (1.0 + np.exp(0.5 * (precursor - 15.0)))
    r_cat = 0.25 * structure * chi
    r_ana = 0.15 * atp * (precursor / (precursor + 5.0)) * (1.0 - structure / structure_max)

    # --- Precursor Pool ---
    g_c = precursor * np.exp(precursor / k_m)
    if is_charging:
        forcing_physical = 0.8 * precursor_max * 0.35
    else:
        stomatal_resistance = np.clip(precursor / 40.0, 0.15, 1.0)
        forcing_physical = -(0.06 + 1.2 * kinetic_cost) * stomatal_resistance
    forcing_physical += 0.1 * r_cat - 0.2 * r_ana
    dg_c = forcing_physical * (1.0 + precursor / k_m) * np.exp(precursor / k_m)
    g_c_next = exact_linear_step(g_c, rate=-0.05, forcing=dg_c, dt=dt)
    precursor_next = k_m * lambert_w0(max(g_c_next, 0.0) / k_m)

    # --- ATP Pool (Ratio Bounded) ---
    g_e = atp / (atp_max - atp)  # Now completely safe from division-by-zero
    c_active = 0.05 * kinetic_cost + 0.02 * effort_cost + 0.3 * r_ana - 0.6 * r_cat
    c_g = ((1.0 + g_e) ** 2 / atp_max) * c_active
    lam = 0.4 * np.tanh(precursor / 25.0)
    g_e_next = exact_linear_step(g_e, rate=lam, forcing=-c_g, dt=dt)
    g_e_next = max(g_e_next, 1e-9)
    atp_next = atp_max * g_e_next / (1.0 + g_e_next)

    # --- Structure Pool ---
    a_struct = 0.15 * atp * (precursor / (precursor + 5.0))
    b_struct = a_struct / structure_max + 0.25 * chi + 0.02
    structure_eq = a_struct / b_struct
    structure_next = structure_eq + (structure - structure_eq) * np.exp(-b_struct * dt)

    # Apply standard exit clamps before returning updated values
    return (
        float(np.clip(atp_next, 1e-9, atp_max - 1e-9)),
        float(np.clip(structure_next, 1e-9, structure_max - 1e-9)),
        float(np.clip(precursor_next, 0.0, precursor_max))
    )

# =====================================================================
# SIMULATION ENGINE RUNNER
# =====================================================================

def run_hybrid_agent_simulation():
    print("Initializing Hybrid Cognitive Architecture...")
    bounds = (np.array([0.0, 0.0]), np.array([20.0, 20.0]))
    initial_pos = np.array([5.0, 1.0])  # Starts docked at the charger
    dt = 0.1
    
    # 1. Instantiate the Symbolic AI layer
    planner = SymbolicCognitivePlanner()
    
    # 2. Define low-level potential fields (Drives) for every location
    pickup_a_drive = Drive("pickup_a", SpatialTarget("pickup_a", STATIONS["pickup_a"], gain=2.5))
    pickup_b_drive = Drive("pickup_b", SpatialTarget("pickup_b", STATIONS["pickup_b"], gain=2.5))
    assembly_drive = Drive("assembly", SpatialTarget("assembly", STATIONS["assembly"], gain=2.5))
    delivery_drive = Drive("delivery", SpatialTarget("delivery", STATIONS["delivery"], gain=2.5))
    recharge_drive = Drive("recharge", SpatialTarget("charger", STATIONS["charger"], gain=2.5))
    idle_drive     = Drive("idle",     SpatialTarget("charger_rest", STATIONS["charger"], gain=1.5))

    # Static Reactive Background Obstacle Drive
    obstacle_target = SpatialTarget("obstacle", STATIONS["obstacle"], behavior="REPULSIVE", influence_radius=1.8, gain=3.5)
    obstacle_drive  = Drive("avoid_obstacle", obstacle_target, gating_type="REACTIVE", default_priority=4.5)

    # 3. Create StateMachine registering ALL possible states before compilation
    mission = StateMachine(start_state="STATE_IDLE")
    mission.add_state("STATE_IDLE", State(drive=idle_drive))
    mission.add_state("STATE_PICK_A", State(drive=pickup_a_drive))
    mission.add_state("STATE_PICK_B", State(drive=pickup_b_drive))
    mission.add_state("STATE_GO_ASSEMBLY", State(drive=assembly_drive))
    mission.add_state("STATE_GO_DELIVERY", State(drive=delivery_drive))
    mission.add_state("STATE_RECHARGE", State(drive=recharge_drive))

    # Map high-level Intention keys to the compiled State names
    intention_state_map = {
        "PICK_A": "STATE_PICK_A",
        "PICK_B": "STATE_PICK_B",
        "GO_ASSEMBLY": "STATE_GO_ASSEMBLY",
        "GO_DELIVERY": "STATE_GO_DELIVERY",
        "RECHARGE": "STATE_RECHARGE",
        "IDLE": "STATE_IDLE"
    }

    initial_states = {
        "atp": 10.0,
        "structure": 90.0,
        "precursor": 100.0,
        "pos": initial_pos
    }

    def robot_metabolism(engine, kinetic_cost, effort_cost, dt):
        atp, struct, prec = evaluate_tripartite_metabolism(
            engine.internal_states["atp"],
            engine.internal_states["structure"],
            engine.internal_states["precursor"],
            kinetic_cost, effort_cost,
            is_charging=(np.linalg.norm(engine.internal_states["pos"] - STATIONS["charger"]) < 1.0),
            dt=dt
        )
        engine.internal_states["atp"] = atp
        engine.internal_states["structure"] = struct
        engine.internal_states["precursor"] = prec

    engine = CDFCompilerEngine(
        mission=mission,
        initial_internal_states=initial_states,
        metabolism_fn=robot_metabolism,
        bounds=bounds
    )
    # Register reactive collision avoidance
    engine.register_background_drive(obstacle_drive)
    engine.compile_environment()

    pos = initial_pos
    
    print("\n======================================================================")
    print("               STARTING WAREHOUSE RUNTIME SIMULATION")
    print("======================================================================")

    for step in range(500):
        # Read low-level physical observations
        battery = engine.internal_states["precursor"]  # Mapping Precursor directly as raw battery storage
        engine.internal_states["pos"] = pos

        # System 2 Reasoning Cycle (runs at every step to monitor beliefs)
        planner.execute_reasoning_cycle(pos, battery)
        active_intention = planner.beliefs["active_intention"]

        # Dynamic State Transition:
        # Instead of replacing the drive, we simply update the state name
        engine.mission.current_state_name = intention_state_map[active_intention]

        # Execute low-level physics step
        pos, psi = engine.step(pos, dt)

        if step % 20 == 0:
            inv_str = str(planner.beliefs["inventory"]) if planner.beliefs["inventory"] else "Empty"
            orders_left = len(planner.beliefs["pending_orders"])
            print(f"Step {step:03d} | Pos: {pos.round(1)} | Battery: {battery:.1f}% | "
                  f"Inventory: {inv_str} | Intention: {active_intention} | Pending Orders: {orders_left}")
            time.sleep(0.01)

    print("\n======================================================================")
    print("                    SIMULATION COMPLETE")
    print("======================================================================")

if __name__ == "__main__":
    run_hybrid_agent_simulation()