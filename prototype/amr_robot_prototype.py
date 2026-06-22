"""
prototype/factory_robot.py — Kinematic Factory Robot Simulation using the CDF Compiler.
"""

import sys
import os
import time
import numpy as np

# Automatically resolve imports by adding the parent folder of 'prototype' to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from compiler.dsl import SpatialTarget, Drive, FeedbackLoop, LambdaInstruction, State, StateMachine
from compiler.core import CDFCompilerEngine


def factory_metabolism(engine, kinetic_cost, effort_cost, dt):
    """
    Simulates continuous battery consumption and charging behavior.
    """
    battery = engine.internal_states.get("battery", 100.0)
    pos = engine.internal_states.get("pos", np.array([0.0, 0.0]))
    
    # Check proximity to the charging station target coordinates
    charger_drive = next((d for d in engine.all_drives if d.name == "recharge"), None)
    is_charging = False
    
    if charger_drive is not None:
        dist_to_charger = np.linalg.norm(pos - charger_drive.target.coords)
        # If the robot is close enough to the physical dock and gating is actively targeting it
        if dist_to_charger < 0.6:
            is_charging = True
            
    if is_charging:
        # Rapidly replenish battery when docked
        battery = min(100.0, battery + 22.0 * dt)
    else:
        # Passive drain + kinetic consumption
        battery = max(0.0, battery - (1.2 + 2.0 * kinetic_cost) * dt)
        
    engine.internal_states["battery"] = battery


def build_factory_mission():
    """
    Declares the factory floor spatial environment with a Cognitive Recharge Drive
    to allow complete lateral inhibition when the battery is full.
    """
    # 1. Define Spatial Targets (Landmarks / Fields)
    pickup_target = SpatialTarget("pickup_station", coords=[1.0, 8.0], behavior="ATTRACTIVE", influence_radius=3.0, gain=2.0)
    assembly_target = SpatialTarget("assembly_station", coords=[5.0, 5.0], behavior="ATTRACTIVE", influence_radius=3.0, gain=2.0)
    delivery_target = SpatialTarget("delivery_station", coords=[9.0, 2.0], behavior="ATTRACTIVE", influence_radius=3.0, gain=2.0)
    
    # Background Target 1: The charging dock (wide catchment field remains the same)
    charger_target = SpatialTarget("charger_dock", coords=[5.0, 1.0], behavior="ATTRACTIVE", influence_radius=100.0, gain=2.0)
    
    # Background Target 2: Relocated and narrowed obstacle to prevent interference with the charger
    obstacle_target = SpatialTarget("hazardous_obstacle", coords=[6.5, 3.5], behavior="REPULSIVE", influence_radius=1.0, gain=3.0)

    # 2. Map Targets to Gated Decision Channels (Drives)
    pickup_drive = Drive("pickup", target=pickup_target, gating_type="COGNITIVE", default_priority=4.0)
    assembly_drive = Drive("assemble", target=assembly_target, gating_type="COGNITIVE", default_priority=4.0)
    delivery_drive = Drive("deliver", target=delivery_target, gating_type="COGNITIVE", default_priority=4.0)
    
    # ADJUSTMENT: Recharge is now COGNITIVE with a negative default priority baseline
    recharge_drive = Drive("recharge", target=charger_target, gating_type="COGNITIVE", default_priority=-3.0)
    
    # Obstacle remains REACTIVE so it bypasses planning and always pushes the robot away
    obstacle_drive = Drive("avoid_obstacle", target=obstacle_target, gating_type="REACTIVE", default_priority=3.0)

    # 3. Establish System 2 Mission State Machine
    mission = StateMachine(start_state="STATE_PICKUP")
    mission.add_state("STATE_PICKUP", State(drive=pickup_drive, on_reach="STATE_ASSEMBLY"))
    mission.add_state("STATE_ASSEMBLY", State(drive=assembly_drive, on_reach="STATE_DELIVERY"))
    mission.add_state("STATE_DELIVERY", State(drive=delivery_drive, on_reach="STATE_PICKUP"))

    # 4. Connect Biological Homeostasis to Steering Urgency (Feedback Loops)
    # The loop will boost the priority of 'recharge_drive' by up to +25.0 when low, 
    # bringing the overall priority from -3.0 up to +22.0, successfully overriding mission tasks.
    battery_loop = FeedbackLoop(
        source_pool_name="battery",
        target_drive=recharge_drive,
        threshold=35.0,        
        gain=0.18,             
        max_bias=25.0,         
        response_direction="INVERSE"
    )

    # 5. Safety Lambda Rules
    def low_battery_trigger(pos, engine):
        return engine.internal_states.get("battery", 100.0) < 20.0

    def low_battery_modifier(pos, v_actual, engine, dt):
        battery = engine.internal_states.get("battery", 100.0)
        scale_factor = max(0.1, battery / 20.0)
        return pos, v_actual * scale_factor

    safety_lambda = LambdaInstruction(
        trigger_fn=low_battery_trigger,
        modifier_fn=low_battery_modifier,
        name="low_battery_slowdown"
    )

    return mission, recharge_drive, obstacle_drive, battery_loop, safety_lambda


def render_factory_floor(pos, engine, active_state):
    """
    Generates a low-overhead ASCII layout showing the current state of the factory floor.
    """
    grid = [["." for _ in range(10)] for _ in range(10)]
    
    # Define landmark coordinates for visualization
    # Rounded coordinate representation for the ASCII map
    landmarks = {
        "P": (1, 8),  # Pickup
        "A": (5, 5),  # Assembly
        "D": (9, 2),  # Delivery
        "C": (5, 1),  # Charger
        "X": (6, 4)   # Obstacle (6.5, 3.5 rounded to nearest grid cell)
    }
    
    for marker, (x, y) in landmarks.items():
        grid[9 - y][x] = marker
        
    # Map robot current location
    rx = int(np.clip(round(pos[0]), 0, 9))
    ry = int(np.clip(round(pos[1]), 0, 9))
    grid[9 - ry][rx] = "R"
    
    # Format and display
    border = "+" + "---" * 10 + "+"
    print(f"\nActive State: {active_state} | Battery: {engine.internal_states['battery']:.1f}%")
    print(border)
    for row in grid:
        print("| " + "  ".join(row) + " |")
    print(border)


def run_simulation():
    # Initial Configuration
    initial_pos = np.array([1.0, 1.0], dtype=float)
    dt = 0.1
    steps = 400
    
    # Initialize Core Engine
    mission, recharge_drive, obstacle_drive, battery_loop, safety_lambda = build_factory_mission()
    initial_states = {"battery": 100.0, "pos": initial_pos}
    
    bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0])) # Watertight boundaries
    
    engine = CDFCompilerEngine(
        mission=mission,
        initial_internal_states=initial_states,
        metabolism_fn=factory_metabolism,
        bounds=bounds
    )
    
    # Register Background Drivers, Loops, and Safety Rules
    engine.register_background_drive(recharge_drive)
    engine.register_background_drive(obstacle_drive)
    engine.register_feedback_loop(battery_loop)
    engine.register_lambda(safety_lambda)
    
    # Compile
    engine.compile_environment()
    
    pos = initial_pos
    print("Factory compiler initialization finished. Starting simulation run...")
    
    for step_idx in range(steps):
        # Update current coordinates for the metabolism tracker
        engine.internal_states["pos"] = pos
        
        # Capture current mission state before integration step
        active_state = engine.mission.current_state_name
        
        # Advance the simulation step
        pos, psi = engine.step(pos, dt)
        
        # Render visual layout and diagnostics occasionally to track behavior
        if step_idx % 8 == 0:
            render_factory_floor(pos, engine, active_state)
            
            # Print decision weights
            active_drive_names = [d.name for d in engine.all_drives]
            gating_states = {name: float(psi[idx]) for idx, name in enumerate(active_drive_names)}
            formatted_gating = ", ".join([f"{k}: {v:.2f}" for k, v in gating_states.items()])
            print(f"Gating Channels (psi) -> {formatted_gating}")
            time.sleep(0.1)


if __name__ == "__main__":
    run_simulation()