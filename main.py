"""
main.py — Interactive CDF v2-UL Agent Compiler Visualization Dashboard.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Resolve parent directory to allow direct imports of compiler and prototype modules
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import compiler elements
from compiler.dsl import SpatialTarget, Drive, State, StateMachine
from compiler.core import CDFCompilerEngine
from compiler.integrator import lambert_w0, exact_linear_step

# Import prototype elements
import prototype.amr_robot_prototype as amr
import prototype.bio_simulations as bio
import prototype.symbolic_ai_agent as sym

# =====================================================================
# VISUALIZATION WRAPPER: AMR FACTORY FLOOR
# =====================================================================

def run_and_visualize_amr():
    print("\nExecuting AMR Robot Floor Simulation...")
    
    # Initialize Engine
    mission, recharge_drive, obstacle_drive, battery_loop, safety_lambda = amr.build_factory_mission()
    bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    initial_pos = np.array([1.0, 1.0])
    dt = 0.1
    steps = 400

    initial_states = {"battery": 100.0, "pos": initial_pos}
    
    engine = CDFCompilerEngine(
        mission=mission,
        initial_internal_states=initial_states,
        metabolism_fn=amr.factory_metabolism,
        bounds=bounds
    )
    engine.register_background_drive(recharge_drive)
    engine.register_background_drive(obstacle_drive)
    engine.register_feedback_loop(battery_loop)
    engine.register_lambda(safety_lambda)
    engine.compile_environment()

    # Track histories
    traj_x, traj_y = [], []
    battery_history = []
    pos = initial_pos

    for step in range(steps):
        engine.internal_states["pos"] = pos
        pos, _ = engine.step(pos, dt)
        
        traj_x.append(pos[0])
        traj_y.append(pos[1])
        battery_history.append(engine.internal_states["battery"])

    # Matplotlib Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("AMR Robot Factory Floor Navigation & Energetics", fontsize=14, fontweight='bold')

    # Subplot 1: Trajectory Map
    ax1.set_title("2D Factory Floor Trajectory")
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-0.5, 10.5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Draw Stations
    stations = {
        "Pickup": (1.0, 8.0, "blue", "P"),
        "Assembly": (5.0, 5.0, "orange", "A"),
        "Delivery": (9.0, 2.0, "green", "D"),
        "Charger": (5.0, 1.0, "red", "C"),
        "Obstacle": (6.5, 3.5, "purple", "X")
    }
    for name, (x, y, color, marker) in stations.items():
        ax1.plot(x, y, marker="o", color=color, markersize=12)
        ax1.text(x + 0.2, y + 0.2, f"{name} ({marker})", fontsize=10, fontweight="bold")
    
    # Draw Trajectory
    sc = ax1.scatter(traj_x, traj_y, c=battery_history, cmap="RdYlGn", s=6, label="Path (Color: Battery %)")
    ax1.plot(traj_x, traj_y, color="black", alpha=0.3, linestyle=":")
    ax1.plot(initial_pos[0], initial_pos[1], marker="X", color="cyan", markersize=10, label="Start")
    ax1.legend(loc="upper right")
    cbar = fig.colorbar(sc, ax=ax1)
    cbar.set_label("Battery Level (%)", rotation=275, labelpad=15)

    # Subplot 2: Battery Timeline
    ax2.set_title("Battery Depletion / Charging Timeline")
    ax2.plot(np.arange(steps) * dt, battery_history, color="darkgreen", linewidth=2, label="Battery %")
    ax2.axhline(35.0, color="orange", linestyle="--", alpha=0.7, label="Low Battery Threshold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Battery (%)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# =====================================================================
# VISUALIZATION WRAPPER: BIOLOGICAL PARADIGMS
# =====================================================================

def run_bio_forager_plot():
    # Setup
    bounds = (np.array([0.0, 0.0]), np.array([20.0, 20.0]))
    initial_pos = np.array([1.0, 1.0])
    dt = 0.1
    steps = 120
    
    food_target = SpatialTarget("food", coords=[18.0, 18.0], behavior="ATTRACTIVE", influence_radius=6.0, gain=2.5)
    threat_target = SpatialTarget("predator", coords=[10.0, 10.0], behavior="REPULSIVE", influence_radius=4.0, gain=4.0)
    forage_drive = Drive("forage", target=food_target, gating_type="COGNITIVE", default_priority=4.5)
    evasion_drive = Drive("evade", target=threat_target, gating_type="REACTIVE", default_priority=5.0)

    mission = StateMachine(start_state="FORAGING")
    mission.add_state("FORAGING", State(drive=forage_drive))

    initial_states = {"atp": 9.5, "structure": 80.0, "precursor": 30.0, "pos": initial_pos}

    def forager_metabolism(engine, kinetic_cost, effort_cost, dt):
        atp, struct, prec = bio.evaluate_tripartite_metabolism(
            engine.internal_states["atp"], engine.internal_states["structure"], engine.internal_states["precursor"],
            kinetic_cost, effort_cost,
            is_charging=(np.linalg.norm(engine.internal_states["pos"] - food_target.coords) < 1.0),
            dt=dt
        )
        engine.internal_states["atp"] = atp
        engine.internal_states["structure"] = struct
        engine.internal_states["precursor"] = prec

    engine = CDFCompilerEngine(mission=mission, initial_internal_states=initial_states, metabolism_fn=forager_metabolism, bounds=bounds)
    engine.register_background_drive(evasion_drive)
    engine.compile_environment()

    # Log variables
    traj_agent = []
    traj_predator = []
    hist_atp, hist_biomass, hist_precursor = [], [], []
    pos = initial_pos
    predator_pos = np.array([10.0, 10.0])

    for _ in range(steps):
        engine.internal_states["pos"] = pos
        dist_to_agent = np.linalg.norm(pos - predator_pos)
        if dist_to_agent < 5.0:
            predator_pos += ((pos - predator_pos) / (dist_to_agent + 1e-3)) * 1.5 * dt
        else:
            predator_pos += np.random.uniform(-0.5, 0.5, size=2) * dt
            predator_pos = np.clip(predator_pos, 2.0, 18.0)
            
        threat_target.coords = predator_pos
        pos, _ = engine.step(pos, dt)
        
        traj_agent.append(np.copy(pos))
        traj_predator.append(np.copy(predator_pos))
        hist_atp.append(engine.internal_states["atp"])
        hist_biomass.append(engine.internal_states["structure"])
        hist_precursor.append(engine.internal_states["precursor"])

    traj_agent = np.array(traj_agent)
    traj_predator = np.array(traj_predator)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Objective 1: Forager Agent Escape & Foraging Navigation", fontsize=14, fontweight="bold")
    
    ax1.set_title("Tactical Space Tracking Map")
    ax1.plot(traj_agent[:, 0], traj_agent[:, 1], color="blue", label="Foraging Agent (Mouse)")
    ax1.plot(traj_predator[:, 0], traj_predator[:, 1], color="red", linestyle="--", label="Predator (Weasel)")
    ax1.scatter(food_target.coords[0], food_target.coords[1], color="green", marker="*", s=250, label="Food Patch")
    ax1.set_xlim(-0.5, 20.5)
    ax1.set_ylim(-0.5, 20.5)
    ax1.grid(True, linestyle="--")
    ax1.legend()

    ax2.set_title("Metabolic Pool Timelines")
    time_axis = np.arange(steps) * dt
    ax2.plot(time_axis, hist_atp, color="orange", label="ATP (Energy)")
    ax2.plot(time_axis, hist_biomass, color="brown", label="Structure (Somatic Biomass)")
    ax2.plot(time_axis, hist_precursor, color="blue", label="Precursor (Digestive Pool)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Quantity")
    ax2.grid(True, linestyle="--")
    ax2.legend()
    plt.show()

def run_bio_plant_plot():
    # Setup
    dt = 0.5
    steps = 320
    atp, structure, precursor = 9.0, 20.0, 90.0
    
    hist_atp, hist_structure, hist_precursor = [], [], []

    for step in range(steps):
        is_charging = (step > 160)  # Drought ends at step 160
        atp, structure, precursor = bio.evaluate_tripartite_metabolism(
            atp, structure, precursor, kinetic_cost=0.0, effort_cost=0.01, is_charging=is_charging, dt=dt
        )
        hist_atp.append(atp)
        hist_structure.append(structure)
        hist_precursor.append(precursor)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    time_axis = np.arange(steps) * dt
    ax.axvspan(0, 160*dt, color="red", alpha=0.1, label="Drought Phase")
    ax.axvspan(160*dt, steps*dt, color="blue", alpha=0.1, label="Rainfall Recovery Phase")
    
    ax.plot(time_axis, hist_atp, color="orange", linewidth=2.5, label="ATP (Turgor Pressure)")
    ax.plot(time_axis, hist_structure, color="brown", linewidth=2.5, label="Dry Biomass (Structure)")
    ax.plot(time_axis, hist_precursor, color="teal", linewidth=2.5, label="Wet Precursor (Internal Hydration)")
    
    ax.set_title("Objective 2: Growing Plant Drought Resistance & Senescence", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Steps (Relative)")
    ax.set_ylabel("Simulated Pool Capacity")
    ax.grid(True, linestyle=":")
    ax.legend()
    plt.show()

def run_bio_fungi_plot():
    # Nodes configuration mirroring prototype setup
    root_coords = np.array([10.0, 10.0])
    food_patches = [
        np.array([2.0, 15.0]), np.array([7.0, 18.0]),
        np.array([14.0, 17.0]), np.array([18.0, 11.0]),
        np.array([11.0, 2.0])
    ]

    # Re-run a localized tracking of the fungal growth structure
    nodes = {0: bio.MycelialNode(0, root_coords)}
    next_node_id = 1
    active_tips = []
    for theta in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        tip_pos = root_coords + np.array([np.cos(theta), np.sin(theta)])
        node = bio.MycelialNode(next_node_id, tip_pos, parent_id=0)
        nodes[0].children.append(next_node_id)
        nodes[next_node_id] = node
        active_tips.append(next_node_id)
        next_node_id += 1

    dt = 1.0
    gamma = 0.15

    for _ in range(40):
        new_tips = []
        for tip_id in active_tips:
            tip_node = nodes[tip_id]
            if not tip_node.is_active:
                continue
            dists = [np.linalg.norm(tip_node.coords - f) for f in food_patches]
            if not dists:
                continue
            closest_idx = np.argmin(dists)
            target_food = food_patches[closest_idx]
            direction = target_food - tip_node.coords
            dist_to_food = np.linalg.norm(direction)
            on_food = dist_to_food < 1.2
            
            if dist_to_food > 0.5:
                step_dir = (direction / (dist_to_food + 1e-5)) * 0.8
                new_coords = tip_node.coords + step_dir
                new_node = bio.MycelialNode(next_node_id, new_coords, parent_id=tip_id)
                nodes[tip_id].children.append(next_node_id)
                nodes[next_node_id] = new_node
                new_tips.append(next_node_id)
                next_node_id += 1
                tip_node.is_active = False 
            else:
                new_tips.append(tip_id)
        if new_tips:
            active_tips = new_tips

        for _ in range(3):
            for n_id, node in list(nodes.items()):
                if n_id == 0 or not node.is_active:
                    continue
                parent = nodes[node.parent_id]
                flow = gamma * (node.precursor - parent.precursor) * (node.structure / 100.0)
                node.precursor -= flow * dt
                parent.precursor += flow * dt

        for n_id, node in list(nodes.items()):
            on_food = any(np.linalg.norm(node.coords - f) < 1.2 for f in food_patches)
            node.atp, node.structure, node.precursor = bio.evaluate_tripartite_metabolism(
                node.atp, node.structure, node.precursor, kinetic_cost=0.01, effort_cost=0.01, is_charging=on_food, dt=dt
            )
            if node.structure < 1.5 and n_id != 0:
                node.is_active = False
                parent = nodes[node.parent_id]
                if n_id in parent.children:
                     parent.children.remove(n_id)

    # Resolve Greedy TSP path for visual overlay
    current = np.array(root_coords)
    unvisited = list(food_patches)
    tsp_points = [current]
    while unvisited:
        nearest_idx = min(range(len(unvisited)), key=lambda i: np.linalg.norm(current - unvisited[i]))
        nearest_node = unvisited.pop(nearest_idx)
        current = nearest_node
        tsp_points.append(current)
    tsp_points = np.array(tsp_points)

    # Plot Comparison
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Objective 3: Branching Mycelial Network vs Sequential TSP Routing", fontsize=12, fontweight="bold")
    
    # Plot TSP
    ax.plot(tsp_points[:, 0], tsp_points[:, 1], color="red", linestyle="--", linewidth=2.5, marker="s", markersize=8, label="Greedy TSP Path (Min Construction)")
    
    # Plot Active Mycelium Network links
    for n_id, node in nodes.items():
        if node.structure >= 1.5 and node.parent_id is not None:
            parent = nodes[node.parent_id]
            ax.plot([node.coords[0], parent.coords[0]], [node.coords[1], parent.coords[1]], color="brown", linewidth=3.5, alpha=0.8)
            
    # Draw Nodes and Labels
    ax.scatter(root_coords[0], root_coords[1], color="purple", marker="^", s=300, zorder=5, label="Spore Origin (Root)")
    for idx, f in enumerate(food_patches):
        ax.scatter(f[0], f[1], color="green", marker="o", s=200, zorder=5)
        ax.text(f[0] + 0.3, f[1] + 0.3, f"Patch {idx}", fontsize=10, fontweight="bold")

    # Add custom legend entry for Mycelial Network
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color="red", linestyle="--", linewidth=2.5),
        Line2D([0], [0], color="brown", linewidth=3.5)
    ]
    ax.legend(custom_lines, ["Greedy TSP Routing Path", "Mycelial Network Conduits (Resilient & Parallel)"], loc="upper left")
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 21)
    ax.grid(True, linestyle=":")
    plt.show()

def run_bio_bacteria_plot():
    # Setup
    bounds = (np.array([0.0, 0.0]), np.array([20.0, 20.0]))
    initial_pos = np.array([1.0, 1.0])
    dt = 0.1
    steps = 250
    attractant_peak = np.array([10.0, 10.0])
    
    def get_attractant_concentration(pos_val):
        d2 = np.sum((pos_val - attractant_peak) ** 2)
        return 120.0 * np.exp(-d2 / 120.0)

    pos = initial_pos
    theta = np.random.uniform(0, 2 * np.pi)
    tau_adapt = 10.0
    methylation = get_attractant_concentration(pos)
    run_speed = 2.5
    tumble_cooldown = 0
    
    traj_x, traj_y = [], []

    for _ in range(steps):
        concentration = get_attractant_concentration(pos)
        dm = (concentration - methylation) / tau_adapt
        methylation += dm * dt
        receptor_activity = concentration - methylation
        
        if tumble_cooldown > 0:
            tumble_cooldown -= 1
        else:
            p_tumble = 0.02 if receptor_activity > 0.01 else 0.35
            if np.random.rand() < p_tumble:
                theta = np.random.uniform(0, 2 * np.pi)
                tumble_cooldown = 2
            else:
                pos += np.array([np.cos(theta), np.sin(theta)]) * run_speed * dt
                
        pos = np.clip(pos, bounds[0], bounds[1])
        traj_x.append(pos[0])
        traj_y.append(pos[1])

    # Plot over Gradient Heatmap
    fig, ax = plt.subplots(figsize=(9, 8))
    X_grid, Y_grid = np.meshgrid(np.linspace(0, 20, 100), np.linspace(0, 20, 100))
    D2 = (X_grid - 10.0)**2 + (Y_grid - 10.0)**2
    Z_grid = 120.0 * np.exp(-D2 / 120.0)
    
    contour = ax.contourf(X_grid, Y_grid, Z_grid, levels=15, cmap="YlOrRd", alpha=0.7)
    fig.colorbar(contour, label="Attractant Chemical Concentration")
    
    ax.plot(traj_x, traj_y, color="blue", linewidth=2.0, marker=".", label="Bacteria Run-and-Tumble Trajectory")
    ax.scatter(initial_pos[0], initial_pos[1], color="darkblue", marker="X", s=200, label="Starting Point")
    ax.scatter(attractant_peak[0], attractant_peak[1], color="darkred", marker="*", s=300, label="Gradient Peak")
    
    ax.set_title("Objective 4: E. coli Gradient Climbing and Sensory Adaptation")
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.5, 20.5)
    ax.legend(loc="upper left")
    plt.show()

# =====================================================================
# VISUALIZATION WRAPPER: NEURO-SYMBOLIC WAREHOUSE ROBOT
# =====================================================================

def run_and_visualize_symbolic():
    print("\nExecuting Neuro-Symbolic Agent Simulation...")
    bounds = (np.array([0.0, 0.0]), np.array([20.0, 20.0]))
    initial_pos = np.array([5.0, 1.0])
    dt = 0.1
    steps = 500
    
    planner = sym.SymbolicCognitivePlanner()
    
    # Build System 1
    pickup_a_drive = Drive("pickup_a", SpatialTarget("pickup_a", sym.STATIONS["pickup_a"], gain=2.5))
    pickup_b_drive = Drive("pickup_b", SpatialTarget("pickup_b", sym.STATIONS["pickup_b"], gain=2.5))
    assembly_drive = Drive("assembly", SpatialTarget("assembly", sym.STATIONS["assembly"], gain=2.5))
    delivery_drive = Drive("delivery", SpatialTarget("delivery", sym.STATIONS["delivery"], gain=2.5))
    recharge_drive = Drive("recharge", SpatialTarget("charger", sym.STATIONS["charger"], gain=2.5))
    idle_drive     = Drive("idle",     SpatialTarget("charger_rest", sym.STATIONS["charger"], gain=1.5))

    obstacle_target = SpatialTarget("obstacle", sym.STATIONS["obstacle"], behavior="REPULSIVE", influence_radius=1.8, gain=3.5)
    obstacle_drive  = Drive("avoid_obstacle", obstacle_target, gating_type="REACTIVE", default_priority=4.5)

    mission = StateMachine(start_state="STATE_IDLE")
    mission.add_state("STATE_IDLE", State(drive=idle_drive))
    mission.add_state("STATE_PICK_A", State(drive=pickup_a_drive))
    mission.add_state("STATE_PICK_B", State(drive=pickup_b_drive))
    mission.add_state("STATE_GO_ASSEMBLY", State(drive=assembly_drive))
    mission.add_state("STATE_GO_DELIVERY", State(drive=delivery_drive))
    mission.add_state("STATE_RECHARGE", State(drive=recharge_drive))

    intention_state_map = {
        "PICK_A": "STATE_PICK_A", "PICK_B": "STATE_PICK_B",
        "GO_ASSEMBLY": "STATE_GO_ASSEMBLY", "GO_DELIVERY": "STATE_GO_DELIVERY",
        "RECHARGE": "STATE_RECHARGE", "IDLE": "STATE_IDLE"
    }

    initial_states = {"atp": 10.0, "structure": 90.0, "precursor": 100.0, "pos": initial_pos}

    # FIX: Changed 'dt_val' to 'dt' to match the keyword argument passed by the compiler
    def robot_metabolism(engine, kinetic_cost, effort_cost, dt):
        atp, struct, prec = sym.evaluate_tripartite_metabolism(
            engine.internal_states["atp"], engine.internal_states["structure"], engine.internal_states["precursor"],
            kinetic_cost, effort_cost,
            is_charging=(np.linalg.norm(engine.internal_states["pos"] - sym.STATIONS["charger"]) < 1.0),
            dt=dt  # Pass matching 'dt' variable here
        )
        engine.internal_states["atp"] = atp
        engine.internal_states["structure"] = struct
        engine.internal_states["precursor"] = prec

    engine = CDFCompilerEngine(mission=mission, initial_internal_states=initial_states, metabolism_fn=robot_metabolism, bounds=bounds)
    engine.register_background_drive(obstacle_drive)
    engine.compile_environment()

    # Track coordinate history segments paired with high-level active intentions
    traj_segments = []
    battery_history = []
    pos = initial_pos

    for step in range(steps):
        battery = engine.internal_states["precursor"]
        engine.internal_states["pos"] = pos
        
        # System 2 Reasoning
        planner.execute_reasoning_cycle(pos, battery)
        active_intention = planner.beliefs["active_intention"]
        engine.mission.current_state_name = intention_state_map[active_intention]

        # Physics step
        next_pos, _ = engine.step(pos, dt)
        
        # Record trajectory segment mapped to active intention
        traj_segments.append((np.copy(pos), np.copy(next_pos), active_intention))
        battery_history.append(battery)
        pos = next_pos

    # Matplotlib Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Hybrid Neuro-Symbolic Agent Warehouse Floor Simulation", fontsize=14, fontweight='bold')

    # Color mapping for symbolic intentions
    color_map = {
        "PICK_A": "cyan",
        "PICK_B": "blue",
        "GO_ASSEMBLY": "magenta",
        "GO_DELIVERY": "green",
        "RECHARGE": "orange",
        "IDLE": "gray"
    }

    ax1.set_title("Multicolored Symbolic Intention Path Map")
    ax1.set_xlim(-1, 21)
    ax1.set_ylim(-1, 21)
    ax1.grid(True, linestyle=":")

    # Plot Stations
    for name, coords in sym.STATIONS.items():
        color = "red" if name == "obstacle" else "black"
        marker = "X" if name == "obstacle" else "s"
        ax1.scatter(coords[0], coords[1], marker=marker, color=color, s=150, zorder=5)
        ax1.text(coords[0] + 0.3, coords[1] + 0.3, name.upper(), fontsize=9, fontweight="bold")

    # Plot segments with colored path indicating changing intention
    for p_start, p_end, intention in traj_segments:
        ax1.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color=color_map[intention], linewidth=2.5)

    # Manual legend mapping for colors
    legend_elements = [plt.Line2D([0], [0], color=c, lw=3, label=f"Intention: {k}") for k, c in color_map.items()]
    ax1.legend(handles=legend_elements, loc="upper right")

    # Plot Battery timeline on ax2
    ax2.set_title("Warehouse Battery Level Timeline")
    ax2.plot(np.arange(steps) * dt, battery_history, color="purple", linewidth=2.5, label="Battery Storage")
    ax2.axhline(35.0, color="red", linestyle=":", alpha=0.8, label="Low Battery Trigger")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Battery %")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle=":")
    ax2.legend()

    plt.tight_layout()
    plt.show()

# =====================================================================
# INTERACTIVE TERMINAL MENU
# =====================================================================

def interactive_menu():
    while True:
        print("\n" + "="*70)
        print("          CDF v2-UL AGENT COMPILER: VISUALIZATION DASHBOARD")
        print("="*70)
        print("Please choose a simulation to run and visualize:")
        print("  [1] Autonomous Mobile Robot (AMR) Factory Floor")
        print("  [2] Biological Paradigms (Objective 1-4 Multi-selection)")
        print("  [3] Hybrid Neuro-Symbolic Warehouse Robot")
        print("  [0] Exit Visualizer")
        print("-"*70)
        
        choice = input("Enter choice: ").strip()
        
        if choice == "1":
            run_and_visualize_amr()
        elif choice == "2":
            print("\nSelect Biological Paradigm Paradigm:")
            print("  [1] Forager Agent with Threat Evasion (Mouse & Weasel)")
            print("  [2] Growing Plant under Drought & Rain Cycle (Arabidopsis)")
            print("  [3] Decentralized Fungi Growth with Chemotropism vs Greedy TSP")
            print("  [4] Chemotactic Bacteria Gradient Climbing (E. coli)")
            print("  [0] Return to Main Menu")
            print("-"*70)
            bio_choice = input("Enter paradigm choice: ").strip()
            
            if bio_choice == "1":
                run_forager_simulation = run_bio_forager_plot()
            elif bio_choice == "2":
                run_bio_plant_plot()
            elif bio_choice == "3":
                run_bio_fungi_plot()
            elif bio_choice == "4":
                run_bio_bacteria_plot()
        elif choice == "3":
            run_and_visualize_symbolic()
        elif choice == "0":
            print("Exiting Dashboard. Thank you.")
            break
        else:
            print("Invalid selection. Please try again.")

if __name__ == "__main__":
    interactive_menu()