"""
prototype/bio_simulations.py — Biological simulations using the CDF v2-UL mathematical primitives.
"""

import sys
import os
import numpy as np

# Resolve imports from the compiler directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from compiler.integrator import lambert_w0, exact_linear_step
from compiler.dsl import SpatialTarget, Drive, FeedbackLoop, LambdaInstruction, State, StateMachine
from compiler.core import CDFCompilerEngine

# =====================================================================
# METABOLIC MATHEMATICS (Somatic Engine Primitives)
# =====================================================================

def evaluate_tripartite_metabolism(atp, structure, precursor, kinetic_cost, effort_cost, is_charging, dt):
    """
    Mathematical update of the physiological state vector x = [atp, structure, precursor]^T
    using CDF v2-UL exact G-space transformations and Lambert-W inversions.
    """
    atp_max = 10.0
    structure_max = 100.0
    precursor_max = 100.0
    k_m = 50.0  # Saturation constant

    # 1. Compute Autopoietic Metabolic Rates
    # Catabolism increases when precursors are low to protect ATP
    chi = 1.0 / (1.0 + np.exp(0.5 * (precursor - 15.0)))
    r_cat = 0.25 * structure * chi
    r_ana = 0.15 * atp * (precursor / (precursor + 5.0)) * (1.0 - structure / structure_max)

    # --- Precursor Pool: Saturating-Exponential Projection ---
    g_c = precursor * np.exp(precursor / k_m)
    
    if is_charging:
        forcing_physical = 0.8 * precursor_max * 0.3
    else:
        # FIX: Emulated Stomatal Closure
        # As internal water (precursor) depletes, transpiration is restricted down to 15% of baseline
        stomatal_resistance = np.clip(precursor / 40.0, 0.15, 1.0)
        forcing_physical = -(0.06 + 1.2 * kinetic_cost) * stomatal_resistance
        
    forcing_physical += 0.1 * r_cat - 0.2 * r_ana
    
    # Map physical forcing to G-space forcing
    dg_c = forcing_physical * (1.0 + precursor / k_m) * np.exp(precursor / k_m)
    g_c_next = exact_linear_step(g_c, rate=-0.05, forcing=dg_c, dt=dt)
    precursor_next = k_m * lambert_w0(max(g_c_next, 0.0) / k_m)

    # --- ATP Pool: Ratio-Bounded Coordinate Mapping ---
    g_e = atp / (atp_max - atp)
    c_active = 0.05 * kinetic_cost + 0.02 * effort_cost + 0.3 * r_ana - 0.6 * r_cat
    c_g = ((1.0 + g_e) ** 2 / atp_max) * c_active
    
    # Charging potential based on internal water/food availability
    lam = 0.4 * np.tanh(precursor / 25.0)
    g_e_next = exact_linear_step(g_e, rate=lam, forcing=-c_g, dt=dt)
    
    # Inverse ratio mapping
    g_e_next = max(g_e_next, 1e-9)
    atp_next = atp_max * g_e_next / (1.0 + g_e_next)

    # --- Structure Pool: Exact Bounded Relaxation ---
    a_struct = 0.15 * atp * (precursor / (precursor + 5.0))
    b_struct = a_struct / structure_max + 0.25 * chi + 0.02
    structure_eq = a_struct / b_struct
    structure_next = structure_eq + (structure - structure_eq) * np.exp(-b_struct * dt)

    # Defensive safety backstop clamps
    atp_next = float(np.clip(atp_next, 1e-3, atp_max * (1.0 - 1e-9)))
    structure_next = float(np.clip(structure_next, 1e-3, structure_max * (1.0 - 1e-9)))
    precursor_next = float(np.clip(precursor_next, 0.0, precursor_max))

    return atp_next, structure_next, precursor_next


# =====================================================================
# SIMULATION 1: FORAGER AGENT WITH THREAT EVASION
# =====================================================================

def run_forager_simulation():
    print("\n--- Running Objective 1: Forager Agent (Rodent Field Model) ---")
    
    # Environment Setup: 10m x 10m Field (20x20 units)
    bounds = (np.array([0.0, 0.0]), np.array([20.0, 20.0]))
    initial_pos = np.array([1.0, 1.0])
    dt = 0.1
    
    # Targets
    food_target = SpatialTarget("food", coords=[18.0, 18.0], behavior="ATTRACTIVE", influence_radius=6.0, gain=2.5)
    threat_target = SpatialTarget("predator", coords=[10.0, 10.0], behavior="REPULSIVE", influence_radius=4.0, gain=4.0)
    
    # Drives
    forage_drive = Drive("forage", target=food_target, gating_type="COGNITIVE", default_priority=4.5)
    evasion_drive = Drive("evade", target=threat_target, gating_type="REACTIVE", default_priority=5.0)

    mission = StateMachine(start_state="FORAGING")
    mission.add_state("FORAGING", State(drive=forage_drive))

    # Initialize Engine with Generic States
    initial_states = {
        "atp": 9.5,
        "structure": 80.0,
        "precursor": 30.0,
        "pos": initial_pos
    }

    def forager_metabolism(engine, kinetic_cost, effort_cost, dt):
        atp, struct, prec = evaluate_tripartite_metabolism(
            engine.internal_states["atp"],
            engine.internal_states["structure"],
            engine.internal_states["precursor"],
            kinetic_cost, effort_cost,
            is_charging=(np.linalg.norm(engine.internal_states["pos"] - food_target.coords) < 1.0),
            dt=dt
        )
        engine.internal_states["atp"] = atp
        engine.internal_states["structure"] = struct
        engine.internal_states["precursor"] = prec

    engine = CDFCompilerEngine(
        mission=mission,
        initial_internal_states=initial_states,
        metabolism_fn=forager_metabolism,
        bounds=bounds
    )
    engine.register_background_drive(evasion_drive)
    engine.compile_environment()

    pos = initial_pos
    predator_pos = np.array([10.0, 10.0])
    
    for step in range(120):
        engine.internal_states["pos"] = pos
        
        # Predator Tracking Logic: pursues only if the agent is within detection radius (2.5m / 5.0 units)
        dist_to_agent = np.linalg.norm(pos - predator_pos)
        if dist_to_agent < 5.0:
            predator_dir = (pos - predator_pos) / (dist_to_agent + 1e-3)
            predator_pos += predator_dir * 1.5 * dt  # Run-speed pursuit
        else:
            # Harmless localized brownian drift
            predator_pos += np.random.uniform(-0.5, 0.5, size=2) * dt
            predator_pos = np.clip(predator_pos, 2.0, 18.0)
            
        threat_target.coords = predator_pos  # Update physical position of the repelling source
        
        pos, psi = engine.step(pos, dt)
        
        if step % 30 == 0:
            print(f"  Step {step:03d} | Pos: {pos.round(1)} | Predator: {predator_pos.round(1)} | "
                  f"Physiology: [ATP: {engine.internal_states['atp']:.2f}, "
                  f"Biomass: {engine.internal_states['structure']:.2f}, "
                  f"Water/Precursor: {engine.internal_states['precursor']:.2f}]")
            
    print(f"  Result: Final Distance to Food: {np.linalg.norm(pos - food_target.coords):.2f} units.")


# =====================================================================
# SIMULATION 2: GROWING PLANT WITH HYDRATION DYNAMICS
# =====================================================================

def run_plant_simulation():
    print("\n--- Running Objective 2: Growing Plant (Arabidopsis Hydration Model) ---")
    dt = 0.5  # Adjusted step size for smooth non-lethal integration
    
    # Internal physical states
    atp = 9.0        # Turgor equivalent
    structure = 20.0  # Dry biomass
    precursor = 90.0  # Water wet mass (90% initial water fraction)
    
    min_biomass = structure
    
    # Simulate a drought phase followed by a rehydration phase
    for step in range(320):
        is_charging = (step > 160)  # Rainfall resumes after step 160
        
        # Evaluate using the G-space engine
        atp, structure, precursor = evaluate_tripartite_metabolism(
            atp, structure, precursor,
            kinetic_cost=0.0,  # Sessile agent, zero movement cost
            effort_cost=0.01,
            is_charging=is_charging,
            dt=dt
        )
        
        if step <= 160:
            min_biomass = min(min_biomass, structure)
            
        if step == 160:
            print(f"  [End of Drought] Physiology: ATP (Turgor): {atp:.2f} | Dry Biomass: {structure:.2f} | Wet Precursor: {precursor:.2f}")
            
    print(f"  [End of Rain Cycle] Physiology: ATP (Turgor): {atp:.2f} | Dry Biomass: {structure:.2f} | Wet Precursor: {precursor:.2f}")
    print(f"  Plant Report: Max Biomass Loss during drought: {((20.0 - min_biomass)/20.0)*100:.1f}% reduction. "
          f"Anabolic recovery after rain: {structure:.2f} dry mass.")


# =====================================================================
# SIMULATION 3: DECENTRALIZED FUNGI MYCELIAL NETWORK vs GREEDY TSP
# =====================================================================

class MycelialNode:
    def __init__(self, node_id, coords, parent_id=None):
        self.id = node_id
        self.coords = np.array(coords, dtype=float)
        self.parent_id = parent_id
        self.children = []
        
        # Autopoietic states
        self.atp = 9.0
        self.structure = 10.0
        self.precursor = 40.0
        self.is_active = True


def run_fungal_simulation():
    print("\n--- Running Objective 3: Decentralized Fungi vs Greedy TSP Tour ---")
    
    root_coords = np.array([10.0, 10.0])
    food_patches = [
        np.array([2.0, 15.0]),
        np.array([7.0, 18.0]),
        np.array([14.0, 17.0]),
        np.array([18.0, 11.0]),
        np.array([11.0, 2.0])
    ]
    
    # -----------------------------------------------------------------
    # Reference Benchmark: Greedy Traveling Salesperson (TSP) Solver
    # -----------------------------------------------------------------
    def solve_greedy_tsp(start, targets):
        current = np.array(start)
        unvisited = list(targets)
        total_dist = 0.0
        path_segments = []  # Store cumulative distances mapped to target indexes
        
        while unvisited:
            nearest_idx = min(range(len(unvisited)), key=lambda i: np.linalg.norm(current - unvisited[i]))
            nearest_node = unvisited.pop(nearest_idx)
            dist = np.linalg.norm(current - nearest_node)
            total_dist += dist
            current = nearest_node
            
            # Find matching index of this target in the original food_patches list
            original_idx = next(idx for idx, f in enumerate(food_patches) if np.allclose(f, current))
            path_segments.append((original_idx, total_dist))
            
        return total_dist, path_segments

    tsp_length, tsp_segments = solve_greedy_tsp(root_coords, food_patches)
    print(f"  Greedy TSP Route Total Cable Length: {tsp_length:.2f} units.")

    # -----------------------------------------------------------------
    # Decoupled Autopoietic Fungal Growth Simulation
    # -----------------------------------------------------------------
    nodes = {0: MycelialNode(0, root_coords)}
    next_node_id = 1
    
    # Spawn 6 initial symmetrical exploration branches (tips)
    active_tips = []
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    for theta in angles:
        tip_pos = root_coords + np.array([np.cos(theta), np.sin(theta)])
        node = MycelialNode(next_node_id, tip_pos, parent_id=0)
        nodes[0].children.append(next_node_id)
        nodes[next_node_id] = node
        active_tips.append(next_node_id)
        next_node_id += 1

    dt = 1.0
    gamma = 0.15  # Cytoplasmic flow transport rate (120 um/sec equivalent scaling)
    pruned_count = 0

    # Simulate network growth steps
    for step in range(40):
        # 1. Growth Step: Active tips grow toward the nearest unvisited food patch
        new_tips = []
        for tip_id in active_tips:
            tip_node = nodes[tip_id]
            if not tip_node.is_active:
                continue
                
            # Find closest food patch
            dists = [np.linalg.norm(tip_node.coords - f) for f in food_patches]
            if not dists:
                continue
            closest_idx = np.argmin(dists)
            target_food = food_patches[closest_idx]
            
            direction = target_food - tip_node.coords
            dist_to_food = np.linalg.norm(direction)
            
            # Draw nutrient cargo if touching a patch
            is_charging = dist_to_food < 1.2
            
            # Growth extension
            if dist_to_food > 0.5:
                step_dir = (direction / (dist_to_food + 1e-5)) * 0.8
                new_coords = tip_node.coords + step_dir
                
                # Instantiating a new branching segment
                new_node = MycelialNode(next_node_id, new_coords, parent_id=tip_id)
                nodes[tip_id].children.append(next_node_id)
                nodes[next_node_id] = new_node
                new_tips.append(next_node_id)
                next_node_id += 1
                
                # The old tip becomes an established conduit
                tip_node.is_active = False 
            else:
                # Arrived at food source, keep tip active to anchor connection
                new_tips.append(tip_id)

        if new_tips:
            active_tips = new_tips

        # 2. Cytoplasmic Flow: Translocate nutrients along parent-child linkages
        for _ in range(3):
            for n_id, node in list(nodes.items()):
                if n_id == 0 or not node.is_active:
                    continue
                parent = nodes[node.parent_id]
                flow = gamma * (node.precursor - parent.precursor) * (node.structure / 100.0)
                node.precursor -= flow * dt
                parent.precursor += flow * dt

        # 3. Autopoietic Node Updates & Pruning
        for n_id, node in list(nodes.items()):
            on_food = any(np.linalg.norm(node.coords - f) < 1.2 for f in food_patches)
            
            node.atp, node.structure, node.precursor = evaluate_tripartite_metabolism(
                node.atp, node.structure, node.precursor,
                kinetic_cost=0.01, effort_cost=0.01,
                is_charging=on_food, dt=dt
            )
            
            if node.structure < 1.5 and n_id != 0:
                node.is_active = False
                parent = nodes[node.parent_id]
                if n_id in parent.children:
                    parent.children.remove(n_id)
                pruned_count += 1

    # Evaluate Surviving Total Cable Length
    total_cable = 0.0
    active_node_count = 0
    for n_id, node in nodes.items():
        if node.structure >= 1.5:
            active_node_count += 1
            if node.parent_id is not None:
                total_cable += np.linalg.norm(node.coords - nodes[node.parent_id].coords)

    print(f"  Surviving Mycelial Cable Length: {total_cable:.2f} units.")
    print(f"  Surviving Active Nodes: {active_node_count} | Starvation Pruned Segments: {pruned_count}")
    
    # -----------------------------------------------------------------
    # FIX: Calculate Exact Path Distance to each Food Patch
    # -----------------------------------------------------------------
    mycelial_distances = {}
    for f_idx, f_coords in enumerate(food_patches):
        # Locate the closest active network node to this specific food patch
        best_node_id = None
        best_dist = float('inf')
        for n_id, node in nodes.items():
            if node.structure >= 1.5:
                d = np.linalg.norm(node.coords - f_coords)
                if d < best_dist:
                    best_dist = d
                    best_node_id = n_id
                    
        # Trace path along parent links back to the Root (node 0)
        path_len = 0.0
        curr_id = best_node_id
        while curr_id is not None and nodes[curr_id].parent_id is not None:
            p_id = nodes[curr_id].parent_id
            path_len += np.linalg.norm(nodes[curr_id].coords - nodes[p_id].coords)
            curr_id = p_id
        mycelial_distances[f_idx] = path_len

    # Format TSP distances to match indexes
    tsp_dist_map = {idx: d for idx, d in tsp_segments}

    # Print Detailed Quantitative Comparison
    print("\n  ==========================================================================")
    print("  QUANTITATIVE PATH DISTANCES TO RESOURCE PATCHES (From Root)")
    print("  ==========================================================================")
    print("  Patch ID | Coordinates | TSP Route Distance (m) | Mycelial Path Distance (m)")
    print("  ---------+-------------+------------------------+-------------------------")
    for f_idx, f_coords in enumerate(food_patches):
        tsp_dist = tsp_dist_map.get(f_idx, float('inf'))
        myc_dist = mycelial_distances.get(f_idx, float('inf'))
        print(f"  Patch {f_idx}  | [{f_coords[0]:4.1f},{f_coords[1]:4.1f}] |         {tsp_dist:5.2f}          |          {myc_dist:5.2f}")
    print("  ==========================================================================\n")


# =====================================================================
# SIMULATION 4: CHEMOTACTIC BACTERIA (CheR/CheB Temporal Adaptation)
# =====================================================================

def run_bacteria_simulation():
    print("--- Running Objective 4: Chemotactic Bacteria (E. coli CheR/CheB Model) ---")
    
    bounds = (np.array([0.0, 0.0]), np.array([20.0, 20.0]))
    initial_pos = np.array([1.0, 1.0])
    dt = 0.1  # 100ms per step
    
    # Target Peak representing a localized Gaussian nutrient gradient
    attractant_peak = np.array([10.0, 10.0])
    
    # FIX: Increased width of Gaussian profile (from 50.0 to 120.0) 
    # so the bacterium can perceive the gradient signal from the far corners.
    def get_attractant_concentration(pos):
        d2 = np.sum((pos - attractant_peak) ** 2)
        return 120.0 * np.exp(-d2 / 120.0)

    # State vectors
    pos = initial_pos
    theta = np.random.uniform(0, 2 * np.pi)
    
    # Adaptation biochemical memory states (CheR/CheB adaptation mechanism)
    tau_adapt = 10.0  # 10-second methylation adaptation time constant
    methylation = get_attractant_concentration(pos)  # Initial memory baseline M(t)
    
    run_speed = 2.5  # 25 um/s equivalent scaling
    tumble_cooldown = 0
    tumbles_count = 0
    runs_count = 0

    print("  Beginning run-and-tumble gradient climbing...")
    
    for step in range(250):
        concentration = get_attractant_concentration(pos)
        
        # Temporal derivative integration (Biochemical Memory)
        dm = (concentration - methylation) / tau_adapt
        methylation += dm * dt
        
        # Receptor activity: A(t) = S(t) - M(t)
        receptor_activity = concentration - methylation
        
        if tumble_cooldown > 0:
            tumble_cooldown -= 1
        else:
            # Modulation of tumble probability based on derivative detection
            if receptor_activity > 0.01:
                # Favorable heading: suppress tumbling (longer runs)
                p_tumble = 0.02
            else:
                # Neutral/unfavorable heading: return to baseline tumbling frequency
                p_tumble = 0.35

            if np.random.rand() < p_tumble:
                # Tumble event: pick a random direction
                theta = np.random.uniform(0, 2 * np.pi)
                tumble_cooldown = 2  # 0.2-second tumble duration
                tumbles_count += 1
            else:
                # Maintain run forward
                pos += np.array([np.cos(theta), np.sin(theta)]) * run_speed * dt
                runs_count += 1
                
        # Safe sandbox boundary clamp
        pos = np.clip(pos, bounds[0], bounds[1])
        
        if step % 50 == 0:
            print(f"  Step {step:03d} | Pos: {pos.round(1)} | Sensed Conc: {concentration:.1f} | "
                  f"Memory (M): {methylation:.1f} | Activity: {receptor_activity:+.2f}")
            
    print(f"  Bacteria Simulation Ended. Final Distance to peak: {np.linalg.norm(pos - attractant_peak):.2f} units.")
    print(f"  Run Decisions: {runs_count} | Tumbles: {tumbles_count} (Ratio: {runs_count/max(1, tumbles_count):.2f})")


# =====================================================================
# MAIN RUNNER
# =====================================================================

if __name__ == "__main__":
    print("======================================================================")
    print("      CDF v2-UL AGENT COMPILER: BIOLOGICAL PARADIGMS PROTOTYPE        ")
    print("======================================================================")
    
    run_forager_simulation()
    run_plant_simulation()
    run_fungal_simulation()
    run_bacteria_simulation()