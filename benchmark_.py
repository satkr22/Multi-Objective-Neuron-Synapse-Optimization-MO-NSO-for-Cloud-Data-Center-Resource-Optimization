# ===============================
# Multi-Algorithm Benchmarking Suite (MONSO VERSION)
# Comparing: MO-NSO, NSGA-II, NSGA-III, MO-PSO, ACO, Cat Swarm, FCFS
# Modified to use monso.py instead of dynamic.py
# ===============================

from __future__ import annotations
import math
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import sys
from pathlib import Path

# Import from monso.py (vectorized version)
from mm import (
    Host, VM, Container, VMType, HostArray, VMArray, ContainerArray,
    decode_solution_matrix_vectorized,
    build_state_vectorized, PlacementStateVectorized,
    compute_cpu_allocation_vectorized,
    compute_makespan_vectorized, compute_energy_vectorized, 
    compute_cost_vectorized, compute_migration_time_vectorized,
    MO_NSO_Optimizer_Vectorized,
    BANDWIDTH_GBPS, MB_TO_GB, IDLE_COST_FACTOR
)


# ===============================
# Helper function to convert list-based data to arrays
# ===============================
def convert_to_arrays(hosts: List[Host], vms: List[VM], containers: List[Container]):
    """Convert list-based hosts, vms, containers to vectorized arrays"""
    # Create HostArray
    host_array = HostArray(
        cpu_capacities=np.array([h.cpu_capacity for h in hosts], dtype=np.float64),
        ram_capacities=np.array([h.ram_capacity for h in hosts], dtype=np.float64),
        power_idles=np.array([h.power_idle for h in hosts], dtype=np.float64),
        power_maxs=np.array([h.power_max for h in hosts], dtype=np.float64)
    )
    
    # Create VMArray
    vm_array = VMArray(
        cpu_demands=np.array([vm.cpu_demand for vm in vms], dtype=np.float64),
        cpu_capacities=np.array([vm.type.mips_capacity for vm in vms], dtype=np.float64),
        ram_sizes=np.array([vm.type.ram_gb for vm in vms], dtype=np.float64),
        prices_per_sec=np.array([vm.type.price_per_hour / 3600.0 for vm in vms], dtype=np.float64),
        cpu_weights=np.array([vm.cpu_weight for vm in vms], dtype=np.float64),
        memory_gbs=np.array([vm.type.ram_gb for vm in vms], dtype=np.float64)  # For migration
    )
    
    # Create ContainerArray
    container_array = ContainerArray(
        workloads_mi=np.array([c.workload_mi for c in containers], dtype=np.float64),
        state_sizes_mb=np.array([c.state_size_mb for c in containers], dtype=np.float64),
        ram_requirements_mb=np.array([c.ram_requirement_mb for c in containers], dtype=np.float64)
    )
    
    return host_array, vm_array, container_array

# ===============================
# NSGA-II Implementation
# ===============================
class NSGA2:
    def __init__(self, decision_dim, hosts, vms, containers, pop_size=30, max_iters=200, seed=42):
        self.decision_dim = decision_dim
        self.hosts = hosts
        self.vms = vms
        self.containers = containers
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.num_hosts = len(hosts)
        self.num_vms = len(vms)
        self.num_containers = len(containers)
        
        # Convert to arrays for vectorized operations
        self.host_array, self.vm_array, self.container_array = convert_to_arrays(hosts, vms, containers)
        
        self.prev_states = [None] * self.pop_size
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.archive = []
        self.runtime = 0
        
    def initialize_population(self):
        """Initialize random population"""
        return [np.random.randn(self.decision_dim) for _ in range(self.pop_size)]
    
    def evaluate(self, solution, idx):
        """Evaluate objectives: makespan, cost, energy, migration"""
        
        np.random.seed(hash(solution.tobytes()) % (2**32))

        # Decode using monso's vectorized decoder
        vm_to_host, container_to_vm = decode_solution_matrix_vectorized(
            solution, self.host_array, self.vm_array, self.container_array
        )
        
        # Build state using monso's vectorized state builder
        state = build_state_vectorized(
            vm_to_host, container_to_vm, 
            self.num_hosts, self.num_vms, self.num_containers
        )
        
        # # Compute objectives using vectorized functions
        # allocated_cpu = compute_cpu_allocation_vectorized(state, self.host_array, self.vm_array)
        # makespan = compute_makespan_vectorized(state, allocated_cpu, self.container_array)
        
        # # Note: compute_energy_vectorized signature: (state, hosts, vms, allocations, makespan)
        # energy = compute_energy_vectorized(state, self.host_array, self.vm_array, allocated_cpu, makespan)
        
        # # Note: compute_cost_vectorized signature: (state, vms, makespan)
        # cost = compute_cost_vectorized(state, self.vm_array, makespan)
        
        
        # 2. Compute CPU allocation
        allocations, total_under_alloc_penalty = compute_cpu_allocation_vectorized(state, self.host_array, self.vm_array)
        
        # 3. Compute makespan (needed for cost and energy)
        makespan, vm_times = compute_makespan_vectorized(state, allocations, self.container_array)
        
        # 4. Compute other objectives (all vectorized)
        energy = compute_energy_vectorized(state, self.host_array, self.vm_array, allocations, makespan)
        
        cost = compute_cost_vectorized(state, self.vm_array, makespan, vm_times, total_under_alloc_penalty)
        

            
        # Compute migration cost
        if self.prev_states[idx] is None:
            migration = 0.0
        else:
            # compute_migration_time_vectorized signature: (current_state, prev_state, vms, containers, generation, max_generations, apply_scaling)
            migration = compute_migration_time_vectorized(
                state,
                self.prev_states[idx],
                self.vm_array,
                self.container_array,
                generation=0,  # Not tracking generations in benchmark algorithms
                max_generations=self.max_iters,
                apply_scaling=False  # No scaling for fair comparison
            )
        
        # Store current state for next iteration (store PlacementStateVectorized directly)
        self.prev_states[idx] = state
        
        # Normalize migration (same as monso)
        #
                
        return [makespan, cost, energy, migration]
    
    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2 (minimize all objectives)"""
        better_in_one = False
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2:
                return False
            if o1 < o2:
                better_in_one = True
        return better_in_one
    
    def fast_non_dominated_sort(self, population, objectives):
        """NSGA-II fast non-dominated sorting"""
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(i+1, n):
                if self.dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self.dominates(objectives[j], objectives[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
        
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)
        
        # Remove any empty fronts at the end
        fronts = [f for f in fronts if f]
        
        return fronts
    
    def crowding_distance(self, objectives_subset):
        """Calculate crowding distance"""
        n = len(objectives_subset)
        if n <= 2:
            return [float('inf')] * n
        
        distances = [0.0] * n
        num_objectives = len(objectives_subset[0])
        
        for m in range(num_objectives):
            sorted_indices = sorted(range(n), key=lambda i: objectives_subset[i][m])
            
            obj_min = objectives_subset[sorted_indices[0]][m]
            obj_max = objectives_subset[sorted_indices[-1]][m]
            
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            if obj_max - obj_min > 0:
                for i in range(1, n-1):
                    distances[sorted_indices[i]] += (
                        objectives_subset[sorted_indices[i+1]][m] - 
                        objectives_subset[sorted_indices[i-1]][m]
                    ) / (obj_max - obj_min)
        
        return distances
    
    def tournament_selection(self, population, ranks, distances):
        """Binary tournament selection"""
        i, j = random.sample(range(len(population)), 2)
        if ranks[i] < ranks[j]:
            return population[i].copy()
        elif ranks[i] > ranks[j]:
            return population[j].copy()
        else:
            return population[i].copy() if distances[i] > distances[j] else population[j].copy()
    
    def simulated_binary_crossover(self, parent1, parent2, eta=20):
        """SBX crossover"""
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(len(parent1)):
            if random.random() < 0.9:  # crossover probability
                u = random.random()
                if u <= 0.5:
                    beta = (2*u)**(1/(eta+1))
                else:
                    beta = (1/(2*(1-u)))**(1/(eta+1))
                
                child1[i] = 0.5*((1+beta)*parent1[i] + (1-beta)*parent2[i])
                child2[i] = 0.5*((1-beta)*parent1[i] + (1+beta)*parent2[i])
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        
        return child1, child2
    
    def polynomial_mutation(self, individual, eta=20, mutation_rate=0.1):
        """Polynomial mutation"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                u = random.random()
                if u < 0.5:
                    delta = (2*u)**(1/(eta+1)) - 1
                else:
                    delta = 1 - (2*(1-u))**(1/(eta+1))
                mutated[i] += delta
        return mutated
    
    def run(self):
        """Run NSGA-II optimization"""
        start_time = time.time()
        
        # Initialize
        population = self.initialize_population()
        
        for gen in range(self.max_iters):
            # Evaluate
            objectives = [self.evaluate(ind, i) for i, ind in enumerate(population)]
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(population, objectives)
            
            # Assign ranks
            ranks = [0] * len(population)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Calculate crowding distances
            distances = [0.0] * len(population)
            for front in fronts:
                front_objs = [objectives[i] for i in front]
                front_distances = self.crowding_distance(front_objs)
                for i, idx in enumerate(front):
                    distances[idx] = front_distances[i]
            
            # Generate offspring
            offspring = []
            for _ in range(self.pop_size // 2):
                parent1 = self.tournament_selection(population, ranks, distances)
                parent2 = self.tournament_selection(population, ranks, distances)
                
                child1, child2 = self.simulated_binary_crossover(parent1, parent2)
                child1 = self.polynomial_mutation(child1)
                child2 = self.polynomial_mutation(child2)
                
                offspring.extend([child1, child2])
            
            # Combine and select
            combined_pop = population + offspring
            combined_obj = objectives + [self.evaluate(ind, i % self.pop_size) 
                                        for i, ind in enumerate(offspring)]
            
            # Select next generation
            fronts = self.fast_non_dominated_sort(combined_pop, combined_obj)
            
            new_population = []
            new_objectives = []
            
            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    for idx in front:
                        new_population.append(combined_pop[idx])
                        new_objectives.append(combined_obj[idx])
                else:
                    # Need crowding distance for remaining slots
                    remaining = self.pop_size - len(new_population)
                    front_objs = [combined_obj[i] for i in front]
                    front_distances = self.crowding_distance(front_objs)
                    
                    sorted_front = sorted(zip(front, front_distances), 
                                        key=lambda x: x[1], reverse=True)
                    
                    for idx, _ in sorted_front[:remaining]:
                        new_population.append(combined_pop[idx])
                        new_objectives.append(combined_obj[idx])
                    break
            
            population = new_population
            objectives = new_objectives
            
            # Update archive
            for i, obj in enumerate(objectives):
                dominated = False
                to_remove = []
                
                for j, arch_entry in enumerate(self.archive):
                    if self.dominates(arch_entry['objectives'], obj):
                        dominated = True
                        break
                    elif self.dominates(obj, arch_entry['objectives']):
                        to_remove.append(j)
                
                if not dominated:
                    for j in reversed(to_remove):
                        del self.archive[j]
                    self.archive.append({
                        'solution': population[i].copy(),
                        'objectives': obj
                    })
            
            if gen % 50 == 0:
                print(f"NSGA-II Gen {gen}/{self.max_iters}, Archive size: {len(self.archive)}")
        
        self.runtime = time.time() - start_time
        return self.archive


# ===============================
# NSGA-III Implementation
# ===============================
class NSGA3:
    def __init__(self, decision_dim, hosts, vms, containers, pop_size=30, max_iters=200, seed=42):
        self.decision_dim = decision_dim
        self.hosts = hosts
        self.vms = vms
        self.containers = containers
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.num_hosts = len(hosts)
        self.num_vms = len(vms)
        self.num_containers = len(containers)
        
        # Convert to arrays for vectorized operations
        self.host_array, self.vm_array, self.container_array = convert_to_arrays(hosts, vms, containers)
        
        self.prev_states = [None] * self.pop_size
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Generate reference points for 4 objectives
        self.reference_points = self.generate_reference_points(4, 12)
        
        self.archive = []
        self.runtime = 0
    
    def generate_reference_points(self, num_objectives, num_divisions):
        """Generate uniformly distributed reference points (Das and Dennis method)"""
        points = []
        
        def generate_recursive(current_point, remaining_sum, remaining_objs):
            if remaining_objs == 1:
                points.append(current_point + [remaining_sum])
            else:
                for i in range(remaining_sum + 1):
                    generate_recursive(current_point + [i], remaining_sum - i, remaining_objs - 1)
        
        generate_recursive([], num_divisions, num_objectives)
        
        # Normalize
        reference_points = np.array(points, dtype=np.float64) / num_divisions
        return reference_points
    
    def initialize_population(self):
        """Initialize random population"""
        return [np.random.randn(self.decision_dim) for _ in range(self.pop_size)]
    
    def evaluate(self, solution, idx):
        """Evaluate objectives"""
        np.random.seed(hash(solution.tobytes()) % (2**32))
        vm_to_host, container_to_vm = decode_solution_matrix_vectorized(
            solution, self.host_array, self.vm_array, self.container_array
        )
        
        state = build_state_vectorized(
            vm_to_host, container_to_vm, 
            self.num_hosts, self.num_vms, self.num_containers
        )
        
        # allocated_cpu = compute_cpu_allocation_vectorized(state, self.host_array, self.vm_array)
        # makespan = compute_makespan_vectorized(state, allocated_cpu, self.container_array)
        # energy = compute_energy_vectorized(state, self.host_array, self.vm_array, allocated_cpu, makespan)
        # cost = compute_cost_vectorized(state, self.vm_array, makespan)
        
        # 2. Compute CPU allocation
        allocations, total_under_alloc_penalty = compute_cpu_allocation_vectorized(state, self.host_array, self.vm_array)
        
        # 3. Compute makespan (needed for cost and energy)
        makespan, vm_times = compute_makespan_vectorized(state, allocations, self.container_array)
        
        # 4. Compute other objectives (all vectorized)
        energy = compute_energy_vectorized(state, self.host_array, self.vm_array, allocations, makespan)
        
        cost = compute_cost_vectorized(state, self.vm_array, makespan, vm_times, total_under_alloc_penalty)
        
        if self.prev_states[idx] is None:
            migration = 0.0
        else:
            migration = compute_migration_time_vectorized(
                state, self.prev_states[idx], self.vm_array, self.container_array,
                generation=0, max_generations=self.max_iters, apply_scaling=False
            )
        
        self.prev_states[idx] = state
        #
        
        return [makespan, cost, energy, migration]
    
    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2"""
        better_in_one = False
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2:
                return False
            if o1 < o2:
                better_in_one = True
        return better_in_one
    
    def fast_non_dominated_sort(self, population, objectives):
        """NSGA-II style fast non-dominated sorting"""
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(i+1, n):
                if self.dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self.dominates(objectives[j], objectives[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
        
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)
        
        fronts = [f for f in fronts if f]
        return fronts
    
    def associate_to_reference_points(self, objectives_subset):
        """Associate solutions to reference points"""
        if not objectives_subset:
            return [], []
        
        # Normalize objectives
        obj_array = np.array(objectives_subset)
        ideal = np.min(obj_array, axis=0)
        nadir = np.max(obj_array, axis=0)
        
        # Avoid division by zero
        range_val = nadir - ideal
        range_val[range_val == 0] = 1.0
        
        normalized = (obj_array - ideal) / range_val
        
        # Calculate distances to reference points
        distances_to_refs = []
        associations = []
        
        for norm_obj in normalized:
            dists = np.linalg.norm(self.reference_points - norm_obj, axis=1)
            min_idx = np.argmin(dists)
            associations.append(min_idx)
            distances_to_refs.append(dists[min_idx])
        
        return associations, distances_to_refs
    
    def niching(self, last_front_indices, last_front_objs, remaining_slots):
        """Niching procedure for selecting from last front"""
        if remaining_slots <= 0 or not last_front_indices:
            return []
        
        associations, distances = self.associate_to_reference_points(last_front_objs)
        
        # Count solutions associated with each reference point
        ref_counts = {}
        for ref_idx in associations:
            ref_counts[ref_idx] = ref_counts.get(ref_idx, 0) + 1
        
        selected = []
        available = list(range(len(last_front_indices)))
        
        for _ in range(remaining_slots):
            if not available:
                break
            
            # Find reference point with minimum count
            min_count = min(ref_counts.get(associations[i], 0) for i in available)
            candidates = [i for i in available 
                         if ref_counts.get(associations[i], 0) == min_count]
            
            # Select solution with minimum distance to reference point
            selected_idx = min(candidates, key=lambda i: distances[i])
            selected.append(last_front_indices[selected_idx])
            
            # Update counts
            ref_idx = associations[selected_idx]
            ref_counts[ref_idx] = ref_counts.get(ref_idx, 0) + 1
            
            available.remove(selected_idx)
        
        return selected
    
    def tournament_selection(self, population):
        """Binary tournament selection"""
        i, j = random.sample(range(len(population)), 2)
        # Random selection for simplicity
        return population[i].copy() if random.random() < 0.5 else population[j].copy()
    
    def simulated_binary_crossover(self, parent1, parent2, eta=20):
        """SBX crossover"""
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(len(parent1)):
            if random.random() < 0.9:
                u = random.random()
                if u <= 0.5:
                    beta = (2*u)**(1/(eta+1))
                else:
                    beta = (1/(2*(1-u)))**(1/(eta+1))
                
                child1[i] = 0.5*((1+beta)*parent1[i] + (1-beta)*parent2[i])
                child2[i] = 0.5*((1-beta)*parent1[i] + (1+beta)*parent2[i])
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        
        return child1, child2
    
    def polynomial_mutation(self, individual, eta=20, mutation_rate=0.1):
        """Polynomial mutation"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                u = random.random()
                if u < 0.5:
                    delta = (2*u)**(1/(eta+1)) - 1
                else:
                    delta = 1 - (2*(1-u))**(1/(eta+1))
                mutated[i] += delta
        return mutated
    
    def run(self):
        """Run NSGA-III optimization"""
        start_time = time.time()
        
        population = self.initialize_population()
        
        for gen in range(self.max_iters):
            # Evaluate
            objectives = [self.evaluate(ind, i) for i, ind in enumerate(population)]
            
            # Generate offspring
            offspring = []
            for _ in range(self.pop_size // 2):
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                child1, child2 = self.simulated_binary_crossover(parent1, parent2)
                child1 = self.polynomial_mutation(child1)
                child2 = self.polynomial_mutation(child2)
                
                offspring.extend([child1, child2])
            
            # Combine and select
            combined_pop = population + offspring
            combined_obj = objectives + [self.evaluate(ind, i % self.pop_size) 
                                        for i, ind in enumerate(offspring)]
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(combined_pop, combined_obj)
            
            new_population = []
            new_objectives = []
            
            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    for idx in front:
                        new_population.append(combined_pop[idx])
                        new_objectives.append(combined_obj[idx])
                else:
                    # Use niching for last front
                    remaining = self.pop_size - len(new_population)
                    last_front_objs = [combined_obj[i] for i in front]
                    selected_indices = self.niching(front, last_front_objs, remaining)
                    
                    for idx in selected_indices:
                        new_population.append(combined_pop[idx])
                        new_objectives.append(combined_obj[idx])
                    break
            
            population = new_population
            objectives = new_objectives
            
            # Update archive
            for i, obj in enumerate(objectives):
                dominated = False
                to_remove = []
                
                for j, arch_entry in enumerate(self.archive):
                    if self.dominates(arch_entry['objectives'], obj):
                        dominated = True
                        break
                    elif self.dominates(obj, arch_entry['objectives']):
                        to_remove.append(j)
                
                if not dominated:
                    for j in reversed(to_remove):
                        del self.archive[j]
                    self.archive.append({
                        'solution': population[i].copy(),
                        'objectives': obj
                    })
            
            if gen % 50 == 0:
                print(f"NSGA-III Gen {gen}/{self.max_iters}, Archive size: {len(self.archive)}")
        
        self.runtime = time.time() - start_time
        return self.archive


# ===============================
# MO-PSO Implementation
# ===============================
class MOPSO:
    def __init__(self, decision_dim, hosts, vms, containers, pop_size=30, max_iters=200, seed=42):
        self.decision_dim = decision_dim
        self.hosts = hosts
        self.vms = vms
        self.containers = containers
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.num_hosts = len(hosts)
        self.num_vms = len(vms)
        self.num_containers = len(containers)
        
        # Convert to arrays
        self.host_array, self.vm_array, self.container_array = convert_to_arrays(hosts, vms, containers)
        
        self.prev_states = [None] * self.pop_size
        
        # PSO parameters
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.archive = []
        self.runtime = 0
    
    def initialize_population(self):
        """Initialize random population and velocities"""
        particles = [np.random.randn(self.decision_dim) for _ in range(self.pop_size)]
        velocities = [np.random.randn(self.decision_dim) * 0.1 for _ in range(self.pop_size)]
        return particles, velocities
    
    def evaluate(self, solution, idx):
        """Evaluate objectives"""
        np.random.seed(hash(solution.tobytes()) % (2**32))
        vm_to_host, container_to_vm = decode_solution_matrix_vectorized(
            solution, self.host_array, self.vm_array, self.container_array
        )
        
        state = build_state_vectorized(
            vm_to_host, container_to_vm, 
            self.num_hosts, self.num_vms, self.num_containers
        )
        
        # allocated_cpu = compute_cpu_allocation_vectorized(state, self.host_array, self.vm_array)
        # makespan = compute_makespan_vectorized(state, allocated_cpu, self.container_array)
        # energy = compute_energy_vectorized(state, self.host_array, self.vm_array, allocated_cpu, makespan)
        # cost = compute_cost_vectorized(state, self.vm_array, makespan)
        
        # 2. Compute CPU allocation
        allocations, total_under_alloc_penalty = compute_cpu_allocation_vectorized(state, self.host_array, self.vm_array)
        
        # 3. Compute makespan (needed for cost and energy)
        makespan, vm_times = compute_makespan_vectorized(state, allocations, self.container_array)
        
        # 4. Compute other objectives (all vectorized)
        energy = compute_energy_vectorized(state, self.host_array, self.vm_array, allocations, makespan)
        
        cost = compute_cost_vectorized(state, self.vm_array, makespan, vm_times, total_under_alloc_penalty)
        
        if self.prev_states[idx] is None:
            migration = 0.0
        else:
            migration = compute_migration_time_vectorized(
                state, self.prev_states[idx], self.vm_array, self.container_array,
                generation=0, max_generations=self.max_iters, apply_scaling=False
            )
        
        self.prev_states[idx] = state
        #
        
        return [makespan, cost, energy, migration]
    
    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2"""
        better_in_one = False
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2:
                return False
            if o1 < o2:
                better_in_one = True
        return better_in_one
    
    def run(self):
        """Run MO-PSO optimization"""
        start_time = time.time()
        
        # Initialize particles and velocities
        particles, velocities = self.initialize_population()
        
        # Initialize personal best
        personal_best = [p.copy() for p in particles]
        personal_best_obj = [self.evaluate(p, i) for i, p in enumerate(particles)]
        
        for iteration in range(self.max_iters):
            for i in range(self.pop_size):
                # Select global best from archive (random selection)
                if self.archive:
                    global_best = random.choice(self.archive)['solution']
                else:
                    global_best = personal_best[i]
                
                # Update velocity
                r1, r2 = random.random(), random.random()
                velocities[i] = (self.w * velocities[i] + 
                               self.c1 * r1 * (personal_best[i] - particles[i]) +
                               self.c2 * r2 * (global_best - particles[i]))
                
                # Update position
                particles[i] = particles[i] + velocities[i]
                
                # Evaluate
                obj = self.evaluate(particles[i], i)
                
                # Update personal best
                if self.dominates(obj, personal_best_obj[i]):
                    personal_best[i] = particles[i].copy()
                    personal_best_obj[i] = obj
                
                # Update archive
                dominated = False
                to_remove = []
                
                for j, arch_entry in enumerate(self.archive):
                    if self.dominates(arch_entry['objectives'], obj):
                        dominated = True
                        break
                    elif self.dominates(obj, arch_entry['objectives']):
                        to_remove.append(j)
                
                if not dominated:
                    for j in reversed(to_remove):
                        del self.archive[j]
                    self.archive.append({
                        'solution': particles[i].copy(),
                        'objectives': obj
                    })
            
            if iteration % 50 == 0:
                print(f"MO-PSO Iter {iteration}/{self.max_iters}, Archive size: {len(self.archive)}")
        
        self.runtime = time.time() - start_time
        return self.archive


# ===============================
# ACO Implementation (Multi-Objective)
# ===============================
class ACO:
    def __init__(self, decision_dim, hosts, vms, containers, pop_size=30, max_iters=200, seed=42):
        self.decision_dim = decision_dim
        self.hosts = hosts
        self.vms = vms
        self.containers = containers
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.num_hosts = len(hosts)
        self.num_vms = len(vms)
        self.num_containers = len(containers)
        
        # Convert to arrays
        self.host_array, self.vm_array, self.container_array = convert_to_arrays(hosts, vms, containers)
        
        self.prev_states = [None] * self.pop_size
        
        # ACO parameters
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0   # Heuristic importance
        self.rho = 0.1    # Evaporation rate
        self.q = 1.0      # Pheromone deposit constant
        
        # Initialize pheromone matrix
        self.pheromones = np.ones(decision_dim) * 0.1
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.archive = []
        self.runtime = 0
    
    def construct_solution(self):
        """Construct a solution using pheromone trails"""
        solution = np.zeros(self.decision_dim)
        
        for i in range(self.decision_dim):
            # Use pheromone to guide random values
            pheromone_influence = self.pheromones[i] ** self.alpha
            heuristic = np.random.randn()  # Random heuristic
            
            # Combine pheromone and heuristic
            solution[i] = pheromone_influence * heuristic
        
        return solution
    
    def evaluate(self, solution, idx):
        """Evaluate objectives"""
        np.random.seed(hash(solution.tobytes()) % (2**32))
        vm_to_host, container_to_vm = decode_solution_matrix_vectorized(
            solution, self.host_array, self.vm_array, self.container_array
        )
        
        state = build_state_vectorized(
            vm_to_host, container_to_vm, 
            self.num_hosts, self.num_vms, self.num_containers
        )
        
        # allocated_cpu = compute_cpu_allocation_vectorized(state, self.host_array, self.vm_array)
        # makespan = compute_makespan_vectorized(state, allocated_cpu, self.container_array)
        # energy = compute_energy_vectorized(state, self.host_array, self.vm_array, allocated_cpu, makespan)
        # cost = compute_cost_vectorized(state, self.vm_array, makespan)
        
        # 2. Compute CPU allocation
        allocations, total_under_alloc_penalty = compute_cpu_allocation_vectorized(state, self.host_array, self.vm_array)
        
        # 3. Compute makespan (needed for cost and energy)
        makespan, vm_times = compute_makespan_vectorized(state, allocations, self.container_array)
        
        # 4. Compute other objectives (all vectorized)
        energy = compute_energy_vectorized(state, self.host_array, self.vm_array, allocations, makespan)
        
        cost = compute_cost_vectorized(state, self.vm_array, makespan, vm_times, total_under_alloc_penalty)
        
        if self.prev_states[idx] is None:
            migration = 0.0
        else:
            migration = compute_migration_time_vectorized(
                state, self.prev_states[idx], self.vm_array, self.container_array,
                generation=0, max_generations=self.max_iters, apply_scaling=False
            )
        
        self.prev_states[idx] = state
        #
        
        return [makespan, cost, energy, migration]
    
    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2"""
        better_in_one = False
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2:
                return False
            if o1 < o2:
                better_in_one = True
        return better_in_one
    
    def update_pheromones(self, solutions, objectives):
        """Update pheromone trails based on solution quality"""
        # Evaporation
        self.pheromones *= (1 - self.rho)
        
        # Deposit pheromones from good solutions
        for sol, obj in zip(solutions, objectives):
            # Quality based on inverse of sum of objectives
            quality = 1.0 / (1.0 + sum(obj))
            
            # Update pheromones proportional to solution quality
            self.pheromones += self.q * quality * np.abs(sol) / (np.max(np.abs(sol)) + 1e-6)
    
    def run(self):
        """Run ACO optimization"""
        start_time = time.time()
        
        for iteration in range(self.max_iters):
            # Construct solutions
            solutions = [self.construct_solution() for _ in range(self.pop_size)]
            
            # Evaluate solutions
            objectives = [self.evaluate(sol, i) for i, sol in enumerate(solutions)]
            
            # Update pheromones
            self.update_pheromones(solutions, objectives)
            
            # Update archive
            for i, obj in enumerate(objectives):
                dominated = False
                to_remove = []
                
                for j, arch_entry in enumerate(self.archive):
                    if self.dominates(arch_entry['objectives'], obj):
                        dominated = True
                        break
                    elif self.dominates(obj, arch_entry['objectives']):
                        to_remove.append(j)
                
                if not dominated:
                    for j in reversed(to_remove):
                        del self.archive[j]
                    self.archive.append({
                        'solution': solutions[i].copy(),
                        'objectives': obj
                    })
            
            if iteration % 50 == 0:
                print(f"ACO Iter {iteration}/{self.max_iters}, Archive size: {len(self.archive)}")
        
        self.runtime = time.time() - start_time
        return self.archive


# ===============================
# Cat Swarm Optimization
# ===============================
class CatSwarm:
    def __init__(self, decision_dim, hosts, vms, containers, pop_size=30, max_iters=200, seed=42):
        self.decision_dim = decision_dim
        self.hosts = hosts
        self.vms = vms
        self.containers = containers
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.num_hosts = len(hosts)
        self.num_vms = len(vms)
        self.num_containers = len(containers)
        
        # Convert to arrays
        self.host_array, self.vm_array, self.container_array = convert_to_arrays(hosts, vms, containers)
        
        self.prev_states = [None] * self.pop_size
        
        # Cat Swarm parameters
        self.MR = 0.8     # Mixture ratio (seeking vs tracing)
        self.SMP = 5      # Seeking memory pool
        self.SRD = 0.2    # Seeking range of the selected dimension
        self.CDC = 0.8    # Counts of dimension to change
        self.c1 = 2.0     # Acceleration constant
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.archive = []
        self.runtime = 0
    
    def initialize_population(self):
        """Initialize cat population"""
        return [np.random.randn(self.decision_dim) for _ in range(self.pop_size)]
    
    def evaluate(self, solution, idx):
        """Evaluate objectives"""
        np.random.seed(hash(solution.tobytes()) % (2**32))
        vm_to_host, container_to_vm = decode_solution_matrix_vectorized(
            solution, self.host_array, self.vm_array, self.container_array
        )
        
        state = build_state_vectorized(
            vm_to_host, container_to_vm, 
            self.num_hosts, self.num_vms, self.num_containers
        )
        
        # allocated_cpu = compute_cpu_allocation_vectorized(state, self.host_array, self.vm_array)
        # makespan = compute_makespan_vectorized(state, allocated_cpu, self.container_array)
        # energy = compute_energy_vectorized(state, self.host_array, self.vm_array, allocated_cpu, makespan)
        # cost = compute_cost_vectorized(state, self.vm_array, makespan)
        
        # 2. Compute CPU allocation
        allocations, total_under_alloc_penalty = compute_cpu_allocation_vectorized(state, self.host_array, self.vm_array)
        
        # 3. Compute makespan (needed for cost and energy)
        makespan, vm_times = compute_makespan_vectorized(state, allocations, self.container_array)
        
        # 4. Compute other objectives (all vectorized)
        energy = compute_energy_vectorized(state, self.host_array, self.vm_array, allocations, makespan)
        
        cost = compute_cost_vectorized(state, self.vm_array, makespan, vm_times, total_under_alloc_penalty)
        
        if self.prev_states[idx] is None:
            migration = 0.0
        else:
            migration = compute_migration_time_vectorized(
                state, self.prev_states[idx], self.vm_array, self.container_array,
                generation=0, max_generations=self.max_iters, apply_scaling=False
            )
        
        self.prev_states[idx] = state
        #
        
        return [makespan, cost, energy, migration]
    
    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2"""
        better_in_one = False
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2:
                return False
            if o1 < o2:
                better_in_one = True
        return better_in_one
    
    def seeking_mode(self, cat, idx):
        """Seeking mode: local search"""
        copies = [cat.copy() for _ in range(self.SMP)]
        
        # Modify copies
        for copy in copies:
            num_dims = int(self.CDC * self.decision_dim)
            dims_to_change = random.sample(range(self.decision_dim), num_dims)
            
            for dim in dims_to_change:
                copy[dim] += np.random.uniform(-self.SRD, self.SRD)
        
        # Evaluate copies
        objectives = [self.evaluate(copy, idx) for copy in copies]
        
        # Select best (roulette wheel based on quality)
        qualities = [1.0 / (1.0 + sum(obj)) for obj in objectives]
        total_quality = sum(qualities)
        probs = [q / total_quality for q in qualities]
        
        selected_idx = np.random.choice(len(copies), p=probs)
        return copies[selected_idx]
    
    def tracing_mode(self, cat, velocity, best_position):
        """Tracing mode: global search"""
        r = random.random()
        velocity = velocity + r * self.c1 * (best_position - cat)
        cat = cat + velocity
        return cat, velocity
    
    def run(self):
        """Run Cat Swarm Optimization"""
        start_time = time.time()
        
        # Initialize cats
        cats = [np.random.randn(self.decision_dim) for _ in range(self.pop_size)]
        velocities = [np.random.randn(self.decision_dim) * 0.1 for _ in range(self.pop_size)]
        flags = [random.random() < self.MR for _ in range(self.pop_size)]  # seeking or tracing
        
        # Initialize best positions
        best_positions = [cat.copy() for cat in cats]
        best_objectives = [self.evaluate(cat, i) for i, cat in enumerate(cats)]
        
        # Global best (from archive)
        global_best = cats[0].copy()
        
        for iteration in range(self.max_iters):
            for i in range(self.pop_size):
                if flags[i]:  # Seeking mode
                    cats[i] = self.seeking_mode(cats[i], i)
                else:  # Tracing mode
                    cats[i], velocities[i] = self.tracing_mode(
                        cats[i], velocities[i], best_positions[i]
                    )
                
                # Evaluate
                obj = self.evaluate(cats[i], i)
                
                # Update personal best
                if self.dominates(obj, best_objectives[i]):
                    best_positions[i] = cats[i].copy()
                    best_objectives[i] = obj
                
                # Update archive
                dominated = False
                to_remove = []
                
                for j, arch_entry in enumerate(self.archive):
                    if self.dominates(arch_entry['objectives'], obj):
                        dominated = True
                        break
                    elif self.dominates(obj, arch_entry['objectives']):
                        to_remove.append(j)
                
                if not dominated:
                    for j in reversed(to_remove):
                        del self.archive[j]
                    self.archive.append({
                        'solution': cats[i].copy(),
                        'objectives': obj
                    })
            
            # Randomly reassign seeking/tracing modes
            flags = [random.random() < self.MR for _ in range(self.pop_size)]
            
            if iteration % 50 == 0:
                print(f"Cat Swarm Iter {iteration}/{self.max_iters}, Archive size: {len(self.archive)}")
        
        self.runtime = time.time() - start_time
        return self.archive


# ===============================
# FCFS (First Come First Serve) - Greedy Baseline
# ===============================
class FCFS:
    def __init__(self, decision_dim, hosts, vms, containers, seed=42):
        self.decision_dim = decision_dim
        self.hosts = hosts
        self.vms = vms
        self.containers = containers
        self.num_hosts = len(hosts)
        self.num_vms = len(vms)
        self.num_containers = len(containers)
        
        # Convert to arrays
        self.host_array, self.vm_array, self.container_array = convert_to_arrays(hosts, vms, containers)
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.archive = []
        self.runtime = 0
    
    def evaluate(self, solution):
        """Evaluate objectives"""
        np.random.seed(hash(solution.tobytes()) % (2**32))
        vm_to_host, container_to_vm = decode_solution_matrix_vectorized(
            solution, self.host_array, self.vm_array, self.container_array
        )
        
        state = build_state_vectorized(
            vm_to_host, container_to_vm, 
            self.num_hosts, self.num_vms, self.num_containers
        )
        
        # allocated_cpu = compute_cpu_allocation_vectorized(state, self.host_array, self.vm_array)
        # makespan = compute_makespan_vectorized(state, allocated_cpu, self.container_array)
        # energy = compute_energy_vectorized(state, self.host_array, self.vm_array, allocated_cpu, makespan)
        # cost = compute_cost_vectorized(state, self.vm_array, makespan)

        # 2. Compute CPU allocation
        allocations, total_under_alloc_penalty = compute_cpu_allocation_vectorized(state, self.host_array, self.vm_array)
        
        # 3. Compute makespan (needed for cost and energy)
        makespan, vm_times = compute_makespan_vectorized(state, allocations, self.container_array)
        
        # 4. Compute other objectives (all vectorized)
        energy = compute_energy_vectorized(state, self.host_array, self.vm_array, allocations, makespan)
        
        cost = compute_cost_vectorized(state, self.vm_array, makespan, vm_times, total_under_alloc_penalty)
        
        migration = 0.0
        
        return [makespan, cost, energy, migration]
    
    def run(self):
        """Run FCFS greedy allocation"""
        start_time = time.time()
        
        # Greedy VM to Host allocation (first-fit)
        vm_to_host = []
        host_loads = [0.0] * self.num_hosts
        
        for vm in self.vms:
            # Find host with minimum load
            min_load_idx = np.argmin(host_loads)
            vm_to_host.append(min_load_idx)
            host_loads[min_load_idx] += vm.cpu_demand
        
        # Greedy Container to VM allocation (first-fit)
        container_to_vm = []
        vm_loads = [0.0] * self.num_vms
        
        for container in self.containers:
            # Find VM with minimum load
            min_load_idx = np.argmin(vm_loads)
            container_to_vm.append(min_load_idx)
            vm_loads[min_load_idx] += container.workload_mi
        
        # Convert to continuous representation for consistency with monso
        num_hosts = self.num_hosts
        num_vms = self.num_vms
        
        solution = []
        for h in vm_to_host:
            logits = np.zeros(num_hosts)
            logits[h] = 10.0  # High value for selected host
            solution.extend(logits.tolist())
        
        for v in container_to_vm:
            logits = np.zeros(num_vms)
            logits[v] = 10.0  # High value for selected VM
            solution.extend(logits.tolist())
        
        solution = np.array(solution)
        
        # Evaluate
        obj = self.evaluate(solution)
        
        self.archive.append({
            'solution': solution,
            'objectives': obj
        })
        
        self.runtime = time.time() - start_time
        
        print(f"FCFS completed in {self.runtime:.3f}s")
        return self.archive


print("Multi-algorithm benchmarking module (MONSO version) loaded successfully!")