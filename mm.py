# ===============================
# MO-NSO Cloud Data Center Optimizer
# Reference implementation following the agreed blueprint.
# - Continuous neurons
# - Deterministic decoding
# - Fixed VM weights (computed once)
# - Offline multi-objective evaluation
# - Pareto archive
# ===============================

from __future__ import annotations
import math
import random
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from scipy import stats
from scipy.spatial.distance import cdist

# -------------------------------------------------
# config.py (inlined)
# -------------------------------------------------


BANDWIDTH_GBPS = 10.0  # Network bandwidth in GB/s
MB_TO_GB = 1.0 / 1024.0  # Conversion factor
IDLE_COST_FACTOR = 1.3   # 30% of active VM price



# -------------------------------------------------
# models.py (inlined)
# -------------------------------------------------


class VMType:
    """VM type definition with MIPS capacity"""
    def __init__(self, name: str, mips_capacity: float, ram_gb: float, price_per_hour: float):
        self.name = name
        self.mips_capacity = mips_capacity  # MIPS (Million Instructions Per Second)
        self.ram_gb = ram_gb
        self.price_per_hour = price_per_hour
        
        

@dataclass
class Host:
    """Individual host representation"""
    host_id: int
    cpu_capacity: float        # Total CPU capacity (MIPS)
    ram_capacity: float        # Total RAM capacity (GB)
    power_idle: float          # Power consumption when idle (watts)
    power_max: float           # Power consumption at 100% load (watts)

@dataclass
class VM:
    """Individual VM representation"""
    vm_id: int
    type: 'VMType'             # VM type definition
    cpu_demand: float          # CPU demand (MIPS)
    cpu_weight: float          # CPU weight for allocation
    containers: List[int]      # List of container IDs

@dataclass
class Container:
    """Individual container representation"""
    container_id: int
    workload_mi: float         # Workload in million instructions
    state_size_mb: float       # State size in MB
    ram_requirement_mb: float  # RAM requirement in MB



@dataclass
class HostArray:
    """Vectorized host representation"""
    cpu_capacities: np.ndarray      # Shape: [num_hosts]
    ram_capacities: np.ndarray      # Shape: [num_hosts]
    power_idles: np.ndarray         # Shape: [num_hosts]
    power_maxs: np.ndarray          # Shape: [num_hosts]

@dataclass
class VMArray:
    """Vectorized VM representation"""
    cpu_demands: np.ndarray         # Shape: [num_vms]
    cpu_capacities: np.ndarray      # Shape: [num_vms] - type limits
    ram_sizes: np.ndarray           # Shape: [num_vms] - GB
    prices_per_sec: np.ndarray      # Shape: [num_vms] - $/sec
    cpu_weights: np.ndarray         # Shape: [num_vms]
    memory_gbs: np.ndarray          # Shape: [num_vms] - for migration

@dataclass
class ContainerArray:
    """Vectorized container representation"""
    workloads_mi: np.ndarray        # Shape: [num_containers]
    state_sizes_mb: np.ndarray      # Shape: [num_containers]
    ram_requirements_mb: np.ndarray # Shape: [num_containers]


# -------------------------------------------------
# encoding.py (inlined)
# -------------------------------------------------

def decode_solution_matrix_vectorized(
    neuron,
    hosts: HostArray,
    vms: VMArray,
    containers: ContainerArray,
    temperature: float = 0.5
):
    """
    Diversity-preserving, capacity-aware decoding.
    NO argmax collapse.
    NO RAM-based global sorting.
    """

    num_hosts = hosts.cpu_capacities.shape[0]
    num_vms = vms.cpu_demands.shape[0]
    num_containers = containers.workloads_mi.shape[0]

    neuron = np.asarray(neuron, dtype=np.float64)

    # ================= VM → HOST =================
    vm_logits = neuron[:num_vms * num_hosts].reshape(num_vms, num_hosts)

    # 🔑 NEURON-DEPENDENT VM ORDER 
    vm_priority = np.sum(vm_logits, axis=1)
    vm_order = np.argsort(-vm_priority)

    vm_to_host = np.zeros(num_vms, dtype=np.int32)
    host_used_ram = np.zeros(num_hosts, dtype=np.float64)
    host_used_cpu = np.zeros(num_hosts, dtype=np.float64)

    for vm_idx in vm_order:
        logits = vm_logits[vm_idx].copy()

        ram_need = vms.ram_sizes[vm_idx]
        cpu_need = vms.cpu_demands[vm_idx]

        # Feasibility masks
        ram_ok = (hosts.ram_capacities - host_used_ram) >= ram_need
        cpu_ok = (hosts.cpu_capacities - host_used_cpu) >= cpu_need
        feasible = ram_ok & cpu_ok

        if not np.any(feasible):
            chosen_host = np.argmin(host_used_ram)
        else:
            feasible_indices = np.where(feasible)[0]
            feasible_logits = logits[feasible_indices]

            # deterministic symmetry breaking
            feasible_logits += 1e-3 * feasible_logits

            # temperature softmax
            scaled = feasible_logits / max(temperature, 1e-6)
            exp_logits = np.exp(scaled - np.max(scaled))
            probs = exp_logits / np.sum(exp_logits)

            chosen_host = feasible_indices[
                np.random.choice(len(probs), p=probs)
            ]

        vm_to_host[vm_idx] = chosen_host
        host_used_ram[chosen_host] += ram_need
        host_used_cpu[chosen_host] += cpu_need

    # ================= CONTAINER → VM =================
    offset = num_vms * num_hosts
    cont_logits = neuron[offset:offset + num_containers * num_vms]
    cont_logits = cont_logits.reshape(num_containers, num_vms)

    cont_logits = cont_logits / max(temperature, 1e-6)
    cont_logits -= np.max(cont_logits, axis=1, keepdims=True)

    exp_logits = np.exp(cont_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 🔑 NO ARGMAX — SAMPLING
    container_to_vm = np.array(
        [np.random.choice(num_vms, p=probs[i]) for i in range(num_containers)],
        dtype=np.int32
    )

    return vm_to_host, container_to_vm







# def decode_solution_matrix_vectorized(neuron, hosts, vms, containers):
#     """
#     Matrix-based vectorized decoding using cumulative capacity checks.
    
#     Args:
#         neuron: Array of shape [neuron_dim] with continuous values
#         hosts: HostArray with vectorized host information
#         vms: VMArray with vectorized VM information  
#         containers: ContainerArray with vectorized container information
    
#     Returns:
#         vm_to_host: Array of shape [num_vms] with host assignments
#         container_to_vm: Array of shape [num_containers] with VM assignments
#     """
#     # Determine input types and extract needed data
#     if isinstance(hosts, HostArray):
#         num_hosts = hosts.cpu_capacities.shape[0]
#         host_capacities = hosts.ram_capacities
#     else:  # List[Host]
#         num_hosts = len(hosts)
#         host_capacities = np.array([h.ram_capacity for h in hosts], dtype=np.float64)
    
#     if isinstance(vms, VMArray):
#         num_vms = vms.cpu_demands.shape[0]
#         vm_ram = vms.ram_sizes  # Already in GB
#     else:  # List[VM]
#         num_vms = len(vms)
#         vm_ram = np.array([vm.type.ram_gb for vm in vms], dtype=np.float64)
    
#     if isinstance(containers, ContainerArray):
#         num_containers = containers.workloads_mi.shape[0]
#     else:  # List[Container]
#         num_containers = len(containers)
    
#     neuron = np.asarray(neuron, dtype=np.float64)
    
#     # --- VM → Host (Matrix operations) ---
#     vm_logits = neuron[:num_vms * num_hosts].reshape(num_vms, num_hosts)
    
#     # 1. Sort VMs by RAM (largest first) for better packing
#     vm_order = np.argsort(-vm_ram)  # Descending
#     vm_ram_sorted = vm_ram[vm_order]
#     vm_logits_sorted = vm_logits[vm_order]
    
#     # 2. Initialize assignments
#     vm_to_host = np.zeros(num_vms, dtype=np.int32)
#     host_used = np.zeros(num_hosts, dtype=np.float64)
    
#     # 3. Vectorized assignment using broadcasting
#     for i, (ram_needed, logits) in enumerate(zip(vm_ram_sorted, vm_logits_sorted)):
#         # Calculate remaining capacity for each host
#         remaining_capacity = host_capacities - host_used
        
#         # Create mask of feasible hosts
#         feasible_mask = remaining_capacity >= ram_needed
        
#         if not np.any(feasible_mask):
#             # No feasible host: use emptiest
#             chosen_host = np.argmin(host_used)
#         else:
#             # Get logits only for feasible hosts
#             feasible_logits = logits[feasible_mask]
#             feasible_indices = np.where(feasible_mask)[0]
            
#             # Softmax on feasible hosts
#             logits_max = np.max(feasible_logits)
#             exp_logits = np.exp(feasible_logits - logits_max)
#             probs = exp_logits / np.sum(exp_logits)
            
#             # Choose based on probabilities (deterministic argmax)
#             chosen_idx = np.argmax(probs)
#             chosen_host = feasible_indices[chosen_idx]
        
#         # Update assignments
#         original_vm_idx = vm_order[i]
#         vm_to_host[original_vm_idx] = chosen_host
#         host_used[chosen_host] += ram_needed
    
#     # --- Container → VM (Fully Vectorized - Batch Processing) ---
#     offset = num_vms * num_hosts
#     cont_logits = neuron[offset:offset + num_containers * num_vms].reshape(num_containers, num_vms)
    
#     # Batch softmax using broadcasting for numerical stability
#     cont_logits_max = np.max(cont_logits, axis=1, keepdims=True)
    
#     # Vectorized exponentiation and normalization
#     cont_exp = np.exp(cont_logits - cont_logits_max)
#     cont_exp_sum = np.sum(cont_exp, axis=1, keepdims=True)
    
#     # Handle potential division by zero
#     cont_exp_sum = np.maximum(cont_exp_sum, 1e-12)
    
#     cont_probs = cont_exp / cont_exp_sum
#     container_to_vm = np.argmax(cont_probs, axis=1)
    
#     return vm_to_host, container_to_vm

# -------------------------------------------------
# state.py (inlined)
# -------------------------------------------------
@dataclass
class PlacementStateVectorized:
    """Vectorized placement state"""
    vm_to_host: np.ndarray          # Shape: [num_vms], dtype=int
    container_to_vm: np.ndarray     # Shape: [num_containers], dtype=int
    # Derived matrices for fast lookups
    host_vm_mask: np.ndarray        # Shape: [num_hosts, num_vms], dtype=bool
    vm_container_mask: np.ndarray   # Shape: [num_vms, num_containers], dtype=bool
    
    
def build_state_vectorized(vm_to_host: np.ndarray, container_to_vm: np.ndarray,
                          num_hosts: int, num_vms: int, num_containers: int) -> PlacementStateVectorized:
    """
    Fully vectorized state construction using numpy broadcasting.
    """
    # Create VM-to-host assignment matrix
    host_indices = np.arange(num_hosts)[:, np.newaxis]
    vm_indices = np.arange(num_vms)
    
    # Boolean mask: host_vm_mask[h, v] = True if VM v is on host h
    host_vm_mask = (vm_to_host[np.newaxis, :] == host_indices)
    
    # Create container-to-VM assignment matrix
    vm_indices_expanded = np.arange(num_vms)[:, np.newaxis]
    container_indices = np.arange(num_containers)
    
    # Boolean mask: vm_container_mask[v, c] = True if container c is on VM v
    vm_container_mask = (container_to_vm[np.newaxis, :] == vm_indices_expanded)
    
    return PlacementStateVectorized(
        vm_to_host=vm_to_host,
        container_to_vm=container_to_vm,
        host_vm_mask=host_vm_mask,
        vm_container_mask=vm_container_mask
    )
    
    
@dataclass
class PreviousStateVectorized:
    vm_to_host: np.ndarray           # Shape: [num_vms], dtype=int
    container_to_vm: np.ndarray      # Shape: [num_containers], dtype=int
    
    
# -------------------------------------------------
# evaluation.py (inlined)
# -------------------------------------------------
    
def compute_cpu_allocation_vectorized(state: PlacementStateVectorized,
                                     hosts: HostArray,
                                     vms: VMArray) -> np.ndarray:
    """
    Vectorized CPU allocation with VM capacity limits.
    Returns: allocated_cpu array shape [num_vms]
    """
    num_hosts = hosts.cpu_capacities.shape[0]
    num_vms = vms.cpu_demands.shape[0]
    
    # Initialize allocations
    allocations = np.zeros(num_vms, dtype=np.float64)
    total_under_alloc_penalty = 0.0
    
    # For each host, compute allocations for its VMs
    for h in range(num_hosts):
        # Get VMs on this host using precomputed mask
        vm_mask = state.host_vm_mask[h, :]
        vm_indices = np.where(vm_mask)[0]
        
        if len(vm_indices) == 0:
            continue
        
        # Get weights of VMs on this host
        weights = vms.cpu_weights[vm_indices]
        total_weight = np.sum(weights)
        
        if total_weight <= 1e-12:
            continue
        
        # Proportional share
        proportional_shares = (weights / total_weight) * hosts.cpu_capacities[h]
        
        # Apply VM capacity limits
        vm_capacities = vms.cpu_capacities[vm_indices]
        allocated_shares = np.minimum(proportional_shares, vm_capacities)
        
        # Ensure minimum 50% of demand if possible
        vm_demands = vms.cpu_demands[vm_indices]
        # min_required = np.minimum(vm_demands * 0.5, vm_capacities)
        # allocated_shares = np.maximum(allocated_shares, min_required)
        
        SOFT_FLOOR = 0.15  # 15%, not 50%

        min_required = np.minimum(vm_demands * SOFT_FLOOR, vm_capacities)
        allocated_shares = np.maximum(allocated_shares, min_required)
        
        under_alloc = np.maximum(vm_demands - allocated_shares, 0.0)
        under_alloc_penalty_host = np.sum(under_alloc ** 1.5)
        
        total_under_alloc_penalty += under_alloc_penalty_host
        
        total_alloc = np.sum(allocated_shares)
        host_cap = hosts.cpu_capacities[h]

        if total_alloc > host_cap:
            scale = host_cap / total_alloc
            allocated_shares *= scale
        
        # Store allocations
        allocations[vm_indices] = allocated_shares
    
    return allocations, total_under_alloc_penalty



def compute_makespan_vectorized(state: PlacementStateVectorized,
                               allocations: np.ndarray,
                               containers: ContainerArray) -> float:
    """
    Vectorized makespan computation.
    Containers run in parallel on each VM.
    """
    num_vms = allocations.shape[0]
    num_containers = containers.workloads_mi.shape[0]
    
    # Ensure no zero allocations
    allocations_safe = np.maximum(allocations, 1e-9)
    
    # Compute total workload per VM using matrix multiplication
    # vm_container_mask: [num_vms, num_containers] boolean
    # workloads: [num_containers]
    # Result: [num_vms] - total workload per VM
    vm_workloads = state.vm_container_mask @ containers.workloads_mi
    
    # Compute time per VM: workload / allocation
    vm_times = vm_workloads / allocations_safe

    
    return float(np.max(vm_times)) if num_vms > 0 else 0.0, vm_times


def compute_energy_vectorized(state: PlacementStateVectorized,
                             hosts: HostArray,
                             vms: VMArray,
                             allocations: np.ndarray,
                             makespan: float) -> float:
    """
    Vectorized energy computation.
    Energy = makespan × sum of host powers.
    """
    num_hosts = hosts.cpu_capacities.shape[0]
    
    # Compute host utilization: sum of allocations on host / host capacity
    host_utilizations = np.zeros(num_hosts, dtype=np.float64)
    
    for h in range(num_hosts):
        vm_mask = state.host_vm_mask[h, :]
        vm_indices = np.where(vm_mask)[0]
        
        if len(vm_indices) == 0:
            # Host idle but still consumes power
            continue
        
        # Sum of allocations on this host
        total_allocated = np.sum(allocations[vm_indices])
        host_capacity = hosts.cpu_capacities[h]
        
        # Utilization (allow up to 150% overcommit)
        utilization = total_allocated / host_capacity
        utilization = np.clip(utilization, 0.0, 1.0)
        host_utilizations[h] = utilization
    
    # Compute host powers using vectorized operations
    # powers = hosts.power_idles + (hosts.power_maxs - hosts.power_idles) * host_utilizations
    
    # # Total power (include idle hosts)
    # total_power = np.sum(powers)
    
    # # Energy = power × time
    # return total_power * makespan
    
    util_penalty = host_utilizations ** 2.2   # 2–3 is realistic

    powers = (
        hosts.power_idles +
        (hosts.power_maxs - hosts.power_idles) * util_penalty
    )

    energy = np.sum(powers) * makespan
    return energy



def compute_cost_vectorized(state: PlacementStateVectorized,
                           vms: VMArray,
                           makespan: float,
                           vm_times,
                           under_alloc_penalty) -> float:
    """
    Vectorized cost computation.
    Cost = makespan × sum of active VM prices.
    """
    num_vms = vms.prices_per_sec.shape[0]
    
    # Find active VMs (VMs with at least one container)
    # vm_container_mask: [num_vms, num_containers]
    active_vms = np.any(state.vm_container_mask, axis=1)  # Shape: [num_vms]
    
    # Idle VMs = allocated but hosting no containers
    idle_vms = ~active_vms
    
    # Sum prices of active VMs
    # active_prices = vms.prices_per_sec[active_vms]
    # total_price_per_sec = np.sum(active_prices)
    
    active_cost_per_sec = np.sum(vms.prices_per_sec[active_vms])
    # idle_cost_per_sec = np.sum(vms.prices_per_sec[idle_vms]) * IDLE_COST_FACTOR
    
    idle_cost_per_sec = np.sum(vms.prices_per_sec[idle_vms] ** 1.3)
    
    total_cost_per_sec = active_cost_per_sec + idle_cost_per_sec
    
    BILLING_QUANTUM = 10.0  # seconds (AWS-style)
    
    billed_time = np.ceil(makespan / BILLING_QUANTUM) * BILLING_QUANTUM
    cost = billed_time * total_cost_per_sec
    
    # ================= SLA PENALTY =================
    SLA = 5.0  # seconds (tunable, experiment parameter)

    sla_violation = np.maximum(vm_times - SLA, 0.0)
    sla_penalty = np.sum(sla_violation ** 2)

    cost += 0.005 * sla_penalty
    
    # ================= STARVATION PENALTY =================
    STARVATION_WEIGHT = 0.001  # tunable

    cost += STARVATION_WEIGHT * under_alloc_penalty

    
    # Cost = makespan × total price per second
    return cost





def compute_migration_time_vectorized(
    current_state: 'PlacementStateVectorized',
    prev_state: 'PlacementStateVectorized',
    vms: 'VMArray',
    containers: 'ContainerArray',
    generation: int = 0,
    max_generations: int = 200,
    apply_scaling: bool = True
) -> float:
    """
    Vectorized migration time computation with optional scaling.
    
    Args:
        current_state: Current placement state
        prev_state: Previous placement state
        vms: Vectorized VM data
        containers: Vectorized container data
        generation: Current generation (for scaling)
        max_generations: Total generations (for scaling)
        apply_scaling: Whether to apply generation-based scaling
    
    Returns:
        Scaled migration time in seconds
    """
    
    # ==================== 1. Compute RAW migration time ====================
    
    # --- VM Migration ---
    # Find which VMs changed hosts (boolean mask)
    vm_migrated_mask = current_state.vm_to_host != prev_state.vm_to_host
    
    # Sum memory of migrated VMs (GB)
    if np.any(vm_migrated_mask):
        vm_migration_data_gb = np.sum(vms.memory_gbs[vm_migrated_mask])
    else:
        vm_migration_data_gb = 0.0
    
    # --- Container Migration ---
    # Find which containers changed VMs
    container_migrated_mask = current_state.container_to_vm != prev_state.container_to_vm
    
    # Sum state size of migrated containers (convert MB to GB)
    if np.any(container_migrated_mask):
        # State sizes in MB, convert to GB
        container_state_mb = np.sum(containers.state_sizes_mb[container_migrated_mask])
        container_migration_data_gb = container_state_mb * MB_TO_GB
    else:
        container_migration_data_gb = 0.0
    
    # --- Total data to migrate ---
    total_data_gb = vm_migration_data_gb + container_migration_data_gb
    
    # --- Migration time (seconds) ---
    if BANDWIDTH_GBPS > 1e-9:
        raw_migration_time = total_data_gb / BANDWIDTH_GBPS
    else:
        raw_migration_time = 0.0
    
    # ==================== 2. Apply scaling ====================
    if apply_scaling:
        scaled_migration_time = _scale_migration_time(
            raw_migration_time, generation, max_generations
        )
    else:
        scaled_migration_time = raw_migration_time
    
    return float(scaled_migration_time)

def _scale_migration_time(
    raw_migration_time: float, 
    generation: int, 
    max_generations: int
) -> float:
    """
    Scale migration time: 0.5 at start → 0.1 at end (linear decay).
    
    Early generations: Higher penalty (exploration control)
    Late generations: Lower penalty (refinement freedom)
    """
    if max_generations <= 0:
        return raw_migration_time
    
    # Calculate progress (0 to 1)
    progress = generation / max_generations
    
    # Linear decay: 0.5 at generation 0 → 0.1 at final generation
    scale = 0.5 * (1.0 - progress) + 0.1
    
    # Clamp to reasonable bounds
    scale = np.clip(scale, 0.05, 1.0)
    
    return raw_migration_time * scale



def evaluate_solution_vectorized(current_state: PlacementStateVectorized,
                                prev_state: PlacementStateVectorized,
                                hosts: HostArray,
                                vms: VMArray,
                                containers: ContainerArray,  generation: int, 
                                max_generations: int) -> np.ndarray:
    """
    Fully vectorized evaluation.
    Returns: [makespan, energy, cost, migration]
    """
    
    # 2. Compute CPU allocation
    allocations, total_under_alloc_penalty = compute_cpu_allocation_vectorized(current_state, hosts, vms)
    
    # 3. Compute makespan (needed for cost and energy)
    makespan, vm_times = compute_makespan_vectorized(current_state, allocations, containers)
    
    # 4. Compute other objectives (all vectorized)
    energy = compute_energy_vectorized(current_state, hosts, vms, allocations, makespan)
    
    cost = compute_cost_vectorized(current_state, vms, makespan, vm_times, total_under_alloc_penalty)
    
    migration = compute_migration_time_vectorized(current_state, prev_state, vms, containers, generation, max_generations, apply_scaling=False)
    
    return np.array([makespan, energy, cost, migration], dtype=np.float64)



# -------------------------------------------------
# archive.py (inlined)
# -------------------------------------------------

# ================================
# Vectorized ArchiveEntry
# ================================

@dataclass
class ArchiveEntry:
    """Vectorized archive entry for efficient storage and operations."""
    neuron: np.ndarray           # Shape: [neuron_dim]
    objectives: np.ndarray       # Shape: [num_objectives]
    raw_objectives: np.ndarray   # Shape: [num_objectives]
    state: Optional[PlacementStateVectorized] = None
    
    def __post_init__(self):
        """Ensure all arrays are numpy arrays."""
        if not isinstance(self.neuron, np.ndarray):
            self.neuron = np.array(self.neuron, dtype=np.float64)
        if not isinstance(self.objectives, np.ndarray):
            self.objectives = np.array(self.objectives, dtype=np.float64)
        if not isinstance(self.raw_objectives, np.ndarray):
            self.raw_objectives = np.array(self.raw_objectives, dtype=np.float64)
    
    @classmethod
    def from_dict(cls, data: dict) -> ArchiveEntry:
        """Create ArchiveEntry from dictionary."""
        return cls(
            neuron=np.array(data['neuron'], dtype=np.float64),
            objectives=np.array(data['objectives'], dtype=np.float64),
            raw_objectives=np.array(data['raw_objectives'], dtype=np.float64),
            state=data.get('state')
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'neuron': self.neuron.tolist(),
            'objectives': self.objectives.tolist(),
            'raw_objectives': self.raw_objectives.tolist(),
            'state': self.state
        }

# ================================
# Fully Vectorized ParetoArchive
# ================================

class VectorizedParetoArchive:
    def __init__(self, max_size: int = 100):
        """
        Fully vectorized Pareto archive with crowding distance.
        
        Args:
            max_size: Maximum number of solutions in archive
        """
        self.max_size = max_size
        
        # Store data in vectorized format
        self._neurons: Optional[np.ndarray] = None          # Shape: [n_solutions, neuron_dim]
        self._objectives: Optional[np.ndarray] = None       # Shape: [n_solutions, n_objectives]
        self._raw_objectives: Optional[np.ndarray] = None   # Shape: [n_solutions, n_objectives]
        self._states: List[Optional[PlacementStateVectorized]] = []
        
        # Cache for fast operations
        self._crowding_distances: Optional[np.ndarray] = None
        self._pareto_ranks: Optional[np.ndarray] = None
        
    @property
    def size(self) -> int:
        """Number of solutions in archive."""
        if self._objectives is None:
            return 0
        return self._objectives.shape[0]
    
    @property
    def is_empty(self) -> bool:
        """Check if archive is empty."""
        return self.size == 0
    
    def _initialize_arrays(self, neuron_dim: int, n_objectives: int):
        """Initialize empty arrays."""
        self._neurons = np.zeros((0, neuron_dim), dtype=np.float64)
        self._objectives = np.zeros((0, n_objectives), dtype=np.float64)
        self._raw_objectives = np.zeros((0, n_objectives), dtype=np.float64)
        self._states = []
        self._crowding_distances = np.array([], dtype=np.float64)
        self._pareto_ranks = np.array([], dtype=np.int32)
    
    def _update_caches(self):
        """Update cached values (crowding distances and Pareto ranks)."""
        if self.is_empty:
            self._crowding_distances = np.array([], dtype=np.float64)
            self._pareto_ranks = np.array([], dtype=np.int32)
            return
        
        # Update Pareto ranks
        self._pareto_ranks = self._calculate_pareto_ranks_batch(self._objectives)
        
        # Update crowding distances
        self._crowding_distances = self._calculate_crowding_distances_batch(self._objectives)
    
    def _calculate_pareto_ranks_batch(self, objectives: np.ndarray) -> np.ndarray:
        """
        Vectorized Pareto rank calculation for batch of solutions.
        
        Args:
            objectives: 2D array of shape [n_solutions, n_objectives]
            
        Returns:
            Pareto ranks for each solution
        """
        n = objectives.shape[0]
        
        if n == 0:
            return np.array([], dtype=np.int32)
        
        # Create 3D arrays for broadcasting
        obj_i = objectives[:, np.newaxis, :]  # [n, 1, m]
        obj_j = objectives[np.newaxis, :, :]  # [1, n, m]
        
        # Vectorized dominance comparisons
        less_equal = np.all(obj_j <= obj_i, axis=2)    # [n, n]
        strictly_less = np.any(obj_j < obj_i, axis=2)  # [n, n]
        dominates_matrix = less_equal & strictly_less  # [n, n]
        
        # Remove self-dominance
        np.fill_diagonal(dominates_matrix, False)
        
        # Count dominators for each solution
        dominated_counts = np.sum(dominates_matrix, axis=0)  # [n]
        
        # Rank = dominated_count + 1 (rank 1 for non-dominated)
        return dominated_counts + 1
    
    def _calculate_crowding_distances_batch(self, objectives: np.ndarray) -> np.ndarray:
        """
        Vectorized crowding distance calculation for batch of solutions.
        
        Args:
            objectives: 2D array of shape [n_solutions, n_objectives]
            
        Returns:
            Crowding distances for each solution
        """
        n, m = objectives.shape
        
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n, dtype=np.float64)
        
        for obj_idx in range(m):
            # Sort by this objective
            sorted_indices = np.argsort(objectives[:, obj_idx])
            sorted_values = objectives[sorted_indices, obj_idx]
            
            # Range of this objective
            obj_range = sorted_values[-1] - sorted_values[0]
            
            if obj_range > 1e-12:
                # Boundary points get large distance
                distances[sorted_indices[0]] += 2.0
                distances[sorted_indices[-1]] += 2.0
                
                # Internal points: (next - prev) / range
                if n > 2:
                    next_values = sorted_values[2:]
                    prev_values = sorted_values[:-2]
                    internal_distances = (next_values - prev_values) / obj_range
                    distances[sorted_indices[1:-1]] += internal_distances
            else:
                # All values equal
                distances[:] += 1.0
        
        return distances
    
    def _fast_dominance_check(self, new_objectives: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Fast vectorized dominance check between new solution and archive.
        
        Args:
            new_objectives: Objectives of new solution, shape [n_objectives]
            
        Returns:
            Tuple: (is_dominated, dominated_indices_mask)
        """
        if self.is_empty:
            return False, np.array([], dtype=bool)
        
        # Vectorized comparison: check if archive dominates new solution
        archive_less_equal = np.all(self._objectives <= new_objectives, axis=1)  # [n]
        archive_strictly_less = np.any(self._objectives < new_objectives, axis=1)  # [n]
        archive_dominates_new = archive_less_equal & archive_strictly_less  # [n]
        
        # Check if new solution dominates any in archive
        new_less_equal = np.all(new_objectives <= self._objectives, axis=1)  # [n]
        new_strictly_less = np.any(new_objectives < self._objectives, axis=1)  # [n]
        new_dominates_archive = new_less_equal & new_strictly_less  # [n]
        
        is_dominated = np.any(archive_dominates_new)
        
        return is_dominated, new_dominates_archive
    
    def add(self, entry: ArchiveEntry):
        """
        Add solution to archive with vectorized dominance checks.
        
        Args:
            entry: Solution to add
        """
        # Initialize arrays if empty
        if self.is_empty:
            neuron_dim = entry.neuron.shape[0]
            n_objectives = entry.objectives.shape[0]
            self._initialize_arrays(neuron_dim, n_objectives)
        
        # Convert to numpy arrays
        neuron = np.asarray(entry.neuron, dtype=np.float64).reshape(1, -1)
        objectives = np.asarray(entry.objectives, dtype=np.float64).reshape(1, -1)
        raw_objectives = np.asarray(entry.raw_objectives, dtype=np.float64).reshape(1, -1)
        
        # Check for duplicates
        if not self.is_empty:
            # Vectorized duplicate check
            duplicates = np.all(np.isclose(self._objectives, objectives, rtol=1e-10, atol=1e-10), axis=1)
            if np.any(duplicates):
                return  # Skip exact duplicate
        
        # Fast dominance check
        is_dominated, dominated_mask = self._fast_dominance_check(objectives[0])
        
        if is_dominated:
            return  # New solution is dominated
        
        # Remove dominated solutions
        if np.any(dominated_mask):
            keep_mask = ~dominated_mask
            
            self._neurons = self._neurons[keep_mask]
            self._objectives = self._objectives[keep_mask]
            self._raw_objectives = self._raw_objectives[keep_mask]
            self._states = [state for i, state in enumerate(self._states) if keep_mask[i]]
        
        # Add new solution
        self._neurons = np.vstack([self._neurons, neuron]) if not self.is_empty else neuron
        self._objectives = np.vstack([self._objectives, objectives]) if not self.is_empty else objectives
        self._raw_objectives = np.vstack([self._raw_objectives, raw_objectives]) if not self.is_empty else raw_objectives
        self._states.append(entry.state)
        
        # Update caches
        self._update_caches()
        
        # Prune if exceeding max size
        if self.size > self.max_size:
            self._prune_by_diversity()
    
    def add_batch(self, entries: List[ArchiveEntry]):
        """
        Add multiple solutions in batch with vectorized operations.
        
        Args:
            entries: List of solutions to add
        """
        if not entries:
            return
        
        # Process each entry
        for entry in entries:
            self.add(entry)
    
    def _prune_by_diversity(self):
        """Prune archive based on crowding distance."""
        if self.size <= self.max_size:
            return
        
        # Sort by crowding distance (descending)
        sorted_indices = np.argsort(self._crowding_distances)[::-1]
        
        # Keep top solutions
        keep_indices = sorted_indices[:self.max_size]
        
        # Update arrays
        self._neurons = self._neurons[keep_indices]
        self._objectives = self._objectives[keep_indices]
        self._raw_objectives = self._raw_objectives[keep_indices]
        self._states = [self._states[i] for i in keep_indices]
        
        # Update caches
        self._update_caches()
    
    def get_all_solutions(self) -> List[ArchiveEntry]:
        """Get all solutions as ArchiveEntry objects."""
        solutions = []
        for i in range(self.size):
            solutions.append(ArchiveEntry(
                neuron=self._neurons[i].copy(),
                objectives=self._objectives[i].copy(),
                raw_objectives=self._raw_objectives[i].copy(),
                state=self._states[i]
            ))
        return solutions
    
    def get_all_objectives(self) -> np.ndarray:
        """Get all objectives as numpy array."""
        if self._objectives is None:
            return np.array([], dtype=np.float64)
        return self._objectives.copy()
    
    def get_all_neurons(self) -> np.ndarray:
        """Get all neurons as numpy array."""
        if self._neurons is None:
            return np.array([], dtype=np.float64)
        return self._neurons.copy()
    
    def get_best_solutions(self, n: int = 5) -> List[ArchiveEntry]:
        """
        Get top N solutions based on crowding distance.
        
        Args:
            n: Number of solutions to return
            
        Returns:
            List of top N solutions
        """
        if self.is_empty or n <= 0:
            return []
        
        n = min(n, self.size)
        
        # Sort by crowding distance (descending)
        sorted_indices = np.argsort(self._crowding_distances)[::-1][:n]
        
        solutions = []
        for idx in sorted_indices:
            solutions.append(ArchiveEntry(
                neuron=self._neurons[idx].copy(),
                objectives=self._objectives[idx].copy(),
                raw_objectives=self._raw_objectives[idx].copy(),
                state=self._states[idx]
            ))
        
        return solutions
    
    def get_statistics(self) -> dict:
        """Get archive statistics using vectorized operations."""
        if self.is_empty:
            return {
                'archive_size': 0,
                'objectives_mean': [],
                'objectives_std': [],
                'crowding_distance_mean': 0.0,
                'pareto_rank_mean': 0.0,
                'hypervolume_approx': 0.0
            }
        
        stats = {
            'archive_size': self.size,
            'objectives_mean': np.mean(self._objectives, axis=0).tolist(),
            'objectives_std': np.std(self._objectives, axis=0).tolist(),
            'crowding_distance_mean': float(np.mean(self._crowding_distances)),
            'pareto_rank_mean': float(np.mean(self._pareto_ranks)),
            'hypervolume_approx': self._calculate_hypervolume()
        }
        
        return stats
    
    def _calculate_hypervolume(self, n_samples: int = 10000) -> float:
        """
        Calculate hypervolume using Monte Carlo sampling.
        
        Args:
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Approximate hypervolume
        """
        if self.is_empty:
            return 0.0
        
        # Reference point: slightly worse than worst values
        ref_point = np.max(self._objectives, axis=0) * 1.1
        
        # Minimum bounds
        min_vals = np.min(self._objectives, axis=0)
        
        # Generate random samples
        samples = np.random.uniform(
            low=min_vals,
            high=ref_point,
            size=(n_samples, self._objectives.shape[1])
        )
        
        # Vectorized dominance checking
        dominated = np.zeros(n_samples, dtype=bool)
        
        # For each objective vector, check which samples it dominates
        for obj in self._objectives:
            # Check if objective dominates each sample
            dominated_by_obj = np.all(obj <= samples, axis=1)
            dominated = dominated | dominated_by_obj
        
        # Calculate volume and hypervolume
        volume = np.prod(ref_point - min_vals)
        hypervolume = volume * (np.sum(dominated) / n_samples)
        
        return float(hypervolume)
    
    def clear(self):
        """Clear the archive."""
        self._neurons = None
        self._objectives = None
        self._raw_objectives = None
        self._states = []
        self._crowding_distances = None
        self._pareto_ranks = None
    
    def save_to_npy(self, prefix: str):
        """
        Save archive to numpy binary files.
        
        Args:
            prefix: Filename prefix (e.g., "archive_")
        """
        if self.is_empty:
            return
        
        np.save(f"{prefix}_neurons.npy", self._neurons)
        np.save(f"{prefix}_objectives.npy", self._objectives)
        np.save(f"{prefix}_raw_objectives.npy", self._raw_objectives)
        
        # Save states metadata (states themselves can't be saved easily)
        with open(f"{prefix}_metadata.txt", 'w') as f:
            f.write(f"size={self.size}\n")
            f.write(f"max_size={self.max_size}\n")
    
    def load_from_npy(self, prefix: str):
        """
        Load archive from numpy binary files.
        
        Args:
            prefix: Filename prefix
        """
        try:
            self._neurons = np.load(f"{prefix}_neurons.npy")
            self._objectives = np.load(f"{prefix}_objectives.npy")
            self._raw_objectives = np.load(f"{prefix}_raw_objectives.npy")
            
            # Initialize states as None (can't save PlacementStateVectorized)
            self._states = [None] * self.size
            
            # Update caches
            self._update_caches()
            
        except FileNotFoundError:
            print(f"Warning: Archive files not found for prefix '{prefix}'")
            self.clear()
    
    def save_to_csv(self, filename: str):
        """Save archive to CSV file."""
        if self.is_empty:
            return
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            neuron_dim = self._neurons.shape[1]
            n_objectives = self._objectives.shape[1]
            
            header = ['solution_id']
            header.extend([f'neuron_{i}' for i in range(neuron_dim)])
            header.extend([f'objective_{i}' for i in range(n_objectives)])
            header.extend([f'raw_objective_{i}' for i in range(n_objectives)])
            writer.writerow(header)
            
            # Write data
            for idx in range(self.size):
                row = [idx]
                row.extend(self._neurons[idx].tolist())
                row.extend(self._objectives[idx].tolist())
                row.extend(self._raw_objectives[idx].tolist())
                writer.writerow(row)


# -------------------------------------------------
# population_evaluation.py (missing)
# -------------------------------------------------

# def evaluate_population_vectorized(neurons: np.ndarray,
#                                   prev_states: List[Optional[PlacementStateVectorized]],
#                                   hosts: HostArray,
#                                   vms: VMArray,
#                                   containers: ContainerArray,
#                                   generation: int,
#                                   max_generations: int) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Evaluate entire population in vectorized fashion.
#     Returns: (raw_objectives, normalized_objectives)
#     """
#     pop_size = neurons.shape[0]
#     num_objectives = 4
    
#     # Initialize arrays
#     raw_objectives = np.zeros((pop_size, num_objectives), dtype=np.float64)
    
#     # Process each solution
#     for i in range(pop_size):
#         # Decode neuron to placement
#         vm_to_host, container_to_vm = decode_solution_matrix_vectorized(
#             neurons[i], hosts, vms, containers  # Fixed: passing actual objects, not just counts
#         )
        
#         # Build vectorized state
#         current_state = build_state_vectorized(
#             vm_to_host, container_to_vm,
#             hosts.cpu_capacities.shape[0],
#             vms.cpu_demands.shape[0],
#             containers.workloads_mi.shape[0]
#         )
        
#         # Get previous state (or use current if None)
#         prev_state = prev_states[i] if prev_states[i] is not None else current_state
        
#         # Evaluate
#         raw_objectives[i] = evaluate_solution_vectorized(
#             current_state, prev_state, hosts, vms, containers, generation, max_generations
#         )
        
#         # Update previous state
#         prev_states[i] = current_state
    
#     # Normalize objectives per generation
#     obj_min = np.min(raw_objectives, axis=0, keepdims=True)
#     obj_max = np.max(raw_objectives, axis=0, keepdims=True)
#     obj_range = obj_max - obj_min
    
#     # Avoid division by zero
#     obj_range = np.where(obj_range < 1e-9, 1.0, obj_range)
    
#     normalized_objectives = (raw_objectives - obj_min) / obj_range
    
#     return raw_objectives, normalized_objectives


def calculate_pareto_ranks_vectorized(objectives: np.ndarray) -> np.ndarray:
    """
    Vectorized Pareto rank calculation.
    """
    n = objectives.shape[0]
    
    # Create 3D arrays for broadcasting
    obj_i = objectives[:, np.newaxis, :]  # [n, 1, m]
    obj_j = objectives[np.newaxis, :, :]  # [1, n, m]
    
    # Vectorized dominance comparisons
    less_equal = np.all(obj_j <= obj_i, axis=2)    # [n, n]
    strictly_less = np.any(obj_j < obj_i, axis=2)  # [n, n]
    dominates_matrix = less_equal & strictly_less  # [n, n]
    
    # Remove self-dominance
    np.fill_diagonal(dominates_matrix, False)
    
    # Count dominators for each solution
    dominated_counts = np.sum(dominates_matrix, axis=0)  # [n]
    
    # Rank = dominated_count + 1 (rank 1 for non-dominated)
    return dominated_counts + 1


def calculate_crowding_distances_vectorized(objectives: np.ndarray) -> np.ndarray:
    """
    Fully vectorized crowding distance calculation.
    """
    n, m = objectives.shape
    
    if n <= 2:
        return np.full(n, np.inf)
    
    distances = np.zeros(n, dtype=np.float64)
    
    for obj_idx in range(m):
        # Sort by this objective
        sorted_indices = np.argsort(objectives[:, obj_idx])
        sorted_values = objectives[sorted_indices, obj_idx]
        
        # Range of this objective
        obj_range = sorted_values[-1] - sorted_values[0]
        
        if obj_range > 1e-12:
            # Boundary points get large distance
            distances[sorted_indices[0]] += 2.0
            distances[sorted_indices[-1]] += 2.0
            
            # Internal points: (next - prev) / range
            if n > 2:
                next_values = sorted_values[2:]
                prev_values = sorted_values[:-2]
                internal_distances = (next_values - prev_values) / obj_range
                distances[sorted_indices[1:-1]] += internal_distances
        else:
            # All values equal
            distances[:] += 1.0
    
    return distances




# -------------------------------------------------
# optimizer_nso_vectorized.py (inlined)
# -------------------------------------------------

class MO_NSO_Optimizer_Vectorized:
    """
    Fully vectorized Multi-Objective Neuron Synapse Optimization.
    """
    
    def __init__(
        self,
        hosts: HostArray,
        vms: VMArray,
        containers: ContainerArray,
        pop_size: int = 100,
        alpha: float = 0.05,
        beta: float = 1.0,
        gamma: float = 0.05,
        gamma_f: float = 0.5,
        delta: float = 0.1,
        eta: float = 0.1,
        max_iters: int = 200,
        epsilon: float = 1e-8,
        archive_size: int = 50,
        seed: Optional[int] = None
    ):
        # Store vectorized data structures
        self.hosts = hosts
        self.vms = vms
        self.containers = containers
        
        # Dimensions
        self.num_hosts = hosts.cpu_capacities.shape[0]
        self.num_vms = vms.cpu_demands.shape[0]
        self.num_containers = containers.workloads_mi.shape[0]
        
        # Calculate neuron dimension
        self.neuron_dim = (self.num_vms * self.num_hosts) + (self.num_containers * self.num_vms)
        self.pop_size = pop_size
        self.max_iters = max_iters
        
        # NSO parameters (keep same logic)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gamma_f = gamma_f
        self.delta = delta
        self.eta = eta
        self.epsilon = epsilon
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize neurons in [-5, 5] as in original
        self.neurons = np.random.uniform(-5, 5, size=(pop_size, self.neuron_dim))
        
        # Initialize synaptic weights W in [0, 1], no self-connections
        self.W = np.random.rand(pop_size, pop_size)
        np.fill_diagonal(self.W, 0.0)
        
        # Use VECTORIZED archive
        self.archive = VectorizedParetoArchive()
        
        # For tracking previous states (vectorized approach)
        self.prev_states: List[Optional[PlacementStateVectorized]] = [None] * pop_size
        
        # Cache for fast operations
        self._precompute_indices()
    
    def _precompute_indices(self):
        """Pre-compute indices for faster operations."""
        self.host_indices = np.arange(self.num_hosts)[:, np.newaxis]
        self.vm_indices = np.arange(self.num_vms)
        self.container_indices = np.arange(self.num_containers)
    
    def _decode_population_vectorized(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        VECTORIZED: Decode all neurons in batch.
        """
        vm_to_host_list = []
        container_to_vm_list = []
        
        # Process in small batches for memory efficiency
        batch_size = min(50, self.pop_size)
        
        for batch_start in range(0, self.pop_size, batch_size):
            batch_end = min(batch_start + batch_size, self.pop_size)
            batch_neurons = self.neurons[batch_start:batch_end]
            
            for i in range(batch_end - batch_start):
                vm_to_host, container_to_vm = decode_solution_matrix_vectorized(
                    batch_neurons[i], self.hosts, self.vms, self.containers
                )
                vm_to_host_list.append(vm_to_host)
                container_to_vm_list.append(container_to_vm)
        
        return vm_to_host_list, container_to_vm_list
    
#     def _evaluate_population_vectorized(
#     self, 
#     vm_to_host_list: List[np.ndarray], 
#     container_to_vm_list: List[np.ndarray]
# ) -> np.ndarray:
#         """
#         VECTORIZED: Evaluate all solutions using vectorized functions.
#         Returns: normalized objectives matrix of shape (pop_size, 4)
#         """
#         pop_size = self.pop_size
#         num_objectives = 4
        
#         # Initialize arrays
#         raw_objectives = np.zeros((pop_size, num_objectives), dtype=np.float64)
#         current_states = []
        
#         # Process each solution
#         for i in range(pop_size):
#             # Use the already-decoded placements
#             current_state = build_state_vectorized(
#                 vm_to_host_list[i], container_to_vm_list[i],
#                 self.num_hosts, self.num_vms, self.num_containers
#             )
#             current_states.append(current_state)
            
#             # Get previous state (or use current if None)
#             prev_state = self.prev_states[i] if self.prev_states[i] is not None else current_state
            
#             # Evaluate
#             raw_objectives[i] = evaluate_solution_vectorized(
#                 current_state, prev_state, self.hosts, self.vms, self.containers,
#                 self.current_iteration, self.max_iters
#             )
            
#             # Update previous state for NEXT iteration
#             if self.current_iteration % 5 == 0:
#                 self.prev_states[i] = current_state
        
#         # Normalize objectives per generation
#         obj_min = np.min(raw_objectives, axis=0, keepdims=True)
#         obj_max = np.max(raw_objectives, axis=0, keepdims=True)
#         obj_range = obj_max - obj_min
        
#         # Avoid division by zero
#         obj_range = np.where(obj_range < 1e-9, 1.0, obj_range)
        
#         normalized_objectives = (raw_objectives - obj_min) / obj_range
        
#         # Update archive with vectorized entries
#         for i in range(pop_size):
#             entry = ArchiveEntry(
#                 neuron=self.neurons[i].copy(),
#                 objectives=normalized_objectives[i].copy(),
#                 raw_objectives=raw_objectives[i].copy(),
#                 state=current_states[i]
#             )
#             self.archive.add(entry)
        
#         return normalized_objectives
    
    
    


# Update the _evaluate_population_vectorized method in MO_NSO_Optimizer_Vectorized class:
    def _evaluate_population_vectorized(
        self, 
        vm_to_host_list: List[np.ndarray], 
        container_to_vm_list: List[np.ndarray]
    ) -> np.ndarray:
            """
            VECTORIZED: Evaluate all solutions using vectorized functions.
            Returns: normalized objectives matrix of shape (pop_size, 4)
            """
            pop_size = self.pop_size
            num_objectives = 4
            
            # Initialize arrays
            raw_objectives = np.zeros((pop_size, num_objectives), dtype=np.float64)
            current_states = []
            
            # Process each solution
            for i in range(pop_size):
                # Use the already-decoded placements
                current_state = build_state_vectorized(
                    vm_to_host_list[i], container_to_vm_list[i],
                    self.num_hosts, self.num_vms, self.num_containers
                )
                current_states.append(current_state)
                
                # Get previous state (or use current if None)
                prev_state = self.prev_states[i] if self.prev_states[i] is not None else current_state
                
                # Evaluate
                raw_objectives[i] = evaluate_solution_vectorized(
                    current_state, prev_state, self.hosts, self.vms, self.containers,
                    self.current_iteration, self.max_iters
                )
                
                # Update previous state for NEXT iteration
                self.prev_states[i] = current_state
            
            # NORMALIZE OBJECTIVES using the new robust function
            normalized_objectives = normalize_objectives_vectorized(raw_objectives)
            
            # Update archive with vectorized entries
            for i in range(pop_size):
                entry = ArchiveEntry(
                    neuron=self.neurons[i].copy(),
                    objectives=normalized_objectives[i].copy(),
                    raw_objectives=raw_objectives[i].copy(),
                    state=current_states[i]
                )
                self.archive.add(entry)
            
            return normalized_objectives

# Update the VectorizedParetoArchive class dominance checking:

    def _fast_dominance_check(self, new_objectives: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Fast vectorized dominance check with tolerance.
        
        Args:
            new_objectives: Objectives of new solution, shape [n_objectives]
            
        Returns:
            Tuple: (is_dominated, dominated_indices_mask)
        """
        if self.is_empty:
            return False, np.array([], dtype=bool)
        
        # Use tolerance for comparison to handle numerical issues
        tolerance = 1e-8
        
        # Vectorized comparison with tolerance
        archive_less_equal = np.all(self._objectives <= new_objectives + tolerance, axis=1)
        archive_strictly_less = np.any(self._objectives < new_objectives - tolerance, axis=1)
        archive_dominates_new = archive_less_equal & archive_strictly_less
        
        # Check if new solution dominates any in archive
        new_less_equal = np.all(new_objectives <= self._objectives + tolerance, axis=1)
        new_strictly_less = np.any(new_objectives < self._objectives - tolerance, axis=1)
        new_dominates_archive = new_less_equal & new_strictly_less
        
        is_dominated = np.any(archive_dominates_new)
        
        return is_dominated, new_dominates_archive
    
    def add(self, entry: ArchiveEntry):
        """
        Add solution to archive with improved dominance checks.
        
        Args:
            entry: Solution to add
        """
        # Initialize arrays if empty
        if self.is_empty:
            neuron_dim = entry.neuron.shape[0]
            n_objectives = entry.objectives.shape[0]
            self._initialize_arrays(neuron_dim, n_objectives)
        
        # Convert to numpy arrays
        neuron = np.asarray(entry.neuron, dtype=np.float64).reshape(1, -1)
        objectives = np.asarray(entry.objectives, dtype=np.float64).reshape(1, -1)
        raw_objectives = np.asarray(entry.raw_objectives, dtype=np.float64).reshape(1, -1)
        
        # Check for duplicates with tolerance
        if not self.is_empty:
            # Vectorized duplicate check with tolerance
            objectives_reshaped = objectives.reshape(1, -1)
            differences = np.abs(self._objectives - objectives_reshaped)
            duplicates = np.all(differences < 1e-6, axis=1)
            if np.any(duplicates):
                return  # Skip near-duplicates
        
        # Fast dominance check
        is_dominated, dominated_mask = self._fast_dominance_check(objectives[0])
        
        if is_dominated:
            return  # New solution is dominated
        
        # Remove dominated solutions
        if np.any(dominated_mask):
            keep_mask = ~dominated_mask
            
            self._neurons = self._neurons[keep_mask]
            self._objectives = self._objectives[keep_mask]
            self._raw_objectives = self._raw_objectives[keep_mask]
            self._states = [state for i, state in enumerate(self._states) if keep_mask[i]]
        
        # Add new solution
        if self._neurons is None or self._neurons.shape[0] == 0:
            self._neurons = neuron
            self._objectives = objectives
            self._raw_objectives = raw_objectives
        else:
            self._neurons = np.vstack([self._neurons, neuron])
            self._objectives = np.vstack([self._objectives, objectives])
            self._raw_objectives = np.vstack([self._raw_objectives, raw_objectives])
        
        self._states.append(entry.state)
        
        # Update caches
        self._update_caches()
        
        # Prune if exceeding max size
        if self.size > self.max_size:
            self._prune_by_diversity()
    
    
    
    
    def _calculate_pareto_rank_vectorized(self, objectives: np.ndarray) -> np.ndarray:
        """
        VECTORIZED: Use the function we already defined.
        """
        return calculate_pareto_ranks_vectorized(objectives)
    
    def _synapse_adjustment_mo_vectorized(self, objectives: np.ndarray, ranks: np.ndarray):
        """
        VECTORIZED: Keep your existing vectorized logic (it's already good).
        """
        N = self.pop_size
        
        # Vectorized distance calculation
        X_diff = self.neurons[:, np.newaxis, :] - self.neurons[np.newaxis, :, :]
        d_ij = np.linalg.norm(X_diff, axis=2)
        
        # Normalize ranks
        rank_min, rank_max = ranks.min(), ranks.max()
        if rank_max > rank_min:
            ranks_norm = (ranks - rank_min) / (rank_max - rank_min)
        else:
            ranks_norm = np.zeros_like(ranks)
        
        # Vectorized rank differences
        rank_diff = np.abs(ranks_norm[:, np.newaxis] - ranks_norm[np.newaxis, :])
        
        # Vectorized Hebbian update (KEEP YOUR LOGIC)
        hebbian = np.exp(-self.gamma_f * rank_diff)
        distance = np.exp(-self.beta * d_ij)
        delta_W = self.alpha * hebbian * distance
        
        # Update W (KEEP YOUR LOGIC)
        self.W = np.clip((1 - 0.01) * self.W + delta_W, 0.0, 1.0)
        np.fill_diagonal(self.W, 0.0)
    
    def _neuron_movement_mo_vectorized(self, ranks: np.ndarray):
        """
        VECTORIZED: Keep your existing vectorized logic with FIXED archive guidance.
        """
        # Weight connections by inverse rank
        rank_weights = 1.0 / (ranks[:, np.newaxis] + 1e-12)
        W_weighted = self.W * rank_weights.T
        
        # Vectorized influence calculation
        WX = W_weighted @ self.neurons
        row_sums = W_weighted.sum(axis=1, keepdims=True)
        influence = WX - row_sums * self.neurons
        
        # CORRECT Archive guidance (vectorized) - EXACTLY like your original
        if self.archive.size > 0:
            # Get all solutions from archive
            archive_solutions = self.archive.get_all_solutions()
            
            # Extract neurons from ArchiveEntry objects (same as your original)
            archive_neurons_list = [entry.neuron for entry in archive_solutions]
            archive_neurons = np.array(archive_neurons_list, dtype=np.float64)
            
            # Calculate centroid (same as your original)
            centroid = np.mean(archive_neurons, axis=0)
            
            # Apply archive guidance (same strength 0.1 as your original)
            # Broadcasting: centroid shape [neuron_dim] -> [pop_size, neuron_dim]
            influence += 0.1 * (centroid - self.neurons)
        
        # Update with perturbation (KEEP YOUR LOGIC)
        self.neurons += 0.05 * influence + self.gamma * np.random.normal(size=self.neurons.shape)
        self.neurons = np.clip(self.neurons, -5, 5)
        
    
    def _pruning_and_reinforcement_mo_vectorized(self, ranks: np.ndarray):
        """
        VECTORIZED: Keep your existing vectorized logic.
        """
        # Vectorized pruning (weakest 25%)
        if np.any(self.W > 0):
            threshold = np.percentile(self.W[self.W > 0], 25)
            self.W[self.W < threshold] = 0.0
        
        # Vectorized reinforcement to good solutions
        good_mask = ranks == 1
        if np.any(good_mask):
            good_indices = np.where(good_mask)[0]
            # Vectorized update for all connections to good solutions
            for j in good_indices:
                mask = np.ones(self.pop_size, dtype=bool)
                mask[j] = False  # Skip self
                self.W[mask, j] = np.minimum(self.W[mask, j] + self.eta, 1.0)
    
    def run(self) -> VectorizedParetoArchive:
        """
        VECTORIZED: Run the optimization with vectorized operations.
        """
        for iteration in range(self.max_iters):
            self.current_iteration = iteration
            
            # 1. Decode population (vectorized batch)
            vm_to_host_list, container_to_vm_list = self._decode_population_vectorized()
            
            # 2. Evaluate population (fully vectorized)
            objectives = self._evaluate_population_vectorized(vm_to_host_list, container_to_vm_list)
            
            # 3. Calculate Pareto ranks (vectorized)
            ranks = self._calculate_pareto_rank_vectorized(objectives)
            
            # 4. NSO operations (already vectorized - keep your logic)
            self._synapse_adjustment_mo_vectorized(objectives, ranks)
            self._neuron_movement_mo_vectorized(ranks)
            self._pruning_and_reinforcement_mo_vectorized(ranks)
            
            # 5. Progress tracking
            if iteration % 50 == 0:
                stats = self.get_statistics()
                print(f"Iter {iteration}: Archive={stats['archive_size']}, "
                      f"Rank-1={np.sum(ranks==1)}, "
                      f"Avg Crowding={stats['crowding_distance_mean']:.3f}")
        
        return self.archive
    
    def get_statistics(self) -> dict:
        """Get statistics from vectorized archive."""
        stats = self.archive.get_statistics()
        stats.update({
            'population_size': self.pop_size,
            'current_iteration': self.current_iteration if hasattr(self, 'current_iteration') else 0,
            'neuron_dim': self.neuron_dim,
            'num_hosts': self.num_hosts,
            'num_vms': self.num_vms,
            'num_containers': self.num_containers
        })
        return stats
    
    def analyze_pareto_front_raw(self, precision: int = 4):
        """Analyze Pareto front using raw objectives."""
        if self.archive.is_empty:
            print("Archive is empty!")
            return
        
        solutions = self.archive.get_all_solutions()
        raw_objectives = np.array([sol.raw_objectives for sol in solutions])
        
        print("\n=== Pareto Front Analysis (RAW OBJECTIVES) ===")
        print(f"Total solutions: {len(solutions)}\n")
        
        # Ranges
        print("Objective ranges (RAW):")
        print(f"Makespan   : [{raw_objectives[:,0].min():.{precision}f}, "
            f"{raw_objectives[:,0].max():.{precision}f}] seconds")
        print(f"Energy     : [{raw_objectives[:,1].min():.{precision}f}, "
            f"{raw_objectives[:,1].max():.{precision}f}] Joules")
        print(f"Cost       : [${raw_objectives[:,2].min():.{precision}f}, "
            f"${raw_objectives[:,2].max():.{precision}f}]")
        print(f"Migration  : [{raw_objectives[:,3].min():.{precision}f}, "
            f"{raw_objectives[:,3].max():.{precision}f}] seconds")
        
        # Find extreme solutions
        fastest_idx = np.argmin(raw_objectives[:, 0])
        lowest_energy_idx = np.argmin(raw_objectives[:, 1])
        cheapest_idx = np.argmin(raw_objectives[:, 2])
        lowest_migration_idx = np.argmin(raw_objectives[:, 3])
        
        print("\nExtreme solutions (RAW):")
        print(f"Fastest makespan (#{fastest_idx}): {raw_objectives[fastest_idx][0]:.{precision}f} seconds")
        print(f"  Energy: {raw_objectives[fastest_idx][1]:.{precision}f}, "
            f"Cost: ${raw_objectives[fastest_idx][2]:.{precision}f}, "
            f"Migration: {raw_objectives[fastest_idx][3]:.{precision}f} seconds")
        
        print(f"Lowest energy (#{lowest_energy_idx}): {raw_objectives[lowest_energy_idx][1]:.{precision}f} Joules")
        print(f"  Makespan: {raw_objectives[lowest_energy_idx][0]:.{precision}f}, "
            f"Cost: ${raw_objectives[lowest_energy_idx][2]:.{precision}f}, "
            f"Migration: {raw_objectives[lowest_energy_idx][3]:.{precision}f} seconds")
        
        print(f"Cheapest cost (#{cheapest_idx}): ${raw_objectives[cheapest_idx][2]:.{precision}f}")
        print(f"  Makespan: {raw_objectives[cheapest_idx][0]:.{precision}f}, "
            f"Energy: {raw_objectives[cheapest_idx][1]:.{precision}f}, "
            f"Migration: {raw_objectives[cheapest_idx][3]:.{precision}f} seconds")
        
        print(f"Lowest migration (#{lowest_migration_idx}): {raw_objectives[lowest_migration_idx][3]:.{precision}f} seconds")
        print(f"  Makespan: {raw_objectives[lowest_migration_idx][0]:.{precision}f}, "
            f"Energy: {raw_objectives[lowest_migration_idx][1]:.{precision}f}, "
            f"Cost: ${raw_objectives[lowest_migration_idx][2]:.{precision}f}")
        
        # Find balanced solutions (closest to "utopia point")
        # Utopia point: minimum of each objective
        utopia_point = np.min(raw_objectives, axis=0)
        
        # Normalize for distance calculation
        obj_min = np.min(raw_objectives, axis=0)
        obj_max = np.max(raw_objectives, axis=0)
        obj_range = obj_max - obj_min
        obj_range = np.where(obj_range < 1e-9, 1.0, obj_range)
        normalized_obj = (raw_objectives - obj_min) / obj_range
        
        # Calculate distances to utopia point (normalized)
        utopia_normalized = np.zeros(4)  # [0, 0, 0, 0] in normalized space
        distances = np.linalg.norm(normalized_obj - utopia_normalized, axis=1)
        
        # Find top 5 balanced solutions
        balanced_indices = np.argsort(distances)[:5]
        
        print("\nTop 5 Balanced Solutions (closest to ideal):")
        for i, idx in enumerate(balanced_indices):
            dist = distances[idx]
            print(f"Sol #{idx} (dist={dist:.3f}): "
                f"Makespan={raw_objectives[idx][0]:.{precision}f}s, "
                f"Energy={raw_objectives[idx][1]:.{precision}f}J, "
                f"Cost=${raw_objectives[idx][2]:.{precision}f}, "
                f"Migration={raw_objectives[idx][3]:.{precision}f}s")
        
        # Trade-off analysis
        print("\nTrade-off Analysis:")
        
        # Makespan vs Energy trade-off
        makespan_energy_corr = np.corrcoef(raw_objectives[:, 0], raw_objectives[:, 1])[0, 1]
        print(f"Makespan-Energy correlation: {makespan_energy_corr:.3f}")
        
        # Makespan vs Cost trade-off
        makespan_cost_corr = np.corrcoef(raw_objectives[:, 0], raw_objectives[:, 2])[0, 1]
        print(f"Makespan-Cost correlation: {makespan_cost_corr:.3f}")
        
        # Energy vs Migration trade-off
        energy_migration_corr = np.corrcoef(raw_objectives[:, 1], raw_objectives[:, 3])[0, 1]
        print(f"Energy-Migration correlation: {energy_migration_corr:.3f}")
        
        
        print("\nSample Pareto solutions (RAW):")
        for i, r in enumerate(raw_objectives[:]):
            print(
                f"Sol {i}: "
                f"Makespan={r[0]:.{precision}f}, "
                f"Energy={r[1]:.{precision}f}, "
                f"Cost={r[2]:.{precision}f}, "
                f"Migration={r[3]:.{precision}f}"
            )

        
        
        # Return the actual best balanced solution for further analysis
        best_balanced_idx = balanced_indices[0]
        return solutions[best_balanced_idx]


    # Add this function to normalize objectives properly
def normalize_objectives_vectorized(objectives: np.ndarray) -> np.ndarray:
        """
        Normalize objectives with robust handling of edge cases.
        
        Args:
            objectives: Raw objectives matrix shape [n, m]
            
        Returns:
            Normalized objectives matrix
        """
        if objectives.shape[0] == 0:
            return objectives
        
        # Calculate min and max for each objective
        obj_min = np.min(objectives, axis=0, keepdims=True)
        obj_max = np.max(objectives, axis=0, keepdims=True)
        obj_range = obj_max - obj_min
        
        # For objectives with zero range (all same value), set range to 1
        zero_range_mask = obj_range < 1e-9
        obj_range[zero_range_mask] = 1.0
        
        # Normalize
        normalized = (objectives - obj_min) / obj_range
        
        # Apply small noise to break ties (helps with diversity)
        if objectives.shape[0] > 1:
            noise_scale = 1e-6
            normalized += np.random.normal(0, noise_scale, normalized.shape)
        
        return normalized

# -------------------------------------------------
# realistic_data_generator_mips.py
# -------------------------------------------------

def generate_realistic_dataset_mips(
    num_hosts: int = 10,
    num_vms: int = 50,
    num_containers: int = 200,
    seed: int = 42
) -> Tuple[List[Host], List[VM], List[Container], HostArray, VMArray, ContainerArray]:
    """
    Generate realistic cloud data center dataset with MIPS-based CPU capacities.
    
    Returns:
        - Lists of individual objects (for debugging)
        - Vectorized arrays (for optimization)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # ------------------------------------------------------------------
    # 1. Define realistic VM types with MIPS capacities
    # Typical MIPS values:
    # - Modern CPU core: 5,000-15,000 MIPS per core
    # - Entire server: 50,000-500,000 MIPS
    # ------------------------------------------------------------------
    vm_types = [
        # Small VMs (shared cores)
        VMType("t3.nano", 2500, 0.5, 0.0052),      # ~0.5 core equivalent
        VMType("t3.micro", 5000, 1.0, 0.0104),     # ~1 core equivalent
        VMType("t3.small", 10000, 2.0, 0.0208),    # ~2 core equivalent
        VMType("t3.medium", 20000, 4.0, 0.0416),   # ~4 core equivalent
        
        # General purpose VMs
        VMType("m5.large", 30000, 8.0, 0.096),     # ~6 core equivalent
        VMType("m5.xlarge", 60000, 16.0, 0.192),   # ~12 core equivalent
        VMType("m5.2xlarge", 120000, 32.0, 0.384), # ~24 core equivalent
        VMType("m5.4xlarge", 240000, 64.0, 0.768), # ~48 core equivalent
        
        # Compute optimized VMs (higher MIPS/core)
        VMType("c5.large", 40000, 4.0, 0.085),     # ~8 core equivalent
        VMType("c5.xlarge", 80000, 8.0, 0.170),    # ~16 core equivalent
        VMType("c5.2xlarge", 160000, 16.0, 0.340), # ~32 core equivalent
        
        # Memory optimized VMs
        VMType("r5.large", 30000, 16.0, 0.126),    # ~6 core equivalent
        VMType("r5.xlarge", 60000, 32.0, 0.252),   # ~12 core equivalent
    ]
    
    # ------------------------------------------------------------------
    # 2. Generate realistic hosts with MIPS capacities
    # Typical server MIPS ranges:
    # - Low-end: 50,000-100,000 MIPS
    # - Mid-range: 100,000-300,000 MIPS
    # - High-end: 300,000-800,000 MIPS
    # ------------------------------------------------------------------
    hosts = []
    host_cpu_capacities = []  # Now in MIPS
    host_ram_capacities = []
    host_power_idles = []
    host_power_maxs = []
    
    for i in range(num_hosts):
        # Create heterogeneous hosts
        if i < 3:  # 30% high-end hosts (modern servers)
            # High-end: Dual Xeon, 48-96 cores total
            mips_capacity = np.random.uniform(400000, 800000)  # 400K-800K MIPS
            ram_capacity = np.random.uniform(192, 512)  # 192-512 GB
            power_idle = np.random.uniform(150, 250)  # 150-250W idle
            power_max = np.random.uniform(600, 1000)  # 600-1000W max
            host_type = "High-end"
        elif i < 7:  # 40% mid-range hosts
            # Mid-range: Single Xeon, 16-32 cores
            mips_capacity = np.random.uniform(150000, 350000)  # 150K-350K MIPS
            ram_capacity = np.random.uniform(96, 192)  # 96-192 GB
            power_idle = np.random.uniform(100, 180)  # 100-180W idle
            power_max = np.random.uniform(350, 600)  # 350-600W max
            host_type = "Mid-range"
        else:  # 30% low-end/legacy hosts
            # Low-end: Older servers or desktop hardware
            mips_capacity = np.random.uniform(50000, 150000)  # 50K-150K MIPS
            ram_capacity = np.random.uniform(32, 96)  # 32-96 GB
            power_idle = np.random.uniform(50, 120)  # 50-120W idle
            power_max = np.random.uniform(200, 350)  # 200-350W max
            host_type = "Low-end"
        
        host = Host(
            host_id=i,
            cpu_capacity=round(mips_capacity, 0),  # MIPS
            ram_capacity=round(ram_capacity, 1),
            power_idle=round(power_idle, 1),
            power_max=round(power_max, 1)
        )
        
        hosts.append(host)
        host_cpu_capacities.append(mips_capacity)
        host_ram_capacities.append(ram_capacity)
        host_power_idles.append(power_idle)
        host_power_maxs.append(power_max)
    
    # Create HostArray
    host_array = HostArray(
        cpu_capacities=np.array(host_cpu_capacities, dtype=np.float64),  # MIPS
        ram_capacities=np.array(host_ram_capacities, dtype=np.float64),
        power_idles=np.array(host_power_idles, dtype=np.float64),
        power_maxs=np.array(host_power_maxs, dtype=np.float64)
    )
    
    # ------------------------------------------------------------------
    # 3. Generate realistic VMs with MIPS-based capacities
    # ------------------------------------------------------------------
    vms = []
    vm_cpu_demands = []  # Now in MIPS
    vm_cpu_capacities = []  # VM's max MIPS capacity
    vm_ram_sizes = []
    vm_prices_per_sec = []
    vm_cpu_weights = []
    vm_memory_gbs = []
    
    for i in range(num_vms):
        # Randomly select VM type
        vm_type = random.choice(vm_types)
        
        # Determine VM CPU demand in MIPS (60-100% of VM MIPS capacity)
        # Real workloads typically use 60-90% of allocated capacity
        cpu_demand_mips = np.random.uniform(0.6, 0.95) * vm_type.mips_capacity
        
        # Determine CPU weight (based on application type)
        app_type = random.choice(["web", "batch", "database", "analytics", "streaming"])
        if app_type == "web" or app_type == "streaming":
            cpu_weight = np.random.uniform(0.8, 1.0)  # High priority, latency-sensitive
        elif app_type == "database":
            cpu_weight = np.random.uniform(0.6, 0.8)  # Medium priority, I/O intensive
        else:
            cpu_weight = np.random.uniform(0.4, 0.6)  # Low priority, batch processing
        
        # Convert price per hour to price per second
        price_per_sec = vm_type.price_per_hour / 3600.0
        
        # Determine memory usage (50-90% of VM RAM)
        memory_usage = np.random.uniform(0.5, 0.9) * vm_type.ram_gb
        
        vm = VM(
            vm_id=i,
            type=vm_type,
            cpu_demand=round(cpu_demand_mips, 0),  # MIPS demand
            cpu_weight=round(cpu_weight, 2),
            containers=[]  # Will be populated later
        )
        
        vms.append(vm)
        vm_cpu_demands.append(cpu_demand_mips)
        vm_cpu_capacities.append(vm_type.mips_capacity)  # Max MIPS capacity
        vm_ram_sizes.append(vm_type.ram_gb)
        vm_prices_per_sec.append(price_per_sec)
        vm_cpu_weights.append(cpu_weight)
        vm_memory_gbs.append(memory_usage)
    
    # Create VMArray
    vm_array = VMArray(
        cpu_demands=np.array(vm_cpu_demands, dtype=np.float64),  # MIPS
        cpu_capacities=np.array(vm_cpu_capacities, dtype=np.float64),  # MIPS
        ram_sizes=np.array(vm_ram_sizes, dtype=np.float64),
        prices_per_sec=np.array(vm_prices_per_sec, dtype=np.float64),
        cpu_weights=np.array(vm_cpu_weights, dtype=np.float64),
        memory_gbs=np.array(vm_memory_gbs, dtype=np.float64)
    )
    
    # ------------------------------------------------------------------
    # 4. Generate realistic containers with MIPS-based workloads
    # Container workloads in MI (Million Instructions)
    # ------------------------------------------------------------------
    containers = []
    container_workloads_mi = []  # Million Instructions
    container_state_sizes_mb = []
    container_ram_requirements_mb = []
    
    # Define realistic container types with MIPS workload ranges
    container_types = {
        "microservice": {
            "workload_range": (50, 300),      # 50-300 MI (light computation)
            "state_range": (10, 100),         # 10-100 MB
            "ram_range": (128, 512)           # 128-512 MB
        },
        "api-endpoint": {
            "workload_range": (30, 200),      # 30-200 MI (very light)
            "state_range": (5, 50),           # 5-50 MB
            "ram_range": (256, 768)           # 256-768 MB
        },
        "batch-job": {
            "workload_range": (1000, 10000),  # 1-10 GI (heavy computation)
            "state_range": (100, 1000),       # 100-1000 MB
            "ram_range": (512, 2048)          # 512-2048 MB
        },
        "database": {
            "workload_range": (200, 1500),    # 200-1500 MI (I/O intensive)
            "state_range": (500, 5000),       # 500-5000 MB
            "ram_range": (1024, 8192)         # 1-8 GB
        },
        "cache": {
            "workload_range": (20, 100),      # 20-100 MI (very light)
            "state_range": (200, 2000),       # 200-2000 MB
            "ram_range": (512, 4096)          # 512-4096 MB
        },
        "analytics": {
            "workload_range": (2000, 20000),  # 2-20 GI (very heavy)
            "state_range": (200, 2000),       # 200-2000 MB
            "ram_range": (2048, 16384)        # 2-16 GB
        },
        "stream-processor": {
            "workload_range": (100, 800),     # 100-800 MI (continuous)
            "state_range": (50, 500),         # 50-500 MB
            "ram_range": (512, 3072)          # 512-3072 MB
        }
    }
    
    # Generate container distribution (more microservices, fewer heavy jobs)
    container_type_distribution = [
        "microservice", "microservice", "microservice",  # 30% microservices
        "api-endpoint", "api-endpoint",                 # 20% API endpoints
        "batch-job",                                    # 10% batch jobs
        "database",                                     # 10% databases
        "cache",                                        # 10% caches
        "analytics",                                    # 10% analytics
        "stream-processor"                             # 10% stream processors
    ]
    
    for i in range(num_containers):
        # Select container type
        container_type = random.choice(container_type_distribution)
        type_config = container_types[container_type]
        
        # Generate values based on type
        workload_mi = np.random.uniform(*type_config["workload_range"])
        state_size_mb = np.random.uniform(*type_config["state_range"])
        ram_mb = np.random.uniform(*type_config["ram_range"])
        
        container = Container(
            container_id=i,
            workload_mi=round(workload_mi, 0),  # Million Instructions
            state_size_mb=round(state_size_mb, 0),
            ram_requirement_mb=round(ram_mb, 0)
        )
        
        containers.append(container)
        container_workloads_mi.append(workload_mi)
        container_state_sizes_mb.append(state_size_mb)
        container_ram_requirements_mb.append(ram_mb)
        
        # Assign container to a random VM (for initial assignment)
        # Distribute containers with some load balancing
        vm_idx = i % num_vms
        vms[vm_idx].containers.append(i)
    
    # Create ContainerArray
    container_array = ContainerArray(
        workloads_mi=np.array(container_workloads_mi, dtype=np.float64),  # MI
        state_sizes_mb=np.array(container_state_sizes_mb, dtype=np.float64),
        ram_requirements_mb=np.array(container_ram_requirements_mb, dtype=np.float64)
    )
    
    # ------------------------------------------------------------------
    # 5. Calculate and print dataset statistics
    # ------------------------------------------------------------------
    total_host_mips = sum(host_cpu_capacities)
    total_vm_demand_mips = sum(vm_cpu_demands)
    total_vm_capacity_mips = sum(vm_cpu_capacities)
    total_workload_mi = sum(container_workloads_mi)
    
    # Estimate makespan lower bound (if all host MIPS were perfectly utilized)
    min_possible_makespan = total_workload_mi / total_host_mips if total_host_mips > 0 else 0
    
    print("=" * 70)
    print("REALISTIC DATASET GENERATED (MIPS-BASED)")
    print("=" * 70)
    print(f"Hosts: {num_hosts}")
    print(f"  - Total MIPS capacity: {total_host_mips:,.0f} MIPS")
    print(f"  - Average per host: {total_host_mips/num_hosts:,.0f} MIPS")
    print(f"  - Total RAM: {sum(host_ram_capacities):.1f} GB")
    print(f"  - Power range: {min(host_power_idles):.1f}-{max(host_power_maxs):.1f} W")
    print()
    print(f"VMs: {num_vms}")
    print(f"  - Total MIPS demand: {total_vm_demand_mips:,.0f} MIPS")
    print(f"  - Total MIPS capacity: {total_vm_capacity_mips:,.0f} MIPS")
    print(f"  - Utilization: {(total_vm_demand_mips/total_vm_capacity_mips*100):.1f}%")
    print(f"  - Total RAM allocation: {sum(vm_ram_sizes):.1f} GB")
    print(f"  - Cost range: ${min(vm_prices_per_sec)*3600:.4f}-${max(vm_prices_per_sec)*3600:.4f}/hour")
    print()
    print(f"Containers: {num_containers}")
    print(f"  - Total workload: {total_workload_mi:,.0f} MI")
    print(f"  - Average per container: {total_workload_mi/num_containers:,.0f} MI")
    print(f"  - Total state size: {sum(container_state_sizes_mb)/1024:.1f} GB")
    print(f"  - Total RAM requirement: {sum(container_ram_requirements_mb)/1024:.1f} GB")
    print()
    print(f"Resource Ratios:")
    print(f"  - VM demand / Host capacity: {(total_vm_demand_mips/total_host_mips*100):.1f}%")
    print(f"  - VM RAM / Host RAM: {(sum(vm_ram_sizes)/sum(host_ram_capacities)*100):.1f}%")
    print(f"  - Theoretical min makespan: {min_possible_makespan:.2f} seconds")
    print("=" * 70)
    
    return hosts, vms, containers, host_array, vm_array, container_array


# -------------------------------------------------
# Example usage with MIPS
# -------------------------------------------------

def run_mips_optimization_example():
    """Run the MO-NSO optimizer with MIPS-based realistic data"""
    
    # Generate realistic dataset with MIPS
    print("Generating realistic MIPS-based cloud data center dataset...")
    hosts, vms, containers, host_array, vm_array, container_array = generate_realistic_dataset_mips(
        num_hosts=10,
        num_vms=40,
        num_containers=400,
        seed=42
    )
    
    # Create and run the optimizer
    print("\nInitializing MO-NSO optimizer with MIPS data...")
    optimizer = MO_NSO_Optimizer_Vectorized(
        hosts=host_array,
        vms=vm_array,
        containers=container_array,
        pop_size=40,
        max_iters=300,
        alpha=0.12,          # Smoother synapse updates
        beta=0.9,            # Stronger distance influence
        gamma=0.21,          # Controlled exploration noise
        gamma_f=0.20,        # Better fitness adaptation
        delta=0.12,          # Mutation strength
        eta=0.15,            # Gradual reinforcement
        seed=42
    )
    
    print("Starting optimization...")
    archive = optimizer.run()
    

    os.makedirs("results", exist_ok=True)
    archive.save_to_csv("results/mo_nso_archive.csv")
    archive.save_to_npy("results/mo_nso_archive")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS (MIPS-BASED)")
    print("=" * 60)
    
    # Get the best balanced solution
    best_solution = optimizer.analyze_pareto_front_raw(precision=3)
    
    if best_solution and best_solution.state:
        state = best_solution.state
        
        print(f"\nBest Balanced Solution Analysis:")
        print(f"Makespan: {best_solution.raw_objectives[0]:.2f} seconds")
        print(f"Energy: {best_solution.raw_objectives[1]:.2f} Joules")
        print(f"Cost: ${best_solution.raw_objectives[2]:.4f}")
        print(f"Migration time: {best_solution.raw_objectives[3]:.2f} seconds")
        
        # Calculate actual MIPS utilization
        allocations, _ = compute_cpu_allocation_vectorized(state, host_array, vm_array)
        total_allocated_mips = np.sum(allocations)
        total_host_mips = np.sum(host_array.cpu_capacities)
        total_vm_demand = np.sum(vm_array.cpu_demands)
        
        print(f"\nResource Utilization:")
        print(f"Total MIPS allocated: {total_allocated_mips:,.0f}/{total_host_mips:,.0f} MIPS")
        print(f"MIPS utilization: {(total_allocated_mips/total_host_mips*100):.1f}%")
        print(f"VM demand satisfaction: {(total_allocated_mips/total_vm_demand*100):.1f}%")
        
        # Host-level utilization
        print(f"\nHost Utilization (top 5):")
        host_utilizations = []
        for h in range(len(hosts)):
            vm_mask = state.host_vm_mask[h, :]
            vm_indices = np.where(vm_mask)[0]
            if len(vm_indices) > 0:
                allocated_mips = np.sum(allocations[vm_indices])
                host_mips = hosts[h].cpu_capacity
                mips_util = (allocated_mips / host_mips) * 100
                
                ram_used = sum(vms[vm_idx].type.ram_gb for vm_idx in vm_indices)
                ram_capacity = hosts[h].ram_capacity
                ram_util = (ram_used / ram_capacity) * 100
                
                host_utilizations.append((h, mips_util, ram_util, len(vm_indices)))
        
        # Sort by MIPS utilization (descending)
        host_utilizations.sort(key=lambda x: x[1], reverse=True)
        for h, mips_util, ram_util, num_vms in host_utilizations[:5]:
            print(f"  Host {h}: {num_vms} VMs, "
                  f"MIPS util: {mips_util:.1f}%, "
                  f"RAM util: {ram_util:.1f}%")
        
        # VM distribution analysis
        print(f"\nVM Distribution:")
        vms_per_host = np.sum(state.host_vm_mask, axis=1)
        print(f"  Max VMs per host: {int(np.max(vms_per_host))}")
        print(f"  Min VMs per host: {int(np.min(vms_per_host))}")
        print(f"  Avg VMs per host: {np.mean(vms_per_host):.1f}")
        print(f"  Hosts with VMs: {np.sum(vms_per_host > 0)}/{len(hosts)}")
        
        # Container distribution
        print(f"\nContainer Distribution:")
        containers_per_vm = np.sum(state.vm_container_mask, axis=1)
        active_vms = np.sum(containers_per_vm > 0)
        print(f"  Active VMs: {active_vms}/{len(vms)} ({active_vms/len(vms)*100:.1f}%)")
        print(f"  Max containers per VM: {int(np.max(containers_per_vm))}")
        print(f"  Avg containers per active VM: {np.mean(containers_per_vm[containers_per_vm > 0]):.1f}")
    
    
    
    print("\n" + "=" * 60)
    print("GENERATING PARETO FRONT VISUALIZATIONS")
    print("=" * 60)
    
    # Visualize all Pareto fronts
    visualize_pareto_fronts(archive, save_path="mips_pareto_fronts.png", precision=3)
    
    
    
    
    return archive, hosts, vms, containers






# -------------------------------------------------
# visualization.py
# -------------------------------------------------

def visualize_pareto_fronts(archive: VectorizedParetoArchive, 
                           save_path: str = "pareto_fronts.png",
                           precision: int = 3):
    """
    Visualize Pareto fronts for all combinations of objectives.
    
    Args:
        archive: VectorizedParetoArchive containing the solutions
        save_path: Path to save the visualization
        precision: Number of decimal places for annotations
    """
    if archive.is_empty:
        print("Archive is empty! Cannot visualize Pareto front.")
        return
    
    # Get all solutions from archive
    solutions = archive.get_all_solutions()
    
    if len(solutions) < 2:
        print(f"Not enough solutions ({len(solutions)}) for Pareto visualization.")
        return
    
    # Extract raw objectives
    raw_objectives = np.array([sol.raw_objectives for sol in solutions])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # ==================== 1. Makespan vs Cost ====================
    ax1 = plt.subplot(2, 3, 1)
    makespan = raw_objectives[:, 0]
    cost = raw_objectives[:, 2]
    
    # Find Pareto front for this 2D projection
    pareto_mask_mc = _find_2d_pareto_front(makespan, cost, minimize=True)
    
    ax1.scatter(makespan[~pareto_mask_mc], cost[~pareto_mask_mc], 
                alpha=0.6, s=50, label='Dominated', color='gray')
    ax1.scatter(makespan[pareto_mask_mc], cost[pareto_mask_mc], 
                alpha=0.8, s=80, label='Pareto Front', color='red', edgecolors='black')
    
    # Annotate extreme points
    min_makespan_idx = np.argmin(makespan)
    min_cost_idx = np.argmin(cost)
    
    ax1.annotate(f'Fastest\n{makespan[min_makespan_idx]:.{precision}f}s',
                 xy=(makespan[min_makespan_idx], cost[min_makespan_idx]),
                 xytext=(10, 10), textcoords='offset points')
    
    ax1.annotate(f'Cheapest\n${cost[min_cost_idx]:.{precision}f}',
                 xy=(makespan[min_cost_idx], cost[min_cost_idx]),
                 xytext=(10, -20), textcoords='offset points')
    
    ax1.set_xlabel('Makespan (seconds)')
    ax1.set_ylabel('Cost ($)')
    ax1.set_title('Pareto Front: Makespan vs Cost')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ==================== 2. Cost vs Migration ====================
    ax2 = plt.subplot(2, 3, 2)
    migration = raw_objectives[:, 3]
    
    # Find Pareto front for this 2D projection
    pareto_mask_cm = _find_2d_pareto_front(cost, migration, minimize=True)
    
    ax2.scatter(cost[~pareto_mask_cm], migration[~pareto_mask_cm], 
                alpha=0.6, s=50, label='Dominated', color='gray')
    ax2.scatter(cost[pareto_mask_cm], migration[pareto_mask_cm], 
                alpha=0.8, s=80, label='Pareto Front', color='blue', edgecolors='black')
    
    # Annotate extreme points
    min_migration_idx = np.argmin(migration)
    
    ax2.annotate(f'Cheapest\n${cost[min_cost_idx]:.{precision}f}',
                 xy=(cost[min_cost_idx], migration[min_cost_idx]),
                 xytext=(10, 10), textcoords='offset points')
    
    ax2.annotate(f'Min Migration\n{migration[min_migration_idx]:.{precision}f}s',
                 xy=(cost[min_migration_idx], migration[min_migration_idx]),
                 xytext=(10, -20), textcoords='offset points')
    
    ax2.set_xlabel('Cost ($)')
    ax2.set_ylabel('Migration Time (seconds)')
    ax2.set_title('Pareto Front: Cost vs Migration')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # ==================== 3. Makespan vs Energy ====================
    ax3 = plt.subplot(2, 3, 3)
    energy = raw_objectives[:, 1]
    
    # Find Pareto front for this 2D projection
    pareto_mask_me = _find_2d_pareto_front(makespan, energy, minimize=True)
    
    ax3.scatter(makespan[~pareto_mask_me], energy[~pareto_mask_me], 
                alpha=0.6, s=50, label='Dominated', color='gray')
    ax3.scatter(makespan[pareto_mask_me], energy[pareto_mask_me], 
                alpha=0.8, s=80, label='Pareto Front', color='green', edgecolors='black')
    
    # Annotate extreme points
    min_energy_idx = np.argmin(energy)
    
    ax3.annotate(f'Fastest\n{makespan[min_makespan_idx]:.{precision}f}s',
                 xy=(makespan[min_makespan_idx], energy[min_makespan_idx]),
                 xytext=(10, 10), textcoords='offset points')
    
    ax3.annotate(f'Lowest Energy\n{energy[min_energy_idx]:.{precision}f}J',
                 xy=(makespan[min_energy_idx], energy[min_energy_idx]),
                 xytext=(10, -20), textcoords='offset points')
    
    ax3.set_xlabel('Makespan (seconds)')
    ax3.set_ylabel('Energy (Joules)')
    ax3.set_title('Pareto Front: Makespan vs Energy')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # ==================== 4. Makespan vs Migration ====================
    ax4 = plt.subplot(2, 3, 4)
    
    # Find Pareto front for this 2D projection
    pareto_mask_mm = _find_2d_pareto_front(makespan, migration, minimize=True)
    
    ax4.scatter(makespan[~pareto_mask_mm], migration[~pareto_mask_mm], 
                alpha=0.6, s=50, label='Dominated', color='gray')
    ax4.scatter(makespan[pareto_mask_mm], migration[pareto_mask_mm], 
                alpha=0.8, s=80, label='Pareto Front', color='purple', edgecolors='black')
    
    ax4.annotate(f'Fastest\n{makespan[min_makespan_idx]:.{precision}f}s',
                 xy=(makespan[min_makespan_idx], migration[min_makespan_idx]),
                 xytext=(10, 10), textcoords='offset points')
    
    ax4.annotate(f'Min Migration\n{migration[min_migration_idx]:.{precision}f}s',
                 xy=(makespan[min_migration_idx], migration[min_migration_idx]),
                 xytext=(10, -20), textcoords='offset points')
    
    ax4.set_xlabel('Makespan (seconds)')
    ax4.set_ylabel('Migration Time (seconds)')
    ax4.set_title('Pareto Front: Makespan vs Migration')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # ==================== 5. 3D Pareto Analysis ====================
    ax5 = plt.subplot(2, 3, (5, 6), projection='3d')
    
    # Find 3D Pareto front (Makespan, Cost, Energy)
    pareto_mask_3d = _find_3d_pareto_front(
        makespan, cost, energy, minimize=True
    )
    
    # Plot all solutions
    ax5.scatter(makespan[~pareto_mask_3d], cost[~pareto_mask_3d], energy[~pareto_mask_3d],
                alpha=0.4, s=40, label='Dominated', color='gray')
    
    # Plot Pareto front solutions
    ax5.scatter(makespan[pareto_mask_3d], cost[pareto_mask_3d], energy[pareto_mask_3d],
                alpha=0.9, s=80, label='3D Pareto Front', color='orange', edgecolors='black')
    
    # Highlight extreme points
    ax5.scatter(makespan[min_makespan_idx], cost[min_makespan_idx], energy[min_makespan_idx],
                s=150, marker='*', color='red', label='Min Makespan')
    ax5.scatter(makespan[min_cost_idx], cost[min_cost_idx], energy[min_cost_idx],
                s=150, marker='s', color='blue', label='Min Cost')
    ax5.scatter(makespan[min_energy_idx], cost[min_energy_idx], energy[min_energy_idx],
                s=150, marker='^', color='green', label='Min Energy')
    
    ax5.set_xlabel('Makespan (s)')
    ax5.set_ylabel('Cost ($)')
    ax5.set_zlabel('Energy (J)')
    ax5.set_title('3D Pareto Front: Makespan vs Cost vs Energy')
    ax5.legend()
    
    # Add table with statistics
    stats_text = (
        f'Archive Size: {len(solutions)}\n'
        f'2D Pareto Points:\n'
        f'  Makespan-Cost: {np.sum(pareto_mask_mc)}\n'
        f'  Cost-Migration: {np.sum(pareto_mask_cm)}\n'
        f'  Makespan-Energy: {np.sum(pareto_mask_me)}\n'
        f'  Makespan-Migration: {np.sum(pareto_mask_mm)}\n'
        f'  3D Pareto Points: {np.sum(pareto_mask_3d)}\n'
        f'\nObjective Ranges:\n'
        f'  Makespan: {makespan.min():.{precision}f}-{makespan.max():.{precision}f}s\n'
        f'  Energy: {energy.min():.{precision}f}-{energy.max():.{precision}f}J\n'
        f'  Cost: ${cost.min():.{precision}f}-${cost.max():.{precision}f}\n'
        f'  Migration: {migration.min():.{precision}f}-{migration.max():.{precision}f}s'
    )
    
    # plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                # bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.suptitle('Multi-Objective Pareto Front Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Pareto fronts visualization saved to: {save_path}")
    
    # Additional analysis
    print("\n" + "="*60)
    print("ADDITIONAL PARETO FRONT ANALYSIS")
    print("="*60)
    
    # Trade-off coefficients
    print("\nTrade-off Coefficients (Spearman Correlation):")
    print(f"Makespan-Cost: {_spearman_correlation(makespan, cost):.3f}")
    print(f"Makespan-Energy: {_spearman_correlation(makespan, energy):.3f}")
    print(f"Cost-Migration: {_spearman_correlation(cost, migration):.3f}")
    print(f"Makespan-Migration: {_spearman_correlation(makespan, migration):.3f}")
    
    # Pareto front diversity
    print(f"\nPareto Front Diversity Metrics:")
    print(f"Spread (Makespan-Cost): {_calculate_spread(makespan[pareto_mask_mc], cost[pareto_mask_mc]):.3f}")
    print(f"Uniformity (Makespan-Cost): {_calculate_uniformity(makespan[pareto_mask_mc], cost[pareto_mask_mc]):.3f}")

def _find_2d_pareto_front(x: np.ndarray, y: np.ndarray, minimize: bool = True) -> np.ndarray:
    """
    Find Pareto optimal points in 2D space.
    
    Args:
        x: First objective values
        y: Second objective values
        minimize: True if minimizing both objectives
        
    Returns:
        Boolean mask indicating Pareto optimal points
    """
    n = len(x)
    pareto_mask = np.ones(n, dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            if minimize:
                # Check if point j dominates point i
                if x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                    pareto_mask[i] = False
                    break
            else:
                # For maximization
                if x[j] >= x[i] and y[j] >= y[i] and (x[j] > x[i] or y[j] > y[i]):
                    pareto_mask[i] = False
                    break
    
    return pareto_mask

def _find_3d_pareto_front(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                         minimize: bool = True) -> np.ndarray:
    """
    Find Pareto optimal points in 3D space.
    
    Args:
        x: First objective values
        y: Second objective values
        z: Third objective values
        minimize: True if minimizing all objectives
        
    Returns:
        Boolean mask indicating Pareto optimal points
    """
    n = len(x)
    pareto_mask = np.ones(n, dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            if minimize:
                # Check if point j dominates point i
                if (x[j] <= x[i] and y[j] <= y[i] and z[j] <= z[i] and 
                    (x[j] < x[i] or y[j] < y[i] or z[j] < z[i])):
                    pareto_mask[i] = False
                    break
            else:
                # For maximization
                if (x[j] >= x[i] and y[j] >= y[i] and z[j] >= z[i] and 
                    (x[j] > x[i] or y[j] > y[i] or z[j] > z[i])):
                    pareto_mask[i] = False
                    break
    
    return pareto_mask

def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Spearman rank correlation coefficient.
    """
    from scipy import stats
    return stats.spearmanr(x, y).correlation

def _calculate_spread(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate spread (diversity) metric for Pareto front.
    Lower values indicate better spread.
    """
    if len(x) <= 1:
        return float('inf')
    
    # Calculate Euclidean distances between consecutive points on the front
    points = np.column_stack([x, y])
    points = points[np.argsort(x)]  # Sort by x
    
    distances = []
    for i in range(len(points) - 1):
        dist = np.linalg.norm(points[i+1] - points[i])
        distances.append(dist)
    
    if not distances:
        return float('inf')
    
    # Spread = standard deviation of distances
    return np.std(distances)

def _calculate_uniformity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate uniformity metric for Pareto front.
    Lower values indicate more uniform distribution.
    """
    if len(x) <= 1:
        return float('inf')
    
    # Calculate Euclidean distances between consecutive points
    points = np.column_stack([x, y])
    points = points[np.argsort(x)]  # Sort by x
    
    distances = []
    for i in range(len(points) - 1):
        dist = np.linalg.norm(points[i+1] - points[i])
        distances.append(dist)
    
    if not distances:
        return float('inf')
    
    distances = np.array(distances)
    mean_dist = np.mean(distances)
    
    # Uniformity = mean absolute deviation from mean distance
    return np.mean(np.abs(distances - mean_dist))





# -------------------------------------------------
# Run the MIPS-based example
# -------------------------------------------------

if __name__ == "__main__":
    print("MO-NSO Cloud Data Center Optimizer")
    print("MIPS-Based Realistic Dataset Example\n")
    
    # Run the MIPS-based optimization
    archive, hosts, vms, containers = run_mips_optimization_example()