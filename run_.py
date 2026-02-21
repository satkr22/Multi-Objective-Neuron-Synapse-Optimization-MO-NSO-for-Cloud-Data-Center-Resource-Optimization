# ===============================
# Parallelized Benchmarking Pipeline (CPU Multi-core) - MONSO VERSION FIXED
# Runs multiple algorithm configurations in parallel
# Modified to use monso.py instead of dynamic.py
# ===============================

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import csv
from multiprocessing import Pool, cpu_count
import os
import openpyxl

# Import monso.py components
from mm import Host, VM, Container, VMType, HostArray, VMArray, ContainerArray

# Import benchmark algorithms (monso version)
from benchmark_ import NSGA2, NSGA3, MOPSO, ACO, CatSwarm, FCFS
from mm import MO_NSO_Optimizer_Vectorized



# ===============================
# CSV Export Functions for Raw Objectives
# ===============================

def save_raw_objectives_to_csv(archive, algorithm_name, config_info, results_dir="raw_objectives"):
    """
    Save raw objective values to CSV for each algorithm and test case.
    
    Args:
        archive: Archive from algorithm (list of dicts or VectorizedParetoArchive)
        algorithm_name: Name of algorithm (e.g., 'MO-NSO', 'NSGA2')
        config_info: Dictionary with configuration info:
            {
                'scenario': 1 or 2,
                'num_hosts': int,
                'num_vms': int, 
                'num_containers': int,
                'run': int (1, 2, 3, 4)
            }
        results_dir: Directory to save CSV files
    """
    import os
    import csv
    from datetime import datetime
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract config info
    scenario = config_info.get('scenario', 1)
    num_hosts = config_info.get('num_hosts', 10)
    num_vms = config_info.get('num_vms', 40)
    num_containers = config_info.get('num_containers', 200)
    run_id = config_info.get('run', 1)
    
    # Create folder structure: results_dir/scenario_1/10hosts_40vms_200containers/
    scenario_dir = os.path.join(results_dir, f'scenario_{scenario}')
    config_dir = os.path.join(scenario_dir, f'{num_hosts}hosts_{num_vms}vms_{num_containers}containers')
    os.makedirs(config_dir, exist_ok=True)
    
    # Create filename
    algo_clean = algorithm_name.replace('-', '_').replace(' ', '_').upper()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{algo_clean}_run{run_id}_{timestamp}.csv"
    filepath = os.path.join(config_dir, filename)
    
    # Extract objectives based on archive type
    objectives_list = []
    
    if hasattr(archive, 'get_all_solutions'):
        # MO-NSO returns VectorizedParetoArchive
        solutions = archive.get_all_solutions()
        for sol in solutions:
            # Order in monso raw_objectives: [makespan, energy, cost, migration]
            # Reorder to: [makespan, cost, energy, migration]
            objectives_list.append([
                sol.raw_objectives[0],  # makespan
                sol.raw_objectives[2],  # cost  
                sol.raw_objectives[1],  # energy
                sol.raw_objectives[3]   # migration
            ])
    elif isinstance(archive, list) and archive:
        # Other algorithms return list of dicts
        for entry in archive:
            if 'objectives' in entry:
                objectives_list.append(entry['objectives'])
    else:
        print(f"⚠ {algorithm_name}: No valid archive data found")
        return None
    
    # Write to CSV
    if objectives_list:
        with open(filepath, 'w', newline='') as csvfile:
            # Write metadata as comment
            metadata = (f"# Algorithm: {algorithm_name}, Scenario: {scenario}, "
                       f"Hosts: {num_hosts}, VMs: {num_vms}, Containers: {num_containers}, "
                       f"Run: {run_id}, Solutions: {len(objectives_list)}")
            csvfile.write(metadata + '\n')
            
            # Write header
            writer = csv.writer(csvfile)
            writer.writerow(['solution_id', 'makespan', 'cost', 'energy', 'migration'])
            
            # Write data
            for idx, obj in enumerate(objectives_list):
                writer.writerow([idx, obj[0], obj[1], obj[2], obj[3]])
        
        print(f"✓ {algorithm_name} raw objectives saved: {filename} ({len(objectives_list)} solutions)")
        return filepath
    else:
        print(f"⚠ {algorithm_name}: No objectives to save")
        return None







# ===============================
# Data Generation Functions
# ===============================

def generate_scenario_1(num_hosts, num_vms, num_containers, seed=42, export=True):
    """Generate data for Scenario 1: Fixed Hosts & VMs, Varying Containers"""
    np.random.seed(seed)
    
    # Define VM types (similar to monso.py)
    vm_types = {
        'small': VMType('small', mips_capacity=4000, ram_gb=4, price_per_hour=0.05),
        'medium': VMType('medium', mips_capacity=8000, ram_gb=8, price_per_hour=0.10),
        'large': VMType('large', mips_capacity=12000, ram_gb=16, price_per_hour=0.20),
        'xlarge': VMType('xlarge', mips_capacity=16000, ram_gb=32, price_per_hour=0.40)
    }
    
    # Generate Hosts
    hosts = []
    for i in range(num_hosts):
        cpu_capacity = np.random.choice([50000, 60000, 70000, 80000, 90000, 100000])
        ram_capacity = np.random.choice([128, 256, 384, 512])
        power_idle = np.random.uniform(70, 140)
        power_max = power_idle + np.random.uniform(80, 180)
        
        hosts.append(Host(
            host_id=i,
            cpu_capacity=cpu_capacity,
            ram_capacity=ram_capacity,
            power_idle=power_idle,
            power_max=power_max
        ))
    
    # Generate VMs
    vms = []
    type_choices = ['small', 'medium', 'large', 'xlarge']
    type_probs = [0.45, 0.30, 0.18, 0.07]
    
    for i in range(num_vms):
        vm_type_name = np.random.choice(type_choices, p=type_probs)
        vm_type = vm_types[vm_type_name]
        
        cpu_demand = np.random.uniform(0.3, 0.9) * vm_type.mips_capacity
        
        vms.append(VM(
            vm_id=i,
            type=vm_type,
            cpu_demand=cpu_demand,
            cpu_weight=1.0,
            containers=[]
        ))
    
    # Normalize VM weights
    total_cpu = sum(vm.cpu_demand for vm in vms)
    for vm in vms:
        vm.cpu_weight = vm.cpu_demand / total_cpu
    
    # Generate Containers
    containers = []
    container_types = {
        'microservice': {'workload_range': (5000, 50000), 'state_range': (10, 200), 'ram_range': (100, 500)},
        'api-endpoint': {'workload_range': (2000, 30000), 'state_range': (5, 100), 'ram_range': (50, 300)},
        'batch-job': {'workload_range': (100000, 500000), 'state_range': (500, 2000), 'ram_range': (1000, 4000)},
        'database': {'workload_range': (50000, 200000), 'state_range': (1000, 5000), 'ram_range': (2000, 8000)},
        'cache': {'workload_range': (10000, 80000), 'state_range': (50, 500), 'ram_range': (500, 2000)},
        'analytics': {'workload_range': (80000, 300000), 'state_range': (200, 1000), 'ram_range': (1000, 3000)},
        'stream-processor': {'workload_range': (40000, 150000), 'state_range': (100, 800), 'ram_range': (500, 2000)}
    }
    
    container_type_distribution = [
        'microservice', 'microservice', 'microservice',
        'api-endpoint', 'api-endpoint',
        'batch-job', 'database', 'cache', 'analytics', 'stream-processor'
    ]
    
    for i in range(num_containers):
        container_type = np.random.choice(container_type_distribution)
        type_config = container_types[container_type]
        
        workload_mi = np.random.uniform(*type_config['workload_range'])
        state_size_mb = np.random.uniform(*type_config['state_range'])
        ram_mb = np.random.uniform(*type_config['ram_range'])
        
        containers.append(Container(
            container_id=i,
            workload_mi=round(workload_mi, 0),
            state_size_mb=round(state_size_mb, 0),
            ram_requirement_mb=round(ram_mb, 0)
        ))
    
    # Export dataset if requested
    if export:
        export_dataset_to_files(hosts, vms, containers, 
                               scenario=1, 
                               num_hosts=num_hosts, 
                               num_vms=num_vms, 
                               num_containers=num_containers, 
                               seed=seed)
    
    return hosts, vms, containers


def generate_scenario_1(num_hosts, num_vms, num_containers, seed=42, export=True):
    """Generate data for Scenario 1: Fixed Hosts & VMs, Varying Containers"""
    np.random.seed(seed)
    
    # Define VM types (similar to monso.py)
    vm_types = {
        'small': VMType('small', mips_capacity=4000, ram_gb=4, price_per_hour=0.05),
        'medium': VMType('medium', mips_capacity=8000, ram_gb=8, price_per_hour=0.10),
        'large': VMType('large', mips_capacity=12000, ram_gb=16, price_per_hour=0.20),
        'xlarge': VMType('xlarge', mips_capacity=16000, ram_gb=32, price_per_hour=0.40)
    }
    
    # Generate Hosts
    hosts = []
    for i in range(num_hosts):
        cpu_capacity = np.random.choice([50000, 60000, 70000, 80000, 90000, 100000])
        ram_capacity = np.random.choice([128, 256, 384, 512])
        power_idle = np.random.uniform(70, 140)
        power_max = power_idle + np.random.uniform(80, 180)
        
        hosts.append(Host(
            host_id=i,
            cpu_capacity=cpu_capacity,
            ram_capacity=ram_capacity,
            power_idle=power_idle,
            power_max=power_max
        ))
    
    # Generate VMs
    vms = []
    type_choices = ['small', 'medium', 'large', 'xlarge']
    type_probs = [0.45, 0.30, 0.18, 0.07]
    
    for i in range(num_vms):
        vm_type_name = np.random.choice(type_choices, p=type_probs)
        vm_type = vm_types[vm_type_name]
        
        cpu_demand = np.random.uniform(0.3, 0.9) * vm_type.mips_capacity
        
        vms.append(VM(
            vm_id=i,
            type=vm_type,
            cpu_demand=cpu_demand,
            cpu_weight=1.0,
            containers=[]
        ))
    
    # Normalize VM weights
    total_cpu = sum(vm.cpu_demand for vm in vms)
    for vm in vms:
        vm.cpu_weight = vm.cpu_demand / total_cpu
    
    # Generate Containers
    containers = []
    container_types = {
        'microservice': {'workload_range': (5000, 50000), 'state_range': (10, 200), 'ram_range': (100, 500)},
        'api-endpoint': {'workload_range': (2000, 30000), 'state_range': (5, 100), 'ram_range': (50, 300)},
        'batch-job': {'workload_range': (100000, 500000), 'state_range': (500, 2000), 'ram_range': (1000, 4000)},
        'database': {'workload_range': (50000, 200000), 'state_range': (1000, 5000), 'ram_range': (2000, 8000)},
        'cache': {'workload_range': (10000, 80000), 'state_range': (50, 500), 'ram_range': (500, 2000)},
        'analytics': {'workload_range': (80000, 300000), 'state_range': (200, 1000), 'ram_range': (1000, 3000)},
        'stream-processor': {'workload_range': (40000, 150000), 'state_range': (100, 800), 'ram_range': (500, 2000)}
    }
    
    container_type_distribution = [
        'microservice', 'microservice', 'microservice',
        'api-endpoint', 'api-endpoint',
        'batch-job', 'database', 'cache', 'analytics', 'stream-processor'
    ]
    
    for i in range(num_containers):
        container_type = np.random.choice(container_type_distribution)
        type_config = container_types[container_type]
        
        workload_mi = np.random.uniform(*type_config['workload_range'])
        state_size_mb = np.random.uniform(*type_config['state_range'])
        ram_mb = np.random.uniform(*type_config['ram_range'])
        
        containers.append(Container(
            container_id=i,
            workload_mi=round(workload_mi, 0),
            state_size_mb=round(state_size_mb, 0),
            ram_requirement_mb=round(ram_mb, 0)
        ))
    
    # Export dataset if requested
    if export:
        export_dataset_to_files(hosts, vms, containers, 
                               scenario=1, 
                               num_hosts=num_hosts, 
                               num_vms=num_vms, 
                               num_containers=num_containers, 
                               seed=seed)
    
    return hosts, vms, containers


def generate_scenario_2(num_hosts, num_vms, num_containers, seed=42, export=True):
    """Generate data for Scenario 2: Fixed Hosts & Containers, Varying VMs"""
    hosts, vms, containers = generate_scenario_1(num_hosts, num_vms, num_containers, seed, export=False)
    
    # Export dataset if requested
    if export:
        export_dataset_to_files(hosts, vms, containers,
                               scenario=2,
                               num_hosts=num_hosts,
                               num_vms=num_vms,
                               num_containers=num_containers,
                               seed=seed)
    
    return hosts, vms, containers


# ===============================
# Parallel Execution Functions
# ===============================

def run_single_algorithm_config(args):
    """Run a single algorithm on a single configuration."""
    (algo_name, algo_class, decision_dim, hosts, vms, containers, 
     num_containers, num_vms, num_hosts, scenario, pop_size, max_iters, seed) = args
    
    print(f"[Process {os.getpid()}] Starting {algo_name} - "
          f"Scenario {scenario}, Hosts={num_hosts}, VMs={num_vms}, Containers={num_containers}")
    
    start_time = time.time()
    
    try:
        if algo_name == "FCFS":
            optimizer = algo_class(decision_dim, hosts, vms, containers, seed=seed)
        elif algo_name == "MO-NSO":
            # Convert to arrays for MO-NSO
            host_array = HostArray(
                cpu_capacities=np.array([h.cpu_capacity for h in hosts], dtype=np.float64),
                ram_capacities=np.array([h.ram_capacity for h in hosts], dtype=np.float64),
                power_idles=np.array([h.power_idle for h in hosts], dtype=np.float64),
                power_maxs=np.array([h.power_max for h in hosts], dtype=np.float64)
            )
            
            vm_array = VMArray(
                cpu_demands=np.array([vm.cpu_demand for vm in vms], dtype=np.float64),
                cpu_capacities=np.array([vm.type.mips_capacity for vm in vms], dtype=np.float64),
                ram_sizes=np.array([vm.type.ram_gb for vm in vms], dtype=np.float64),
                prices_per_sec=np.array([vm.type.price_per_hour / 3600.0 for vm in vms], dtype=np.float64),
                cpu_weights=np.array([vm.cpu_weight for vm in vms], dtype=np.float64),
                memory_gbs=np.array([vm.type.ram_gb for vm in vms], dtype=np.float64)
            )
            
            container_array = ContainerArray(
                workloads_mi=np.array([c.workload_mi for c in containers], dtype=np.float64),
                state_sizes_mb=np.array([c.state_size_mb for c in containers], dtype=np.float64),
                ram_requirements_mb=np.array([c.ram_requirement_mb for c in containers], dtype=np.float64)
            )
            
            optimizer = algo_class(
                hosts=host_array,
                vms=vm_array,
                containers=container_array,
                pop_size=pop_size,
                max_iters=max_iters,
                alpha=0.12,
                beta=0.9,
                gamma=0.21,
                gamma_f=0.20,
                delta=0.12,
                eta=0.15,
                seed=seed
            )
        else:
            optimizer = algo_class(
                decision_dim, hosts, vms, containers, 
                pop_size=pop_size, max_iters=max_iters, seed=seed
            )
        
        archive = optimizer.run()
        runtime = time.time() - start_time
        
        # Extract best objectives from archive
        if archive:
            # Check if it's a VectorizedParetoArchive (MO-NSO) or list (others)
            if hasattr(archive, 'get_all_solutions'):
                # MO-NSO returns VectorizedParetoArchive
                solutions = archive.get_all_solutions()
                # Order in monso raw_objectives: [makespan, energy, cost, migration]
                # We want: [makespan, cost, energy, migration] for consistency
                objectives = [[s.raw_objectives[0], s.raw_objectives[2], 
                              s.raw_objectives[1], s.raw_objectives[3]] 
                             for s in solutions]
            else:
                # Other algorithms return list of dicts
                objectives = [entry['objectives'] for entry in archive]
            
            if objectives:
                objectives_array = np.array(objectives)
                best_makespan = np.min(objectives_array[:, 0])
                best_cost = np.min(objectives_array[:, 1])
                best_energy = np.min(objectives_array[:, 2])
                best_migration = np.min(objectives_array[:, 3])
                
                avg_makespan = np.mean(objectives_array[:, 0])
                avg_cost = np.mean(objectives_array[:, 1])
                avg_energy = np.mean(objectives_array[:, 2])
                avg_migration = np.mean(objectives_array[:, 3])
                
                archive_size = len(objectives)
            else:
                best_makespan = best_cost = best_energy = best_migration = 0
                avg_makespan = avg_cost = avg_energy = avg_migration = 0
                archive_size = 0
        else:
            best_makespan = best_cost = best_energy = best_migration = 0
            avg_makespan = avg_cost = avg_energy = avg_migration = 0
            archive_size = 0
        
        
        
        
        # ===== ADD THIS RIGHT BEFORE THE PRINT STATEMENT =====
        # Save raw objectives to CSV
        config_info = {
            'scenario': scenario,
            'num_hosts': num_hosts,
            'num_vms': num_vms,
            'num_containers': num_containers,
            'run': 1  # You can make this dynamic if running multiple repetitions
        }
        
        try:
            # Import the function (it's in this file)
            from run_ import save_raw_objectives_to_csv
            csv_path = save_raw_objectives_to_csv(archive, algo_name, config_info)
        except Exception as e:
            print(f"[Process {os.getpid()}] ⚠ Failed to save CSV for {algo_name}: {e}")
            
            
        # ===== END OF ADDED CODE =====
        
        print(f"[Process {os.getpid()}] ✓ {algo_name} completed in {runtime:.2f}s - "
              f"Archive: {archive_size}, Best Makespan: {best_makespan:.2f}")
        
        return {
            'algorithm': algo_name,
            'runtime': runtime,
            'archive_size': archive_size,
            'best_makespan': best_makespan,
            'best_cost': best_cost,
            'best_energy': best_energy,
            'best_migration': best_migration,
            'avg_makespan': avg_makespan,
            'avg_cost': avg_cost,
            'avg_energy': avg_energy,
            'avg_migration': avg_migration,
            'num_containers': num_containers,
            'num_vms': num_vms,
            'num_hosts': num_hosts,
            'scenario': scenario
        }
    
    except Exception as e:
        print(f"[Process {os.getpid()}] ✗ {algo_name} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'algorithm': algo_name,
            'runtime': 0,
            'archive_size': 0,
            'best_makespan': 0,
            'best_cost': 0,
            'best_energy': 0,
            'best_migration': 0,
            'avg_makespan': 0,
            'avg_cost': 0,
            'avg_energy': 0,
            'avg_migration': 0,
            'num_containers': num_containers,
            'num_vms': num_vms,
            'num_hosts': num_hosts,
            'scenario': scenario
        }


def benchmark_scenario_1_parallel(num_workers=None):
    """Scenario 1: Fixed Hosts=10, VMs=40, Varying Containers"""
    print("\n" + "="*80)
    print("SCENARIO 1: Fixed Hosts=10, VMs=40, Varying Containers (PARALLEL)")
    print("="*80)
    
    num_hosts = 10
    num_vms = 40
    container_counts = [200, 400, 600, 800]
    
    algorithms = {
        'MO-NSO': MO_NSO_Optimizer_Vectorized,
        'NSGA-II': NSGA2,
        'NSGA-III': NSGA3,
        'MO-PSO': MOPSO,
        'ACO': ACO,
        'Cat Swarm': CatSwarm,
        'FCFS': FCFS
    }
    
    pop_size = 30
    max_iters = 150
    
    tasks = []
    for num_containers in container_counts:
        hosts, vms, containers = generate_scenario_1(num_hosts, num_vms, num_containers, seed=42)
        decision_dim = num_vms * num_hosts + num_containers * num_vms
        
        for algo_name, algo_class in algorithms.items():
            tasks.append((
                algo_name, algo_class, decision_dim, hosts, vms, containers,
                num_containers, num_vms, num_hosts, 1, pop_size, max_iters, 42
            ))
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Running {len(tasks)} configurations across {num_workers} workers...")
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_single_algorithm_config, tasks)
    
    print("\n✓ Scenario 1 completed!")
    return results


def benchmark_scenario_2_parallel(num_workers=None):
    """Scenario 2: Fixed Hosts=10, Containers=400, Varying VMs"""
    print("\n" + "="*80)
    print("SCENARIO 2: Fixed Hosts=10, Containers=400, Varying VMs (PARALLEL)")
    print("="*80)
    
    num_hosts = 10
    num_containers = 400
    vm_counts = [20, 40, 60, 80]
    
    algorithms = {
        'MO-NSO': MO_NSO_Optimizer_Vectorized,
        'NSGA-II': NSGA2,
        'NSGA-III': NSGA3,
        'MO-PSO': MOPSO,
        'ACO': ACO,
        'Cat Swarm': CatSwarm,
        'FCFS': FCFS
    }
    
    pop_size = 30
    max_iters = 150
    
    tasks = []
    for num_vms in vm_counts:
        hosts, vms, containers = generate_scenario_2(num_hosts, num_vms, num_containers, seed=42)
        decision_dim = num_vms * num_hosts + num_containers * num_vms
        
        for algo_name, algo_class in algorithms.items():
            tasks.append((
                algo_name, algo_class, decision_dim, hosts, vms, containers,
                num_containers, num_vms, num_hosts, 2, pop_size, max_iters, 42
            ))
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Running {len(tasks)} configurations across {num_workers} workers...")
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_single_algorithm_config, tasks)
    
    print("\n✓ Scenario 2 completed!")
    return results


# ===============================
# Visualization Functions
# ===============================

def plot_scenario_1_results(results_df):
    """Plot results for Scenario 1"""
    df = results_df[results_df['scenario'] == 1].copy()
    algorithms = df['algorithm'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scenario 1: Fixed Hosts=10, VMs=40, Varying Containers', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('best_makespan', 'Makespan vs Containers', 'Makespan', axes[0, 0]),
        ('best_cost', 'Cost vs Containers', 'Cost', axes[0, 1]),
        ('best_energy', 'Energy vs Containers', 'Energy', axes[1, 0]),
        ('best_migration', 'Migration vs Containers', 'Migration', axes[1, 1])
    ]
    
    for metric, title, ylabel, ax in metrics:
        for algo in algorithms:
            algo_data = df[df['algorithm'] == algo].sort_values('num_containers')
            ax.plot(algo_data['num_containers'], algo_data[metric], 
                   marker='o', linewidth=2, markersize=8, label=algo)
        
        ax.set_xlabel('Number of Containers', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/scenario1_results_monso.png', dpi=300, bbox_inches='tight')
    print("Saved: outputs/scenario1_results_monso.png")
    plt.close()


def plot_scenario_2_results(results_df):
    """Plot results for Scenario 2"""
    df = results_df[results_df['scenario'] == 2].copy()
    algorithms = df['algorithm'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scenario 2: Fixed Hosts=10, Containers=400, Varying VMs', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('best_makespan', 'Makespan vs VMs', 'Makespan', axes[0, 0]),
        ('best_cost', 'Cost vs VMs', 'Cost', axes[0, 1]),
        ('best_energy', 'Energy vs VMs', 'Energy', axes[1, 0]),
        ('best_migration', 'Migration vs VMs', 'Migration', axes[1, 1])
    ]
    
    for metric, title, ylabel, ax in metrics:
        for algo in algorithms:
            algo_data = df[df['algorithm'] == algo].sort_values('num_vms')
            ax.plot(algo_data['num_vms'], algo_data[metric], 
                   marker='o', linewidth=2, markersize=8, label=algo)
        
        ax.set_xlabel('Number of VMs', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/scenario2_results_monso.png', dpi=300, bbox_inches='tight')
    print("Saved: outputs/scenario2_results_monso.png")
    plt.close()


def plot_runtime_comparison(results_df):
    """Plot runtime comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    df1 = results_df[results_df['scenario'] == 1].copy()
    algorithms = df1['algorithm'].unique()
    
    for algo in algorithms:
        algo_data = df1[df1['algorithm'] == algo].sort_values('num_containers')
        axes[0].plot(algo_data['num_containers'], algo_data['runtime'], 
                    marker='o', linewidth=2, markersize=8, label=algo)
    
    axes[0].set_xlabel('Number of Containers', fontsize=12)
    axes[0].set_ylabel('Runtime (seconds)', fontsize=12)
    axes[0].set_title('Scenario 1: Runtime vs Containers', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    df2 = results_df[results_df['scenario'] == 2].copy()
    
    for algo in algorithms:
        algo_data = df2[df2['algorithm'] == algo].sort_values('num_vms')
        axes[1].plot(algo_data['num_vms'], algo_data['runtime'], 
                    marker='o', linewidth=2, markersize=8, label=algo)
    
    axes[1].set_xlabel('Number of VMs', fontsize=12)
    axes[1].set_ylabel('Runtime (seconds)', fontsize=12)
    axes[1].set_title('Scenario 2: Runtime vs VMs', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/runtime_comparison_monso.png', dpi=300, bbox_inches='tight')
    print("Saved: outputs/runtime_comparison_monso.png")
    plt.close()


def generate_summary_tables(results_df):
    """Generate summary tables"""
    df1 = results_df[results_df['scenario'] == 1].copy()
    summary1 = df1.groupby('algorithm').agg({
        'best_makespan': ['mean', 'std', 'min'],
        'best_cost': ['mean', 'std', 'min'],
        'best_energy': ['mean', 'std', 'min'],
        'best_migration': ['mean', 'std', 'min'],
        'runtime': ['mean', 'std'],
        'archive_size': ['mean', 'std']
    }).round(3)
    
    summary1.to_csv('outputs/scenario1_summary_monso.csv')
    print("\nScenario 1 Summary saved")
    
    df2 = results_df[results_df['scenario'] == 2].copy()
    summary2 = df2.groupby('algorithm').agg({
        'best_makespan': ['mean', 'std', 'min'],
        'best_cost': ['mean', 'std', 'min'],
        'best_energy': ['mean', 'std', 'min'],
        'best_migration': ['mean', 'std', 'min'],
        'runtime': ['mean', 'std'],
        'archive_size': ['mean', 'std']
    }).round(3)
    
    summary2.to_csv('outputs/scenario2_summary_monso.csv')
    print("Scenario 2 Summary saved")
    
    return summary1, summary2


# ===============================
# Generate dataset
# ===============================

# ===============================
# Dataset Export Functions
# ===============================

def export_dataset_to_files(hosts, vms, containers, scenario=1, num_hosts=10, num_vms=40, num_containers=200, seed=42):
    """
    Export the generated dataset to Excel and CSV files.
    
    Args:
        hosts: List of Host objects
        vms: List of VM objects
        containers: List of Container objects
        scenario: Scenario number (1 or 2)
        num_hosts, num_vms, num_containers: Configuration parameters
        seed: Random seed used
    """
    import pandas as pd
    from datetime import datetime
    
    # Create datasets directory
    datasets_dir = Path('datasets')
    datasets_dir.mkdir(exist_ok=True)
    
    # Create scenario-specific directory
    scenario_dir = datasets_dir / f'scenario_{scenario}'
    scenario_dir.mkdir(exist_ok=True)
    
    # Create config-specific directory
    config_name = f'{num_hosts}hosts_{num_vms}vms_{num_containers}containers_seed{seed}'
    config_dir = scenario_dir / config_name
    config_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ==================== 1. Hosts Dataset ====================
    hosts_data = []
    for host in hosts:
        hosts_data.append({
            'host_id': host.host_id,
            'cpu_capacity_mips': host.cpu_capacity,
            'ram_capacity_gb': host.ram_capacity,
            'power_idle_watts': host.power_idle,
            'power_max_watts': host.power_max
        })
    
    hosts_df = pd.DataFrame(hosts_data)
    
    # Save Hosts to CSV and Excel
    hosts_csv = config_dir / f'hosts_{timestamp}.csv'
    hosts_excel = config_dir / f'hosts_{timestamp}.xlsx'
    
    hosts_df.to_csv(hosts_csv, index=False)
    hosts_df.to_excel(hosts_excel, index=False, sheet_name='Hosts')
    
    # ==================== 2. VMs Dataset ====================
    vms_data = []
    for vm in vms:
        vms_data.append({
            'vm_id': vm.vm_id,
            'vm_type': vm.type.name,
            'mips_capacity': vm.type.mips_capacity,
            'ram_gb': vm.type.ram_gb,
            'price_per_hour': vm.type.price_per_hour,
            'cpu_demand_mips': vm.cpu_demand,
            'cpu_weight': vm.cpu_weight,
            'num_containers': len(vm.containers)
        })
    
    vms_df = pd.DataFrame(vms_data)
    
    # Save VMs to CSV and Excel
    vms_csv = config_dir / f'vms_{timestamp}.csv'
    vms_excel = config_dir / f'vms_{timestamp}.xlsx'
    
    vms_df.to_csv(vms_csv, index=False)
    vms_df.to_excel(vms_excel, index=False, sheet_name='VMs')
    
    # ==================== 3. Containers Dataset ====================
    containers_data = []
    for container in containers:
        containers_data.append({
            'container_id': container.container_id,
            'workload_mi': container.workload_mi,
            'state_size_mb': container.state_size_mb,
            'ram_requirement_mb': container.ram_requirement_mb
        })
    
    containers_df = pd.DataFrame(containers_data)
    
    # Save Containers to CSV and Excel
    containers_csv = config_dir / f'containers_{timestamp}.csv'
    containers_excel = config_dir / f'containers_{timestamp}.xlsx'
    
    containers_df.to_csv(containers_csv, index=False)
    containers_df.to_excel(containers_excel, index=False, sheet_name='Containers')
    
    # ==================== 4. Summary Excel File (All Sheets) ====================
    summary_excel = config_dir / f'dataset_summary_{timestamp}.xlsx'
    
    with pd.ExcelWriter(summary_excel, engine='openpyxl') as writer:
        hosts_df.to_excel(writer, sheet_name='Hosts', index=False)
        vms_df.to_excel(writer, sheet_name='VMs', index=False)
        containers_df.to_excel(writer, sheet_name='Containers', index=False)
        
        # Add summary sheet
        summary_data = {
            'Metric': [
                'Scenario',
                'Number of Hosts',
                'Number of VMs', 
                'Number of Containers',
                'Random Seed',
                'Total Host MIPS Capacity',
                'Total Host RAM (GB)',
                'Total VM MIPS Demand',
                'Total VM RAM (GB)',
                'Total Container Workload (MI)',
                'Total Container State (MB)',
                'Total Container RAM (MB)',
                'Generation Timestamp'
            ],
            'Value': [
                scenario,
                num_hosts,
                num_vms,
                num_containers,
                seed,
                f"{sum(h.cpu_capacity for h in hosts):,.0f}",
                f"{sum(h.ram_capacity for h in hosts):.1f}",
                f"{sum(vm.cpu_demand for vm in vms):,.0f}",
                f"{sum(vm.type.ram_gb for vm in vms):.1f}",
                f"{sum(c.workload_mi for c in containers):,.0f}",
                f"{sum(c.state_size_mb for c in containers):,.0f}",
                f"{sum(c.ram_requirement_mb for c in containers):,.0f}",
                timestamp
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Add VM type distribution
        vm_type_counts = {}
        for vm in vms:
            vm_type_counts[vm.type.name] = vm_type_counts.get(vm.type.name, 0) + 1
        
        type_distribution = pd.DataFrame([
            {'VM Type': k, 'Count': v, 'Percentage': f'{(v/num_vms*100):.1f}%'} 
            for k, v in vm_type_counts.items()
        ])
        type_distribution.to_excel(writer, sheet_name='VM_Type_Distribution', index=False)
    
    print(f"\n✓ Dataset exported successfully!")
    print(f"  Location: {config_dir}")
    print(f"  Files created:")
    print(f"    - Hosts: hosts_{timestamp}.csv / .xlsx")
    print(f"    - VMs: vms_{timestamp}.csv / .xlsx")
    print(f"    - Containers: containers_{timestamp}.csv / .xlsx")
    print(f"    - Complete summary: dataset_summary_{timestamp}.xlsx")
    
    return {
        'hosts_csv': hosts_csv,
        'hosts_excel': hosts_excel,
        'vms_csv': vms_csv,
        'vms_excel': vms_excel,
        'containers_csv': containers_csv,
        'containers_excel': containers_excel,
        'summary_excel': summary_excel,
        'directory': config_dir
    }






# ===============================
# Main Execution
# ===============================

def main():
    """Main benchmarking pipeline"""
    print("="*80)
    print("MULTI-ALGORITHM BENCHMARKING (MONSO VERSION)")
    print("="*80)
    print(f"\nCPU cores: {cpu_count()}, Using: {max(1, cpu_count() - 1)} workers")
    print("="*80)
    
    Path('outputs').mkdir(parents=True, exist_ok=True)
    
    overall_start = time.time()
    
     # ===== EXPORT DATASETS FIRST =====
    print("\n" + "="*80)
    print("STEP 1: GENERATING AND EXPORTING DATASETS")
    print("="*80)
    
    # Export Scenario 1 datasets
    print("\nScenario 1: Fixed Hosts=10, VMs=40, Varying Containers")
    container_counts = [200, 400, 600, 800]
    for num_containers in container_counts:
        print(f"\n  Generating dataset with {num_containers} containers...")
        hosts, vms, containers = generate_scenario_1(
            num_hosts=10, 
            num_vms=40, 
            num_containers=num_containers, 
            seed=42,
            export=True
        )
    
    # Export Scenario 2 datasets
    print("\nScenario 2: Fixed Hosts=10, Containers=400, Varying VMs")
    vm_counts = [20, 40, 60, 80]
    for num_vms in vm_counts:
        print(f"\n  Generating dataset with {num_vms} VMs...")
        hosts, vms, containers = generate_scenario_2(
            num_hosts=10, 
            num_vms=num_vms, 
            num_containers=400, 
            seed=42,
            export=True
        )
    
    # ===== RUN BENCHMARKS =====
    print("\n" + "="*80)
    print("STEP 2: RUNNING BENCHMARKS")
    print("="*80)
    

    
    results_scenario1 = benchmark_scenario_1_parallel()
    results_scenario2 = benchmark_scenario_2_parallel()
    
    overall_time = time.time() - overall_start
    
    all_results = results_scenario1 + results_scenario2
    results_df = pd.DataFrame(all_results)
    
    results_df.to_csv('outputs/detailed_results_monso.csv', index=False)
    print("\nSaved: outputs/detailed_results_monso.csv")
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_scenario_1_results(results_df)
    plot_scenario_2_results(results_df)
    plot_runtime_comparison(results_df)
    
    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80)
    
    generate_summary_tables(results_df)
    
    print("\n" + "="*80)
    print("BENCHMARKING COMPLETE!")
    print("="*80)
    print(f"\nTotal time: {overall_time:.2f}s ({overall_time/60:.2f} min)")
    print("\n" + "="*80)
    print("\n📁 Datasets saved in: ./datasets/")
    print("📁 Results saved in: ./outputs/")
    print("="*80)
    


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()