# MO-NSO: Multi-Objective Optimization for Cloud Data Center Scheduling

This project implements a **Multi-Objective Neuron Synapse Optimization (MO-NSO)** framework for solving cloud data center scheduling problems.

The goal is to optimize VM–Host and Container–VM placement while simultaneously minimizing:

- Makespan (execution time)
- Energy consumption
- Monetary cost
- Migration overhead

Instead of combining these into a single weighted value, this system preserves multiple trade-off solutions using Pareto dominance.

---

## Project Motivation

Cloud data centers operate under conflicting objectives.

- Reducing makespan may increase energy usage.
- Minimizing cost may reduce performance.
- Avoiding migration may hurt flexibility and load balancing.

Most traditional systems collapse objectives into a single scalar score, hiding real trade-offs.

This project models the scheduling problem as a **true multi-objective optimization problem** and returns a set of non-dominated solutions (Pareto front).

---

## Core Components

### 1. MO-NSO Implementation (`mm.py`)

This is the main optimizer.

Key features:

- Continuous neuron-based solution representation
- Capacity-aware decoding (VM → Host and Container → VM)
- Vectorized CPU allocation model
- Makespan calculation using workload/allocated CPU
- Energy model based on utilization-dependent power
- Cost model with:
  - Billing quantum
  - SLA penalty
  - Idle VM penalty
  - Starvation (under-allocation) penalty
- Migration time modeling for VM memory and container states
- Pareto archive maintenance

The implementation is vectorized using NumPy for scalability.

---

### 2. Benchmark Algorithms (`benchmark_.py`)

For comparison, the following algorithms are implemented:

- NSGA-II  
- NSGA-III  
- MOPSO  
- ACO  
- Cat Swarm Optimization  
- FCFS (baseline heuristic)

All algorithms are evaluated on the exact same generated scenarios.

---

### 3. Benchmark Execution (`run_.py`)

Responsible for:

- Generating synthetic cloud scenarios
- Running algorithms in parallel (multi-core CPU)
- Collecting results
- Saving raw objective values to CSV
- Organizing results by scenario and configuration



---

### 4. Visualization (`new_plot.py`)

Generates:

- 2D Pareto plots:
  - Makespan vs Cost
  - Cost vs Migration
  - Makespan vs Energy
  - Makespan vs Migration
- 3D Pareto plots (all objective combinations)

Used for analyzing trade-offs and solution diversity.

---

## System Model

The simulation models three entities:

### Hosts
- CPU capacity (MIPS)
- RAM capacity (GB)
- Idle power
- Maximum power

### Virtual Machines (VMs)
- Type-based CPU capacity
- RAM size
- Price per hour
- CPU demand
- CPU weight (for proportional allocation)

### Containers
- Workload (Million Instructions)
- RAM requirement
- Migration state size (MB)

---

## Objective Formulation (Summary)

### Makespan
Maximum execution time across all VMs

---

### Energy

Host power increases non-linearly with utilization.

---

### Cost

Includes:

- Active VM billing
- Idle VM penalty
- Billing time rounding
- SLA penalty for delayed containers
- Starvation penalty for under-allocated VMs

---

### Migration

Based on:

- VM memory moved
- Container state size transferred
- Network bandwidth

---

## Scenarios Evaluated

### Scenario 1: Workload Scalability

- Fixed Hosts
- Fixed VMs
- Increasing number of containers

Used to test stability under heavier workload.

### Scenario 2: Infrastructure Scaling

- Fixed containers
- Increasing VMs

Used to evaluate behavior under changing resource availability.

---

## How to Run

### Install Dependencies
```bash
pip install numpy matplotlib pandas openpyxl scipy

python run_.py
