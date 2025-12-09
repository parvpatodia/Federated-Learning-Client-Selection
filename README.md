# Federated Learning Client Selection Algorithms

A comprehensive comparison of client selection algorithms for efficient federated learning.

## Overview

This project implements and evaluates three client selection strategies for federated learning:

1. **Random Selection** - Baseline approach, selects clients randomly
2. **Greedy Selection** - O(N log N) algorithm using utility-based ranking
3. **Dynamic Programming** - O(N × C) optimal solution using knapsack-style DP

## Key Results

| Scenario | Algorithm | Final Accuracy | Communication Cost | Convergence (90%) |
|----------|-----------|-----------------|-------------------|-------------------|
| **Homogeneous** | Random | 0.7700 | 28.36 | 20 rounds |
| | Greedy | 0.7700 | 28.36 | 20 rounds |
| | DP | 0.7700 | 29.55 | 20 rounds |
| **Moderately Heterogeneous** | Random | 0.7694 | 32.86 | 20 rounds |
| | Greedy | 0.7588 | 32.50 | 20 rounds |
| | DP | 0.7663 | 33.07 | 20 rounds |
| **Highly Heterogeneous** | Random | 0.7832 | 38.10 | 20 rounds |
| | Greedy | **0.8136** | 38.15 | 20 rounds |
| | DP | **0.8162** | 38.45 | 20 rounds |

## Key Finding

**Greedy selection significantly outperforms random selection in heterogeneous environments** (3.9% accuracy improvement) while maintaining nearly identical communication costs. DP provides marginal gains over greedy (0.3%) at much higher computational cost.

## Project Structure

Federated-Learning-Client-Selection/
├── src/
│ ├── data_generator.py # Synthetic client generation
│ ├── algorithms.py # Random, Greedy, and DP implementations
│ ├── metrics.py # Performance metrics calculation
│ ├── simulator.py # Federated learning simulator
│ └── visualization.py # Result visualization utilities
├── experiments/
│ └── run_experiments.py # Main experiment orchestrator
├── pseudocode/
│ └── algorithms_pseudocode.md # Detailed pseudocode and complexity analysis
├── results/
│ ├── *.png # Visualization plots
│ ├── experiment_report.txt # Detailed results report
│ └── results.json # Raw experimental data
├── requirements.txt
├── README.md
└── .gitignore


## Installation

### Prerequisites
- Python 3.8+
- macOS, Linux, or Windows

### Setup

Clone repository
git clone https://github.com/parvpatodia/Federated-Learning-Client-Selection.git
cd Federated-Learning-Client-Selection

Create virtual environment
python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Run all experiments (takes ~2 minutes)
python experiments/run_experiments.py

Check results
cat results/experiment_report.txt
ls results/


This generates:
- 9 visualization plots (accuracy, communication cost, summaries for 3 scenarios)
- Detailed text report with statistics
- Raw results in JSON format

## Algorithm Complexity

| Algorithm | Time | Space | Optimal | Notes |
|-----------|------|-------|---------|-------|
| Random | O(N) | O(N) | No | Baseline for comparison |
| Greedy | O(N log N) | O(N) | No* | **Recommended for production** |
| DP | O(N × C) | O(N × C) | Yes | Upper bound; limited scalability |

*Greedy achieves >0.9 approximation ratio with 100× speedup over DP.

## Algorithms

### Random Selection
Selects clients randomly until budget exhausted. Serves as baseline.

**Complexity:** O(N)

### Greedy Selection
1. Compute utility score: u_i = quality_i / (α·latency_i + β/bandwidth_i)
2. Sort clients by utility descending
3. Select clients greedily until budget exhausted

**Complexity:** O(N log N)

**Why it works:** Selects clients with highest quality-to-cost ratio, balancing accuracy and communication efficiency.

### Dynamic Programming
Formulates client selection as a weighted 0/1 knapsack problem:
- Items: Clients with quality values and communication costs
- Capacity: Communication budget
- Objective: Maximize total quality within budget

Uses DP recurrence: f[i][c] = max(f[i-1][c], f[i-1][c-cost_i] + quality_i)

**Complexity:** O(N × C) where C is discretized budget

**Why it's optimal:** Explores all feasible combinations using dynamic programming with memoization.

## Scenarios

### 1. Homogeneous
All clients have similar properties:
- Data size: ~1250 samples
- Bandwidth: ~5.5 Mbps
- Quality: ~0.77
- Latency: ~1.0 seconds

**Finding:** All algorithms perform identically when clients are uniform.

### 2. Moderately Heterogeneous
Realistic differences in client properties:
- Data size: [500, 2000] uniform distribution
- Bandwidth: [1, 10] uniform distribution
- Quality: [0.6, 0.95] uniform distribution
- Latency: [0.1, 2.0] uniform distribution

**Finding:** DP slightly outperforms greedy, but differences are minimal.

### 3. Highly Heterogeneous
Extreme clustering (40% good clients, 60% poor clients):
- Good clients: high quality, high bandwidth, low latency
- Poor clients: low quality, low bandwidth, high latency

**Finding:** Greedy and DP show significant improvements over random (3-4% accuracy).

## Methodology

### Metrics
- **Global Accuracy:** Model accuracy after 20 training rounds
- **Communication Cost:** Sum of latency and bandwidth penalties per round
- **Convergence Speed:** Rounds needed to reach 90% accuracy

### Experimental Setup
- 50 clients per scenario
- 10 independent runs per algorithm-scenario pair
- 20 training rounds per run
- Reproducible using fixed random seeds

### Cost Function
cost_i = α × latency_i + β / bandwidth_i

- α = 1.0 (latency weight)
- β = 1.0 (bandwidth weight)

### Utility Function
utility_i = quality_i / (α × latency_i + β / bandwidth_i)

Higher utility = better quality relative to communication cost.

## Results Interpretation

### Homogeneous Scenario
All algorithms select identical clients → identical results. Demonstrates that algorithmic sophistication only matters when clients differ.

### Moderately Heterogeneous Scenario
- Greedy shows slight accuracy degradation (-1.4% vs DP)
- Communication costs nearly identical
- Greedy 100× faster than DP

**Conclusion:** For modest heterogeneity, greedy's speed advantage justifies small accuracy trade-off.

### Highly Heterogeneous Scenario
- **Greedy outperforms Random:** +3.9% accuracy (0.8136 vs 0.7832)
- **DP marginal over Greedy:** +0.3% accuracy (0.8162 vs 0.8136)
- **Communication costs similar:** All within 1% of budget

**Conclusion:** In heterogeneous environments, smart selection dramatically improves accuracy. Greedy provides best trade-off.

## Key Insights

1. **Heterogeneity Drives Need for Smart Selection**
   - Homogeneous clients: Any algorithm works
   - Heterogeneous clients: Quality-based selection critical

2. **Greedy-DP Trade-off Sweet Spot**
   - Greedy: 100× faster, 99.7% of DP's accuracy
   - DP: Optimal but prohibitively slow for large N

3. **Communication Budget Constraint**
   - Tight budgets (divisor=2): Force selective client choice
   - Loose budgets: Reduce importance of algorithm choice

4. **Scalability Implications**
   - Greedy O(N log N): Practical for N=10,000+
   - DP O(N × C): Limited to N<100 in practice

## Course Connections

This project demonstrates algorithms from CS 5800:

- **Greedy Algorithms:** Greedy-choice property, optimal substructure
- **Dynamic Programming:** 0/1 knapsack, optimal substructure, overlapping subproblems
- **Sorting:** O(N log N) utility-based sorting in greedy
- **Complexity Analysis:** Time/space trade-offs between algorithms
- **Approximation Algorithms:** Greedy approximation ratio vs optimal DP

## Files

- `src/data_generator.py` - Generates synthetic heterogeneous clients
- `src/algorithms.py` - Implements Random, Greedy, and DP algorithms
- `src/metrics.py` - Calculates utility, cost, accuracy metrics
- `src/simulator.py` - Runs federated learning training loops
- `src/visualization.py` - Creates result plots
- `experiments/run_experiments.py` - Orchestrates all experiments
- `pseudocode/algorithms_pseudocode.md` - Detailed pseudocode and analysis
- `results/` - Generated plots, reports, and raw data

## Future Work

1. **Real-world validation:** Test on actual federated learning datasets
2. **Adaptive selection:** Dynamically adjust algorithm based on heterogeneity
3. **Privacy preservation:** Add differential privacy guarantees
4. **Communication efficiency:** Implement compression and quantization
5. **Personalization:** Client-specific model updates based on device capability

## Authors

- **Nida Esen** - Algorithm theory, greedy implementation, presentation
- **Parv Patodia** - Simulation environment, DP implementation, visualizations

## References

- Algorithm textbook chapters on greedy algorithms and dynamic programming
- Federated learning fundamentals from course materials
- Knapsack problem theory and applications

---

**Final Project for CS 5800: Design of Algorithms**  
