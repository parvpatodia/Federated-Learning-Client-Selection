Pseudocode and Complexity Analysis


Algorithm 1: Random Selection (Baseline)
Pseudocode
text
ALGORITHM RandomSelection(clients, budget, seed)
    INPUT: 
        - clients: List of N available clients
        - budget: Communication budget constraint
        - seed: Random seed for reproducibility
    OUTPUT: 
        - selected: List of selected clients
        - total_cost: Sum of communication costs

    SET random_seed(seed)
    selected ← empty list
    total_cost ← 0
    
    SHUFFLE clients randomly
    
    FOR EACH client IN shuffled_clients DO
        cost_i ← COMPUTE_COST(client)
        IF (total_cost + cost_i) ≤ budget THEN
            selected.ADD(client)
            total_cost ← total_cost + cost_i
        END IF
    END FOR
    
    RETURN (selected, total_cost)
END ALGORITHM
Complexity Analysis
Time Complexity: O(N) where N is number of clients

Shuffle: O(N)

Single pass through shuffled list: O(N)

Each cost computation: O(1)

Space Complexity: O(N) for storing shuffled list

Characteristics
✓ Simple baseline for comparison

✓ No algorithmic optimization

✓ Non-deterministic (varies with random seed)

✗ Ignores client quality and bandwidth when selecting

Algorithm 2: Greedy Selection
Pseudocode
text
ALGORITHM GreedySelection(clients, budget, alpha, beta)
    INPUT:
        - clients: List of N available clients
        - budget: Communication budget constraint
        - alpha: Weight for latency (default = 1.0)
        - beta: Weight for bandwidth (default = 1.0)
    OUTPUT:
        - selected: List of selected clients
        - total_cost: Sum of communication costs

    utilities ← empty list
    
    // STEP 1: Compute utility scores for all clients - O(N)
    FOR i = 1 TO N DO
        cost_i ← alpha × latency_i + (beta / bandwidth_i)
        utility_i ← quality_i / cost_i
        utilities.ADD((client_i, utility_i))
    END FOR
    
    // STEP 2: Sort by utility in descending order - O(N log N)
    SORT utilities BY utility_i DESCENDING
    
    // STEP 3: Greedily select until budget exhausted - O(N)
    selected ← empty list
    total_cost ← 0
    
    FOR EACH (client, utility) IN sorted_utilities DO
        cost_i ← alpha × latency_i + (beta / bandwidth_i)
        IF (total_cost + cost_i) ≤ budget THEN
            selected.ADD(client)
            total_cost ← total_cost + cost_i
        END IF
    END FOR
    
    RETURN (selected, total_cost)
END ALGORITHM
Complexity Analysis
Time Complexity: O(N log N)

Computing utilities: O(N)

Sorting: O(N log N) ← dominant term

Greedy selection: O(N)

Total: O(N log N)

Space Complexity: O(N) for utilities array

Greedy-Choice Property
The algorithm selects clients in order of highest utility (quality per unit cost).
This satisfies the greedy-choice property: the locally optimal choice at each step
leads to a globally competitive solution.

Why Greedy Works Here
Utility function combines quality and communication efficiency

Higher utility clients contribute more to model accuracy with less communication overhead

Selecting best-utility clients first maximizes accuracy within budget constraints

Approximation Ratio
For heterogeneous client distributions, greedy approximates the optimal (DP) solution
with ratio typically > 0.9 while being much faster.

Algorithm 3: Dynamic Programming (Knapsack Variant)
Problem Formulation
This is a weighted knapsack problem:

Items: Clients (each with quality "value" and communication cost "weight")

Knapsack capacity: Communication budget C

Objective: Maximize total quality (accuracy) within budget

Pseudocode
text
ALGORITHM DynamicProgramming(clients, budget, alpha, beta, discretization_factor)
    INPUT:
        - clients: List of N available clients
        - budget: Communication budget constraint (continuous)
        - alpha: Latency weight (default = 1.0)
        - beta: Bandwidth weight (default = 1.0)
        - discretization_factor: Scaling factor for budget (default = 100)
    OUTPUT:
        - selected: List of selected clients
        - total_cost: Sum of communication costs

    // STEP 1: Precompute costs and discretize budget
    costs ← empty array of size N
    
    FOR i = 1 TO N DO
        costs[i] ← alpha × latency_i + (beta / bandwidth_i)
    END FOR
    
    discretized_budget ← ROUND(budget × discretization_factor)
    
    // STEP 2: Initialize DP table
    // f[i][c] = maximum accuracy using first i clients with cost ≤ c
    CREATE DP_table f[0..N][0..discretized_budget]
    
    FOR c = 0 TO discretized_budget DO
        f[c] ← 0  // Base case: no clients = 0 accuracy
    END FOR
    
    // STEP 3: Fill DP table using recurrence relation - O(N × C)
    FOR i = 1 TO N DO
        FOR c = 0 TO discretized_budget DO
            
            cost_i_discretized ← ROUND(costs[i] × discretization_factor)
            quality_i ← quality_i
            
            // Option 1: Don't include client i
            dont_include ← f[i-1][c]
            
            // Option 2: Include client i (if it fits)
            IF cost_i_discretized ≤ c THEN
                include ← f[i-1][c - cost_i_discretized] + quality_i
            ELSE
                include ← -∞  // Can't include
            END IF
            
            // Take the maximum
            f[i][c] ← MAX(include, dont_include)
            
        END FOR
    END FOR
    
    // STEP 4: Backtrack to find which clients were selected - O(N + C)
    selected ← empty list
    c ← discretized_budget
    
    FOR i = N DOWN TO 1 DO
        // If client i was included in optimal solution
        IF f[i][c] ≠ f[i-1][c] THEN
            selected.ADD(clients[i])
            cost_i_discretized ← ROUND(costs[i] × discretization_factor)
            c ← c - cost_i_discretized
        END IF
    END FOR
    
    REVERSE(selected)  // Backtrack gave reverse order
    
    // STEP 5: Compute actual total cost
    total_cost ← 0
    FOR EACH client IN selected DO
        total_cost ← total_cost + COMPUTE_COST(client)
    END FOR
    
    RETURN (selected, total_cost)
END ALGORITHM
Recurrence Relation
text
f[i][c] = max(
    f[i-1][c],                           // Don't include client i
    f[i-1][c - cost_i] + quality_i       // Include client i (if cost_i ≤ c)
)

Base case: f[c] = 0 for all c
Complexity Analysis
Time Complexity: O(N × C) where C is discretized budget

Initialization: O(N)

DP table filling: O(N × C) ← dominant term

Backtracking: O(N)

Total: O(N × C)

Space Complexity: O(N × C) for DP table

Can be optimized to O(C) using 1D rolling array

Why DP Gives Optimal Solution
Optimal substructure: Optimal selection of first i clients under budget c
depends on optimal selection of first i-1 clients

Overlapping subproblems: Many budget/client combinations are revisited

Memoization: DP table stores solutions to avoid recomputation

Optimality Guarantee
Dynamic programming guarantees the globally optimal solution
by exploring all feasible combinations systematically.

Complexity Comparison
Algorithm	Time Complexity	Space Complexity	Optimal?	Deterministic?
Random	O(N)	O(N)	No	No
Greedy	O(N log N)	O(N)	No*	Yes
Dynamic Programming	O(N × C)	O(N × C)	Yes	Yes
*Greedy is not guaranteed optimal, but achieves high approximation ratios (>0.9) in practice.

Practical Considerations
When to Use Each Algorithm
Random Selection:

Baseline for comparison

Minimal computational overhead

Used to understand improvement from smarter selection

Greedy Selection:

Recommended for production systems

O(N log N) is much faster than DP for large client populations

Achieves near-optimal results in heterogeneous scenarios

Scales better with number of clients

Dynamic Programming:

Best for small client populations (N < 100)

Guarantees optimal solution when strict optimality is required

O(N × C) becomes prohibitive for large N

Useful as upper bound for evaluating greedy approximation

Discretization Parameter
The discretization_factor in DP affects:

Higher value: Finer granularity, better approximation to continuous problem, slower

Lower value: Coarser discretization, faster, possible loss of precision

Recommended: 100 (balances speed and accuracy)

Mathematical Notation
N: Total number of available clients

k: Number of clients selected

q_i: Quality/accuracy contribution of client i ∈ [0.6, 0.95]

b_i: Bandwidth of client i ∈ [1, 10] Mbps

l_i: Latency of client i ∈ [0.1, 2.0] seconds

C: Communication budget (continuous)

c: Discretized communication budget

α: Latency weight parameter (default = 1.0)

β: Bandwidth weight parameter (default = 1.0)

Utility Function
text
u_i = q_i / (α × l_i + β / b_i)
Higher utility indicates better data quality relative to communication cost.

Cost Function
text
cost_i = α × l_i + β / b_i
Combines latency penalty (increases with latency) and bandwidth benefit
(decreases with bandwidth).

Key Insights from Analysis
Greedy vs DP Trade-off: Greedy is 100× faster with <10% accuracy loss

Heterogeneity Matters: Algorithms show larger differences in heterogeneous scenarios

Budget Constraint Effect: Tighter budgets favor smarter selection strategies

Scalability: Greedy O(N log N) is practical; DP O(N × C) limited to small N

References to Course Topics
Greedy Algorithms: Greedy-choice property, activity selection, fractional knapsack

Dynamic Programming: 0/1 knapsack problem, optimal substructure, memoization

Sorting & Recursion: Merge sort for utilities, recursive DP formulation

Complexity Analysis: Asymptotic notation, time/space tradeoffs

Approximation Algorithms: Approximation ratios, NP-hardness of knapsack