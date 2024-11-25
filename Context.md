# Rhetorical Scenario: Logistics Network for a Distribution Hub
Imagine a logistics network that delivers packages from a central distribution hub (S) to a variety of locations in a city via a set of intermediate warehouses and transport routes. The goal is to determine the maximum amount of packages that can be delivered from the central hub (source) to the final destination (sink) without exceeding the capacity of any transport route.

## Network Description
Source Node (S): The central distribution hub, which can handle a total of 120 packages per day. This is the starting point where packages are dispatched.

##Intermediate Nodes:

- Node A: A warehouse that handles packages for three different areas. It has outgoing transport routes to multiple nodes.
- Node B: A logistics center that connects to various distribution routes but does not directly serve customers.
- Node C: A smaller regional warehouse that stores goods and sends them to nearby districts.
- Node D: A distribution node that funnels packages to the final destination.
- Node E: Another regional warehouse that works in parallel with the others but is part of a separate distribution chain.
- Sink Node (T): The final destination, representing the collection point for all delivered packages. This is the node where all the packages are received after being transported through the network.

## Capacities (Units of Packages per Day)
- S → A: Capacity = 50 packages
- S → B: Capacity = 40 packages
- S → C: Capacity = 30 packages
- A → D: Capacity = 40 packages
- B → D: Capacity = 50 packages
- C → D: Capacity = 20 packages
- C → E: Capacity = 20 packages
- E → T: Capacity = 40 packages
- B → E: Capacity = 30 packages
- D → T: Capacity = 60 packages
- A → B: Capacity = 20 packages
- D → E: Capacity = 30 packages

## Problem:
    Determine the maximum number of packages (in units) that can be delivered from the central distribution hub S to the destination T, considering the capacity constraints on each route. The goal is to calculate the maximum flow through this network from S to T.


## Network Flow Details:
- The source node (S) can send 50 packages to A, 40 packages to B, and 30 packages to C.
- From A, 40 packages can be sent to D, and there is an internal pipeline between A and B with a capacity of 20 packages.
- From B, 50 packages can go to D, and 30 packages can go to E.
-  From C, 20 packages can be sent to D, and 20 packages can be sent to E.
- D can send up to 60 packages to T, and 30 packages to E.
- E can send 40 packages to T.

## Solution Overview:

### Algorithm Overview:
Uses genetic algorithms with some unique modifications
### Key features:
- Dominance-based vertex selection
- Variable mutation rate
- Energy level calculations for vertices
- Balance and flow capacity constraints

### Key Components:
- Fitness Function combines:
    - Ratio of balanced vertices
    - Penalty for unbalanced flow
    - Degree of saturation
- Crossover Process:
    - Uses dominance based on vertex energy levels
    - Transfers complete vertices rather than random crossover
- Mutation:
    - Variable mutation rate based on vertex balance
    - Rate = x + sqrt(|excess flow|/total flow)
    - x is typically around 0.02

### Experimental Results:
- Tested on two graphs:
    - Graph 1: 25 vertices, 49 edges, max flow = 90
    - Graph 2: 25 vertices, 56 edges, max flow = 91
- Performance affected by parameters:
    - Balancing factor k
    - Mutation rate
    - Truncation rate
- Complexity:
    - Sequential: O(n³p) where n is vertices, p is population size
    - Parallel: O(n³)
The paper concludes that while the algorithm successfully finds optimal or near-optimal solutions, its complexity is not as competitive as conventional approaches, suggesting room for further optimization.

## Constraints:
- The capacity of each route is the maximum number of packages that can be transported through it per day.
- The total flow from the source S to the sink T must be maximized while respecting the capacity constraints on all routes.
- The flow on any route must not exceed its capacity.
- The flow entering a node must equal the flow leaving that node, except for the source and sink nodes.
