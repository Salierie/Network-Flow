import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

class LogisticsNetwork:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.setup_network()
        
    def setup_network(self):
        nodes = ['S', 'A', 'B', 'C', 'D', 'E', 'T']
        self.graph.add_nodes_from(nodes)
        
        edges_with_capacity = [
            ('S', 'A', 50), ('S', 'B', 40), ('S', 'C', 30),
            ('A', 'D', 40), ('B', 'D', 50), ('C', 'D', 20),
            ('C', 'E', 20), ('E', 'T', 40), ('B', 'E', 30),
            ('D', 'T', 60), ('A', 'B', 20), ('D', 'E', 30)
        ]
        
        for source, target, capacity in edges_with_capacity:
            self.graph.add_edge(source, target, capacity=capacity, flow=0)

    def visualize_network(self, flows=None):
        """Visualize the network with current flows and capacities"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                             node_size=500)
        nx.draw_networkx_labels(self.graph, pos)
        
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            flow = data['flow'] if flows is None else flows.get((u, v), 0)
            edge_labels[(u, v)] = f"{flow}/{data['capacity']}"
        
        nx.draw_networkx_edges(self.graph, pos)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels)
        
        plt.title("Logistics Network Flow")
        plt.axis('off')
        plt.show()

class Individual:
    def __init__(self, network: nx.DiGraph):
        self.network = network.copy()
        self.fitness = 0
        if hasattr(network, 'initial_zero') and network.initial_zero:
            self.initialize_zero_flow()
        else:
            self.initialize_random_flow()
            
    def initialize_zero_flow(self):
        """Initialize all edges with zero flow"""
        for _, _, data in self.network.edges(data=True):
            data['flow'] = 0
            
    def initialize_random_flow(self):
        """Initialize random valid flows for all edges"""
        def dfs_random_flow(node):
            if node == 'T':
                return
            
            outgoing = list(self.network.successors(node))
            if not outgoing:
                return
                
            for next_node in outgoing:
                capacity = self.network[node][next_node]['capacity']
                if capacity > 0:
                    self.network[node][next_node]['flow'] = random.randint(0, capacity)
                    dfs_random_flow(next_node)
                    
        dfs_random_flow('S')
        
    def calculate_vertex_energy(self, vertex: str) -> float:
        if vertex in ['S', 'T']:  
            return 0
            
        incoming_flow = sum(self.network[u][vertex]['flow'] 
                          for u in self.network.predecessors(vertex))
        outgoing_flow = sum(self.network[vertex][v]['flow'] 
                          for v in self.network.successors(vertex))
        
        excess_flow = abs(incoming_flow - outgoing_flow)
        
        max_incoming = sum(self.network[u][vertex]['capacity'] 
                         for u in self.network.predecessors(vertex))
        max_outgoing = sum(self.network[vertex][v]['capacity'] 
                         for v in self.network.successors(vertex))
        max_possible = min(max_incoming, max_outgoing)
        
        actual_flow = min(incoming_flow, outgoing_flow)
        
        k = 1.4
        energy = k * excess_flow + abs(max_possible - actual_flow)
        return energy

def individual_fitness_key(individual):
    return individual.fitness

class GeneticAlgorithm:
    def __init__(self, network: LogisticsNetwork, population_size: int = 50,
                 mutation_rate: float = 0.02, balancing_factor: float = 1.4,
                 truncation_rate: float = 0.8):
        self.network = network
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.mutation_rate = mutation_rate
        self.balancing_factor = balancing_factor
        self.truncation_rate = truncation_rate
        self.history = {'fitness': [], 'max_flow': []}
        self.best_solution = None
        
    def initialize_population(self):
        """Initialize the population with random individuals"""
        self.population = [
            Individual(self.network.graph) 
            for _ in range(self.population_size - 1)
        ]
        
        for individual in self.population:
            individual.fitness = self.calculate_fitness(individual)
            
        self.best_solution = max(self.population, key=individual_fitness_key)

    def calculate_fitness(self, individual: Individual) -> float:
        """Calculate fitness for an individual"""
        sink_flow = sum(individual.network[u]['T']['flow'] 
                       for u in individual.network.predecessors('T'))
        
        balance_penalty = sum(individual.calculate_vertex_energy(v) 
                            for v in individual.network.nodes())
        
        total_flow = sum(data['flow'] for _, _, data in individual.network.edges(data=True))
        total_capacity = sum(data['capacity'] for _, _, data in individual.network.edges(data=True))
        utilization = total_flow / total_capacity if total_capacity > 0 else 0
        
        fitness = (sink_flow * 2) - balance_penalty + utilization
        return max(0, fitness) 

    def display_stats(self):
        """Display current statistics of the genetic algorithm"""
        clear_output(wait=True)
        
        print(f"Generation: {self.generation}")
        print(f"Population Size: {self.population_size}")
        print(f"Mutation Rate: {self.mutation_rate}")
        print(f"Balancing Factor: {self.balancing_factor}")
        print(f"Truncation Rate: {self.truncation_rate}")
        
        if self.best_solution:
            print("\nBest Solution Stats:")
            print(f"Fitness: {self.best_solution.fitness:.2f}")
            
            sink_flow = sum(self.best_solution.network[u]['T']['flow'] 
                          for u in self.best_solution.network.predecessors('T'))
            print(f"Total Flow to Sink: {sink_flow}")
            
            print("\nVertex Energy Levels:")
            for vertex in self.best_solution.network.nodes():
                if vertex not in ['S', 'T']:
                    energy = self.best_solution.calculate_vertex_energy(vertex)
                    print(f"Node {vertex}: {energy:.2f}")
        
        if len(self.history['fitness']) > 0:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(self.history['fitness'])
            plt.title('Best Fitness Over Generations')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            
            plt.subplot(1, 2, 2)
            plt.plot(self.history['max_flow'])
            plt.title('Maximum Flow Over Generations')
            plt.xlabel('Generation')
            plt.ylabel('Flow')
            plt.tight_layout()
            plt.show()
            
        if self.best_solution:
            self.network.visualize_network(
                {(u, v): d['flow'] 
                 for u, v, d in self.best_solution.network.edges(data=True)}
            )

    def select_parent(self):
        """Select parent using tournament selection"""
        tournament_size = min(3, len(self.population))  
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=individual_fitness_key)
        
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Create child solution using vertex-based crossover"""
        child = Individual(self.network.graph)
        
        for vertex in child.network.nodes():
            if vertex in ['S', 'T']:
                continue
                
            energy1 = parent1.calculate_vertex_energy(vertex)
            energy2 = parent2.calculate_vertex_energy(vertex)
            
            dominant_parent = parent1 if energy1 < energy2 else parent2
            
            for u, v, data in dominant_parent.network.edges(data=True):
                if u == vertex or v == vertex:
                    if child.network.has_edge(u, v):
                        current_flow = child.network[u][v]['flow']
                        new_flow = (current_flow + data['flow']) // 2
                        child.network[u][v]['flow'] = min(new_flow, 
                                                        child.network[u][v]['capacity'])
                    else:
                        child.network[u][v]['flow'] = data['flow']
                        
        return child
        
    def mutate(self, individual: Individual):
        """Apply mutation to individual"""
        for u, v, data in individual.network.edges(data=True):
            if random.random() < self.mutation_rate:
                if data['capacity'] > 0:
                    if random.random() < 0.5:
                        data['flow'] = max(0, data['flow'] - random.randint(1, 3))
                    else:
                        data['flow'] = min(data['capacity'], 
                                         data['flow'] + random.randint(1, 3))
                    
    def evolve_population(self):
        """Perform one generation of evolution"""
        new_population = []
        
        self.population.sort(key=individual_fitness_key, reverse=True)
        
        elite_size = int(self.population_size * 0.1)  # Keep top 10%
        new_population.extend(self.population[:elite_size])
        
        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            child = self.crossover(parent1, parent2)
            
            self.mutate(child)
            
            child.fitness = self.calculate_fitness(child)
            
            new_population.append(child)
            
        self.population = new_population
        
        current_best = max(self.population, key=individual_fitness_key)
        if not self.best_solution or current_best.fitness > self.best_solution.fitness:
            self.best_solution = Individual(self.network.graph)
            for u, v, data in current_best.network.edges(data=True):
                self.best_solution.network[u][v]['flow'] = data['flow']
            self.best_solution.fitness = current_best.fitness
            
        self.history['fitness'].append(current_best.fitness)
        sink_flow = sum(current_best.network[u]['T']['flow'] 
                       for u in current_best.network.predecessors('T'))
        self.history['max_flow'].append(sink_flow)

class NetworkFlowGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Network Flow Optimization")
        
        self.root.geometry("1600x900")  
        
        self.left_frame = ttk.Frame(root, padding="10")
        self.right_frame = ttk.Frame(root, padding="10")
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        
        root.grid_columnconfigure(0, weight=4)  
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)
        
        self.network = LogisticsNetwork()
        self.ga = GeneticAlgorithm(
            network=self.network,
            population_size=50,
            mutation_rate=0.02,
            balancing_factor=1.4,
            truncation_rate=0.8
        )
        
        self.pos = nx.spring_layout(self.network.graph, seed=42)  
        
        self.ga.initialize_population()
        
        self.current_individual = self.ga.population[0] if self.ga.population else None
        
        self.setup_visualization()
        self.setup_controls()
        self.setup_stats()
        
    def setup_visualization(self):
        """Setup the network visualization"""
        self.fig = Figure(figsize=(12, 8))  
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.pos = {
            'S': np.array([-2, 0]),      
            'A': np.array([-0.7, 1.2]),  
            'B': np.array([-0.7, 0]),    
            'C': np.array([-0.7, -1.2]), 
            'D': np.array([0.7, 0.8]),   
            'E': np.array([0.7, -0.8]),  
            'T': np.array([2, 0])        
        }
        
        self.update_network_plot()
        
    def setup_controls(self):
        """Setup control buttons and parameters"""
        control_frame = ttk.LabelFrame(self.right_frame, text="Controls", padding="5")
        control_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start", command=self.toggle_simulation)
        self.start_button.pack(pady=5)
        
        param_frame = ttk.LabelFrame(self.right_frame, text="Parameters", padding="5")
        param_frame.pack(fill=tk.X, pady=5)
        
        self.param_vars = {
            "Population Size": tk.StringVar(value="50"),
            "Mutation Rate": tk.StringVar(value="0.02"),
            "Balancing Factor": tk.StringVar(value="1.4"),
            "Truncation Rate": tk.StringVar(value="0.8")
        }
        
        for i, (param, var) in enumerate(self.param_vars.items()):
            ttk.Label(param_frame, text=param).grid(row=i, column=0, padx=5, pady=2)
            ttk.Entry(param_frame, textvariable=var, width=10).grid(row=i, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Max Generations").grid(
            row=len(self.param_vars), column=0, padx=5, pady=2)
        self.max_gen_var = tk.StringVar(value="100")
        ttk.Entry(param_frame, textvariable=self.max_gen_var, width=10).grid(
            row=len(self.param_vars), column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Display Delay (ms)").grid(
            row=len(self.param_vars) + 1, column=0, padx=5, pady=2)
        self.delay_var = tk.StringVar(value="1000")
        ttk.Entry(param_frame, textvariable=self.delay_var, width=10).grid(
            row=len(self.param_vars) + 1, column=1, padx=5, pady=2)
        
        ttk.Button(param_frame, text="Apply", command=self.apply_parameters).grid(
            row=len(self.param_vars) + 2, column=0, columnspan=2, pady=5)
            
    def setup_stats(self):
        """Setup statistics display"""
        stats_frame = ttk.LabelFrame(self.right_frame, text="Statistics", padding="5")
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.stat_vars = {
            "Generation": tk.StringVar(value="1"),
            "Current Flow": tk.StringVar(value="0"),
            "Best Flow": tk.StringVar(value="0")
        }
        
        important_stats = ["Generation", "Current Flow", "Best Flow"]
        
        for i, stat in enumerate(important_stats):
            ttk.Label(stats_frame, text=stat).grid(row=i, column=0, padx=5, pady=2)
            ttk.Label(stats_frame, textvariable=self.stat_vars[stat]).grid(
                row=i, column=1, padx=5, pady=2)
        
    def update_network_plot(self):
        """Update the network visualization"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        node_colors = ['lightgreen' if node == 'S' else 'lightcoral' if node == 'T' 
                      else 'lightblue' for node in self.network.graph.nodes()]
        
        node_size = 2000
        nx.draw_networkx_nodes(self.network.graph, self.pos, 
                              node_color=node_colors,
                              node_size=node_size,
                              edgecolors='black',
                              linewidths=2,
                              ax=ax)
        
        nx.draw_networkx_labels(self.network.graph, self.pos, 
                              font_size=16,
                              font_weight='bold',
                              ax=ax)
        
        current_individual = self.current_individual if hasattr(self, 'current_individual') else self.ga.best_solution
        
        if current_individual:
            node_radius = np.sqrt(node_size / np.pi) / 80 
            
            for u, v, data in current_individual.network.edges(data=True):
                edge_color = 'royalblue' if data['flow'] > 0 else 'lightgray'
                edge_width = 3 if data['flow'] > 0 else 1
                
                start_pos = np.array(self.pos[u])
                end_pos = np.array(self.pos[v])
                
                direction = end_pos - start_pos
                direction_norm = np.linalg.norm(direction)
                direction = direction / direction_norm
                
                start_point = start_pos + direction * node_radius
                end_point = end_pos - direction * node_radius
                
                rad = 0.25
                if (u, v) in [('A', 'B'), ('B', 'C'), ('D', 'E')]:
                    rad = 0.4
                elif (u, v) in [('B', 'D'), ('B', 'E')]:
                    rad = -0.2
                
                connectionstyle = f"arc3,rad={rad}"
                ax.annotate("",
                           xy=end_point,
                           xytext=start_point,
                           arrowprops=dict(arrowstyle="-|>",  
                                         color=edge_color,
                                         lw=edge_width,
                                         connectionstyle=connectionstyle,
                                         mutation_scale=20,
                                         shrinkA=0,  
                                         shrinkB=0)) 
                
                edge_center = start_point * 0.6 + end_point * 0.4
                if rad > 0:
                    edge_center[1] += 0.15
                else:
                    edge_center[1] -= 0.15
                
                edge_label = f"{data['flow']}/{data['capacity']}"
                ax.annotate(edge_label,
                           xy=edge_center,
                           xytext=(0, 0),
                           textcoords='offset points',
                           ha='center',
                           va='center',
                           bbox=dict(boxstyle='round4,pad=0.6', 
                                     fc='white',
                                     ec='gray',
                                     alpha=0.9,
                                     mutation_aspect=0.5),  
                           fontsize=12,  
                           fontweight='bold')  
        
        title = [f"Generation {self.ga.generation + 1}"]
        if hasattr(self, 'current_entity_index'):
            title.append(f"Entity {self.current_entity_index + 1}/{len(self.ga.population)}")
        if self.current_individual:
            current_flow = sum(self.current_individual.network[u]['T']['flow'] 
                             for u in self.current_individual.network.predecessors('T'))
            title.append(f"Current Flow: {current_flow}")
        if self.ga.best_solution:
            best_flow = sum(self.ga.best_solution.network[u]['T']['flow'] 
                           for u in self.ga.best_solution.network.predecessors('T'))
            title.append(f"Best Flow Ever: {best_flow}")
        
        ax.set_title('\n'.join(title), pad=20, fontsize=14)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        ax.margins(0.25)
        
        self.canvas.draw()
        
    def update_stats(self):
        """Update statistics display"""
        if self.ga.best_solution:
            self.stat_vars["Generation"].set(str(self.ga.generation + 1))
            
            best_flow = sum(self.ga.best_solution.network[u]['T']['flow'] 
                          for u in self.ga.best_solution.network.predecessors('T'))
            self.stat_vars["Best Flow"].set(str(best_flow))
            
            if hasattr(self, 'current_individual') and self.current_individual:
                current_flow = sum(self.current_individual.network[u]['T']['flow'] 
                                 for u in self.current_individual.network.predecessors('T'))
                self.stat_vars["Current Flow"].set(str(current_flow))

    def toggle_simulation(self):
        """Toggle simulation start/stop"""
        if self.start_button["text"] == "Start":
            self.ga.generation = 0
            self.ga.initialize_population()
            self.start_button["text"] = "Stop"
            self.run_simulation()
        else:
            self.start_button["text"] = "Start"
            
    def run_simulation(self):
        """Run simulation showing each entity"""
        if self.start_button["text"] == "Stop":
            max_generations = int(self.max_gen_var.get())
            delay = int(self.delay_var.get())
            
            if not hasattr(self, 'current_entity_index'):
                self.current_entity_index = 0
            
            if self.ga.generation >= max_generations - 1:
                self.start_button["text"] = "Start"
                best_flow = sum(self.ga.best_solution.network[u]['T']['flow'] 
                           for u in self.ga.best_solution.network.predecessors('T'))
                tk.messagebox.showinfo("Simulation Complete", 
                    f"Simulation completed after {max_generations} generations\n"
                    f"Best flow found: {best_flow}")
                return
                
            self.current_individual = self.ga.population[self.current_entity_index]
            if self.current_individual.fitness == 0:
                self.current_individual.fitness = self.ga.calculate_fitness(self.current_individual)
            
            self.update_network_plot()
            self.update_stats()
            
            self.current_entity_index += 1
            if self.current_entity_index >= len(self.ga.population):
                self.ga.evolve_population()
                best_flow = sum(self.ga.best_solution.network[u]['T']['flow'] 
                           for u in self.ga.best_solution.network.predecessors('T'))
                print(f"Generation {self.ga.generation + 1} completed. Best flow: {best_flow}")
                self.ga.generation += 1
                self.current_entity_index = 0
            
            self.root.after(delay, self.run_simulation)

    def apply_parameters(self):
        """Apply parameter changes"""
        try:
            self.ga.population_size = int(self.param_vars["Population Size"].get())
            self.ga.mutation_rate = float(self.param_vars["Mutation Rate"].get())
            self.ga.balancing_factor = float(self.param_vars["Balancing Factor"].get())
            self.ga.truncation_rate = float(self.param_vars["Truncation Rate"].get())
            
            self.ga.initialize_population()
            
            self.current_entity_index = 0
            self.update_network_plot()
            self.update_stats()
            
            tk.messagebox.showinfo("Success", "Parameters updated successfully!")
        except ValueError:
            tk.messagebox.showerror("Error", "Invalid parameter values!")


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    app = NetworkFlowGUI(root)
    root.mainloop()