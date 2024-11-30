import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

class TrafficNetwork:
    '''Khởi tạo đồ thị mạng giao thông'''
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

class Individual:
    def __init__(self, network: nx.DiGraph):
        self.network = network.copy()
        self.fitness = 0
        self.initialize_random_flow()
            
    def initialize_random_flow(self):
        """Khởi tạo giá trị luồng ngẫu nhiên hợp lệ cho tất cả các cạnh đồ thị sử dụng dfs"""
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

    '''Hàm tính toán năng lượng của mỗi đỉnh được tính bằng công thức: 

        Năng lượng đỉnh = tham số cân bằng x |Tổng luồng vào - Tổng luồng ra|
    '''    
    def calculate_vertex_energy(self, vertex: str) -> float:
        if vertex in ['S', 'T']:  
            return 0
            
        incoming_flow = sum(self.network[u][vertex]['flow'] #Tổng luồng vào
            for u in self.network.predecessors(vertex))
        
        outgoing_flow = sum(self.network[vertex][v]['flow'] #Tổng luồng ra
            for v in self.network.successors(vertex))
        
        excess_flow = abs(incoming_flow - outgoing_flow)
        
        max_incoming = sum(self.network[u][vertex]['capacity'] #Tổng công suất vào tối đa (hay tổng luồng vào tối đa)
                         for u in self.network.predecessors(vertex)) #Tổng công suất ra tối đa (hay tổng luồng ra tối đa)
        max_outgoing = sum(self.network[vertex][v]['capacity'] 
                         for v in self.network.successors(vertex)) #Công suất tối đa mà đỉnh có thể xử lý được (tức là 1 đỉnh không thể thu về hay giải phóng đi)
        max_possible = min(max_incoming, max_outgoing)
        
        actual_flow = min(incoming_flow, outgoing_flow) #Luồng thực tế tại đỉnh đang xử lý
        
        k = 1.4 # tham số cân bằng 
        energy = k * excess_flow + abs(max_possible - actual_flow)
        return energy

def individual_fitness_key(individual): #Hàm này được dùng để lấy giá trị fitness của cá thể
    return individual.fitness

class GeneticAlgorithm:
    def __init__(self, network: TrafficNetwork, 
                 population_size: int = 50,
                 mutation_rate: float = 0.02, 
                 balancing_factor: float = 1.4,
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
        """Khởi tạo một quần thể thế hệ với các cá thể (mạng giao thông) ngẫu nhiên"""
        self.population = [
            Individual(self.network.graph) 
            for _ in range(self.population_size - 1)
        ]
        
        for individual in self.population:
            individual.fitness = self.calculate_fitness(individual)
            
        self.best_solution = max(self.population, key=individual_fitness_key)

    def calculate_fitness(self, individual: Individual) -> float:
        """Tính toán giá trị fitness của từng cá thể theo công thức:
        fitness = (2 x Luồng vào đỉnh thu T) - Tổng năng lượng + (Tổng luồng / Tổng công suất)
            - Tổng năng lượng: Tổng năng lượng của mọi đỉnh trong cá thế (mạng giao thông) đó
            - Tổng luồng: Tổng giá trị luồng chạy trên cá thể (mạng giao thông) đó
            - Tổng công suất: Tổng công suất của cá thể (mạng giao thông) đó
        """
        sink_flow = sum(individual.network[u]['T']['flow'] #Luồng vào T
            for u in individual.network.predecessors('T'))
        
        balance_penalty = sum(individual.calculate_vertex_energy(v) #Tổng năng lượng
            for v in individual.network.nodes())
        
        total_flow = sum(data['flow'] #Tổng luồng
            for _, _, data in individual.network.edges(data=True))
        
        total_capacity = sum(data['capacity'] #Tổng công suất
            for _, _, data in individual.network.edges(data=True))
        
        utilization = total_flow / total_capacity if total_capacity > 0 else 0
        
        fitness = (sink_flow * 2) - balance_penalty + utilization # Fitness

        return max(0, fitness) 



    def select_parent(self):
        """Chọn lựa cặp bố mẹ sử dụng phương pháp tournament selection"""
        tournament_size = min(3, len(self.population))  
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=individual_fitness_key)
        
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Tạo ra cá thể con bằng cách ghép từng cặp đỉnh tương ứng của cá thể bố mẹ"""
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
        """Đột biến cá thể"""
        for u, v, data in individual.network.edges(data=True):
            if random.random() < self.mutation_rate:
                if data['capacity'] > 0:
                    if random.random() < 0.5:
                        data['flow'] = max(0, data['flow'] - random.randint(1, 3))
                    else:
                        data['flow'] = min(data['capacity'], 
                                         data['flow'] + random.randint(1, 3))
                    
    def evolve_population(self):
        """Tiến hóa quần thể thế hệ nhất định"""
        new_population = []
        
        self.population.sort(key=individual_fitness_key, reverse=True)
        
        elite_size = int(self.population_size * 0.1)  # Chọn lọc các cá thể trong thế hệ đó và chỉ giữ lại 10% số cá thể dựa trên chỉ số fitness
        new_population.extend(self.population[:elite_size]) 
        #Để đảm bảo đủ số lượng cá thể, chương trình sẽ chọn và lai ghép để sinh ra thêm cá thể
        
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

class NetworkAnalyzer:
    def __init__(self, ga_instance):
        self.ga = ga_instance
        self.convergence_data = []
        self.efficiency_data = {}
        self.node_balance_data = {}
        
    def analyze_convergence(self):
        """Phân tích khả năng hội tụ của thuật toán"""
        # Tính tốc độ hội tụ
        max_fitness = max(self.ga.history['fitness'])
        convergence_gen = 0
        threshold = 0.95 * max_fitness
        
        for gen, fitness in enumerate(self.ga.history['fitness']):
            if fitness >= threshold:
                convergence_gen = gen
                break
                
        # Tính độ ổn định
        last_20_gen = self.ga.history['fitness'][-20:]
        stability = np.std(last_20_gen)
        
        self.convergence_data = {
            'generations_to_converge': convergence_gen,
            'max_fitness': max_fitness,
            'stability': stability
        }
        
    def analyze_efficiency(self):
        """Phân tích hiệu quả của giải pháp"""
        if not self.ga.best_solution:
            return
            
        # Tính tổng luồng đến đích
        total_sink_flow = sum(self.ga.best_solution.network[u]['T']['flow'] 
                            for u in self.ga.best_solution.network.predecessors('T'))
                            
        # Tính tổng công suất có thể
        total_capacity = sum(data['capacity'] 
                           for _, _, data in self.ga.best_solution.network.edges(data=True))
                           
        # Tính hiệu suất sử dụng
        utilization = sum(data['flow'] for _, _, data in self.ga.best_solution.network.edges(data=True)) / total_capacity
        
        self.efficiency_data = {
            'total_flow': total_sink_flow,
            'total_capacity': total_capacity,
            'utilization': utilization
        }
        
    def analyze_node_balance(self):
        """Phân tích cân bằng tại các nút"""
        if not self.ga.best_solution:
            return
            
        for node in self.ga.best_solution.network.nodes():
            if node not in ['S', 'T']:
                energy = self.ga.best_solution.calculate_vertex_energy(node)
                self.node_balance_data[node] = energy
                
    def plot_convergence_curve(self):
        """Vẽ đường cong hội tụ"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.ga.history['fitness'], label='Fitness')
        plt.axhline(y=self.convergence_data['max_fitness'] * 0.95, 
                   color='r', linestyle='--', label='95% of max fitness')
        plt.axvline(x=self.convergence_data['generations_to_converge'], 
                   color='g', linestyle='--', label='Convergence point')
        plt.title('Đường cong hội tụ')
        plt.xlabel('Thế hệ')
        plt.ylabel('Fitness')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.ga.history['max_flow'], label='Max Flow')
        plt.title('Luồng cực đại theo thế hệ')
        plt.xlabel('Thế hệ')
        plt.ylabel('Luồng')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def generate_report(self):
        """Tạo báo cáo phân tích"""
        self.analyze_convergence()
        self.analyze_efficiency()
        self.analyze_node_balance()
        
        print("=== BÁO CÁO PHÂN TÍCH ===\n")
        
        print("1. Khả năng hội tụ:")
        print(f"- Số thế hệ để hội tụ: {self.convergence_data['generations_to_converge']}")
        print(f"- Độ ổn định (std): {self.convergence_data['stability']:.2f}")
        print(f"- Fitness tốt nhất: {self.convergence_data['max_fitness']:.2f}\n")
        
        print("2. Hiệu quả giải pháp:")
        print(f"- Tổng luồng đạt được: {self.efficiency_data['total_flow']}")
        print(f"- Tổng công suất: {self.efficiency_data['total_capacity']}")
        print(f"- Hiệu suất sử dụng: {self.efficiency_data['utilization']*100:.1f}%\n")
        
        print("3. Cân bằng tại các nút:")
        for node, energy in self.node_balance_data.items():
            print(f"- Nút {node}: {energy:.2f}")
            
        self.plot_convergence_curve()

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
        
        self.network = TrafficNetwork()
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
                
                # Thêm phân tích sau khi mô phỏng hoàn thành
                analyzer = NetworkAnalyzer(self.ga)
                analyzer.generate_report()
                
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