import numpy as np
from time import time
import matplotlib.pyplot as plt

class History:
    def __init__(self):
        self.best_s: np.array = None
        self.best_f: float = float('inf')
        self.execution_time: float = None
        self.fitness_history = []  # Para almacenar el mejor fitness en cada iteración

class Wolf:
    def __init__(self, solution: np.ndarray):
        self.solution = solution
        self.fitness = None
    
    def info(self):
        print(f"Solution: {self.solution}, Fitness: {self.fitness}")

class GreyWolfOptimizer:
    def __init__(self, swarm_size: int, lb: int, ub: int, dim: int):
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.swarm = [Wolf(np.random.uniform(low=lb, high=ub, size=dim)) for _ in range(swarm_size)]
        self.history = History()
        self.alpha = None
        self.beta = None
        self.delta = None
    
    def show_swarm(self):
        for wolf in self.swarm:
            wolf.info()
    
    def evaluate(self, fn_aptitude):
        for wolf in self.swarm:
            wolf.fitness = fn_aptitude(*wolf.solution)
    
    def update_leaders(self):
        # Ordenamos los lobos por fitness (minimización)
        sorted_wolves = sorted(self.swarm, key=lambda x: x.fitness)
        self.alpha = sorted_wolves[0]
        self.beta = sorted_wolves[1]
        self.delta = sorted_wolves[2]
    
    def optimize(self, fn_aptitude, iterations: int):
        time_start = time()
        
        self.evaluate(fn_aptitude)
        self.update_leaders()
        self.history.fitness_history.append(self.alpha.fitness)
        
        for t in range(iterations):
            a = 2 - t * (2 / iterations)  # a disminuye linealmente de 2 a 0
            
            for wolf in self.swarm:
                # Actualización para alpha, beta y delta
                X_alpha = self.update_position(wolf, self.alpha, a)
                X_beta = self.update_position(wolf, self.beta, a)
                X_delta = self.update_position(wolf, self.delta, a)
                
                # Nueva posición es el promedio de las tres
                new_solution = (X_alpha + X_beta + X_delta) / 3
                new_solution = np.clip(new_solution, self.lb, self.ub)
                
                # Evaluar nueva posición
                new_fitness = fn_aptitude(*new_solution)
                
                # Actualizar si es mejor
                if new_fitness < wolf.fitness:
                    wolf.solution = new_solution
                    wolf.fitness = new_fitness
            
            # Actualizar líderes
            self.update_leaders()
            self.history.fitness_history.append(self.alpha.fitness)
            self.history.best_s = self.alpha.solution.copy()
            self.history.best_f = self.alpha.fitness
        
        self.history.execution_time = time() - time_start
    
    def update_position(self, wolf, leader, a):
        A1 = 2 * a * np.random.rand(self.dim) - a
        C1 = 2 * np.random.rand(self.dim)
        D = np.abs(C1 * leader.solution - wolf.solution)
        return leader.solution - A1 * D

def problem(*args) -> np.float64:
    x1, x2, x3, x4 = args

    f = 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3**2 + 3.1661 * x1**2 * x4 + 19.84 * x1**2 * x3

    penalty = 0

    if -x1 + 0.0193 * x3 > 0:
        penalty += 1e6
    if -x2 + 0.00954 * x3 > 0:  
        penalty += 1e6
    if -np.pi * x3**2 * x4 - (4/3) * np.pi * x3**3 + 1296000 > 0:
        penalty += 1e6
    if x4 - 240 > 0:
        penalty += 1e6

    if not (0 < x1 < 100):
        penalty += 1e6
    if not (0 < x2 < 100):
        penalty += 1e6
    if not (10 <= x3 <= 200):
        penalty += 1e6
    if not (10 <= x4 <= 200):
        penalty += 1e6

    return f + penalty

def runs(executions: int):
    solutions = []
    fitness = []
    execution_times = []

    n = 30  # Number of wolves (va explotar este pdo ----#actualizo esto porque es preferible tenerlo a 30)
    lb = 0   # Lower bound
    ub = 200 # Upper bound
    dim = 4  # Dimensions

    for _ in range(executions):
        gwo = GreyWolfOptimizer(n, lb, ub, dim)
        gwo.optimize(fn_aptitude=problem, iterations=300)
        history = gwo.history

        best_solution = history.best_s
        best_fitness = history.best_f
        execution_time = history.execution_time

        solutions.append(best_solution)
        fitness.append(best_fitness)
        execution_times.append(execution_time)

    mean_fitness = np.mean(fitness)
    std_fitness = np.std(fitness)
    mean_execution_time = np.mean(execution_times)
    best_solution = solutions[np.argmin(fitness)]
    best_fitness = min(fitness)

    # Métricas
    print(f"Número de ejecuciones: {executions}")
    print(f"Evaluación del fitness: {mean_fitness}")
    print(f"Desviación estandar del fitness: {std_fitness}")
    print(f"Tiempo de Ejecución: {mean_execution_time:.4f} segundos")
    print(f"Mejor Solucion: {best_solution}")
    print(f"Mejor fitness: {best_fitness}")

    # Mostrar restricciones
    print("\nConstraints:")
    print(f"{-best_solution[0]} + 0.0193 * {best_solution[2]} <= 0: {(-best_solution[0] + 0.0193 * best_solution[2]) <= 0}")
    print(f"{-best_solution[1]} + 0.00954 * {best_solution[2]} <= 0: {(-best_solution[1] + 0.00954 * best_solution[2]) <= 0}")
    print(f"-pi*{best_solution[2]}^2 * {best_solution[3]} - (4/3) * pi * {best_solution[2]}^3 + 1296000 <= 0: {(-np.pi * best_solution[2]**2 * best_solution[3] - (4/3) * np.pi * best_solution[2]**3 + 1296000) <= 0}")
    print(f"{best_solution[3]} - 240 <= 0: {(best_solution[3] - 240) <= 0}")
    print(f"0 <= {best_solution[0]} <= 100: {(0 <= best_solution[0] <= 100)}")
    print(f"0 <= {best_solution[1]} <= 100: {(0 <= best_solution[1] <= 100)}")
    print(f"10 <= {best_solution[2]} <= 200: {(10 <= best_solution[2] <= 200)}")
    print(f"10 <= {best_solution[3]} <= 200: {(10 <= best_solution[3] <= 200)}")
    
    # Graficar convergencia (usando los datos de la última ejecución)
    plt.plot(gwo.history.fitness_history, label='Mejor Fitness')
    plt.xlabel('Iteraciones')
    plt.ylabel('Fitness')
    plt.title('Curva de Convergencia GWO')
    plt.legend()
    plt.grid()
    plt.show()

executions = 10
runs(executions)