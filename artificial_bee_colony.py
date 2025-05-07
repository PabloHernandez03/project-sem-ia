import numpy as np
from time import time
import matplotlib.pyplot as plt

class History:
    def __init__(self)->None:
        self.best_s : np.array = None
        self.best_f: float = float('inf')
        self.execution_time : float = None

class Bee:
    def __init__(self,solution: np.ndarray)->None:
        self.solution = solution
        self.fitness = None
        self.trial = 0
        self.probability = 0.0

    def info(self)->None:
        print(f"Solution: {self.solution}, Fitness: {self.fitness}, Trial: {self.trial}, Probability: {self.probability}")
    
def init_swarm(swarm_size: int, lb: int, ub:int , dim:int)->list[Bee]:
    swarm = []
    for _ in range(swarm_size):
        bee = Bee(np.random.uniform(low=lb, high=ub, size=dim))
        swarm.append(bee)
    return swarm


class ArtificialBeeColony:
    def __init__(self, swarm_size: int, lb: int, ub:int , dim:int)->None:
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.swarm = init_swarm(swarm_size, lb, ub, dim)
        self.limit= None
        self.history : History = History()

    def show_swarm(self)->None:
        for bee in self.swarm:
            bee.info()

    def optimize(self,fn_aptitude,iterations: int, limit: int)->None:
        time_start = time()

        self.limit = limit
        self.evaluate(fn_aptitude)
        for _ in range(iterations):
            self.employ_bees(fn_aptitude)
            self.onlooker_bees(fn_aptitude)
            self.scout_bees(limit)
            self.update_best_solution()
        
        self.history.execution_time = time() - time_start

    def update_best_solution(self)->None:
        for bee in self.swarm:
            if bee.fitness < self.history.best_f:
                self.history.best_f = bee.fitness
                self.history.best_s = bee.solution.copy()

    def employ_bees(self,fn_aptitude)->None:
        for bee in self.swarm:
            j = np.random.randint(0, len(bee.solution))
            k = np.random.randint(0, len(self.swarm))
            phi = np.random.uniform(-1, 1)
            new_solution = bee.solution.copy()
            new_solution[j] += phi * (self.swarm[k].solution[j] - bee.solution[j])
            new_solution = np.clip(new_solution, self.lb, self.ub)
            new_fitness = fn_aptitude(*new_solution)
            
            if new_fitness < bee.fitness:
                bee.solution = new_solution
                bee.fitness = new_fitness
                bee.trial = 0
            else:
                bee.trial += 1

    def onlooker_bees(self,fn_aptitude)->None:
        for bee in self.swarm:
            if np.random.rand() < bee.probability:
                j = np.random.randint(0, len(bee.solution))
                k = np.random.randint(0, len(self.swarm))
                phi = np.random.uniform(-1, 1)
                new_solution = bee.solution.copy()
                new_solution[j] += phi * (self.swarm[k].solution[j] - bee.solution[j])
                new_solution = np.clip(new_solution, self.lb, self.ub)
                new_fitness = fn_aptitude(*new_solution)

                if new_fitness < bee.fitness:
                    bee.solution = new_solution
                    bee.fitness = new_fitness
                    bee.trial = 0
                else:
                    bee.trial += 1

    def scout_bees(self,limit: int)->None:
        for bee in self.swarm:
            if bee.trial > limit:
                bee.solution = np.random.uniform(low=self.lb, high=self.ub, size=self.dim)
                bee.fitness = float('inf')
                bee.trial = 0
                bee.probability = 0.0

    def evaluate(self,fn_aptitude)->None:
        fitness_values = []
        for bee in self.swarm:
            bee.fitness = fn_aptitude(*bee.solution)
            fitness_values.append(bee.fitness)

        # Para minimizar: probabilidad inversamente proporcional al fitness
        max_fit = max(fitness_values)
        inverted_fitness = [max_fit - f + 1e-8 for f in fitness_values]
        total = sum(inverted_fitness)

        for bee, inv_fit in zip(self.swarm, inverted_fitness):
            bee.probability = inv_fit / total


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

def runs(executions: int) ->None:
    solutions = []
    fitness = []
    execution_times = []

    n = 100 # Number of bees
    lb = 0   # Lower bound of the search space
    ub = 200 # Upper bound of the search space
    dim = 4 # Dimensions of the search space

    abc = ArtificialBeeColony(n, lb, ub, dim)

    for _ in range (executions):
        abc.optimize(fn_aptitude=problem,iterations=200, limit=50)
        history = abc.history

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

    # metricas 
    print(f"Number of executions: {executions}")
    print(f"Mean fitness: {mean_fitness}")
    print(f"Standard deviation of fitness: {std_fitness}")
    print(f"Mean execution time: {mean_execution_time:.4f} seconds")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")

    #mostrando se cumplen las restricciones
    print("Constraints:")
    print(f"{-best_solution[0]} + 0.0193 * {best_solution[2]} <= 0: {(-best_solution[0] + 0.0193 * best_solution[2]) <= 0}")
    print(f"{-best_solution[1]} + 0.00954 * {best_solution[2]} <= 0: {(-best_solution[1] + 0.00954 * best_solution[2]) <= 0}")
    print(f"-pi*{best_solution[2]}^2 * {best_solution[3]} - (4/3) * pi * {best_solution[2]}^3 + 1296000 <= 0: {(-np.pi * best_solution[2]**2 * best_solution[3] - (4/3) * np.pi * best_solution[2]**3 + 1296000) <= 0}")
    print(f"{best_solution[3]} - 240 <= 0: {(best_solution[3] - 240) <= 0}")
    print(f"0 <= {best_solution[0]} <= 100: {(0 <= best_solution[0] <= 100)}")
    print(f"0 <= {best_solution[1]} <= 100: {(0 <= best_solution[1] <= 100)}")
    print(f"10 <= {best_solution[2]} <= 200: {(10 <= best_solution[2] <= 200)}")
    print(f"10 <= {best_solution[3]} <= 200: {(10 <= best_solution[3] <= 200)}")
    
    # graficar los fitness de cada iteracion
    plt.plot(fitness, label='Fitness History')
    plt.xlabel('Executions')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid()
    plt.show()


executions = 10
runs(executions)