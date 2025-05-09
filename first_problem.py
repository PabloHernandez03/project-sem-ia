import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import mean, stdev
from scipy import stats
from collections import Counter
import time

class GearProblem:
    def __init__(self, target_ratio, min_teeth=12, max_teeth=60):
        self.target = target_ratio
        self.min = min_teeth
        self.max = max_teeth
        self.values = list(range(min_teeth, max_teeth + 1))
        
    def evaluate(self, solution):
        A, B, D, F = solution
        ratio = (B * D) / (A * F)
        return (self.target - ratio)**2

class ACO:
    def __init__(self, problem, num_ants=50, iterations=100, evaporation=0.4, 
                 alpha=1, beta=2, elite=3):
        self.problem = problem
        self.num_ants = num_ants
        self.iterations = iterations
        self.error_threshold = 1e-6
        self.evaporation = evaporation
        self.alpha = alpha
        self.beta = beta
        self.elite = elite
        
        self.initialize_pheromones()
        self.initialize_heuristics()
        
    def initialize_pheromones(self):
        n = len(self.problem.values)
        self.pheromones = {
            var: np.ones(n) for var in ['A', 'B', 'D', 'F']
        }
        
    def initialize_heuristics(self):
        self.heuristics = {
            'A': [v/self.problem.max for v in self.problem.values],
            'F': [v/self.problem.max for v in self.problem.values],
            'B': [1.0]*len(self.problem.values),
            'D': [1.0]*len(self.problem.values)
        }
        
    def select_value(self, var):
        probs = (self.pheromones[var]**self.alpha) * (np.array(self.heuristics[var])**self.beta)
        probs /= probs.sum()
        return np.random.choice(len(self.problem.values), p=probs)
    
    def run(self):
        best_error = float('inf')
        best_solution = None
        convergence = []
        iterations_to_solution = 0
        
        for iteration in range(self.iterations):
            iterations_to_solution += 1
            solutions = []
            for __ in range(self.num_ants):
                indices = [self.select_value(var) for var in ['A', 'B', 'D', 'F']]
                solution = [self.problem.values[i] for i in indices]
                error = self.problem.evaluate(solution)
                solutions.append((error, indices))
            
            solutions.sort(key=lambda x: x[0])
            current_best_error = solutions[0][0]
            
            if current_best_error < best_error:
                best_error = current_best_error
                best_solution = [self.problem.values[i] for i in solutions[0][1]]
            
            convergence.append(best_error)
            self.update_pheromones(solutions)
            
            if best_error < self.error_threshold:
                break
        
        return best_solution, best_error, convergence, iterations_to_solution
    
    def update_pheromones(self, solutions):
        for var in self.pheromones:
            self.pheromones[var] *= (1 - self.evaporation)
            
        for i in range(self.elite):
            error, indices = solutions[i]
            delta = 1 / (1 + error)
            for var, idx in zip(['A', 'B', 'D', 'F'], indices):
                self.pheromones[var][idx] += delta

def multiple_runs(num_runs=10, **aco_params):
    problem = GearProblem(1/6.931)
    all_convergence = []
    solutions = defaultdict(list)
    run_times = []
    
    for run in range(num_runs):
        start_time = time.time()
        aco = ACO(problem, **aco_params)
        solution, error, convergence, iterations = aco.run()
        end_time = time.time()
        
        solutions[tuple(solution)].append(error)
        all_convergence.append(convergence)
        run_time = end_time - start_time
        run_times.append(run_time)
        print(f"Ejecución {run+1}/{num_runs} completada - Error: {error:.2e} - Tiempo: {run_time:.2f} segundos - Iteraciones: {iterations}")

    max_length = max(len(conv) for conv in all_convergence)
    padded_convergence = [np.pad(conv, (0, max_length - len(conv)), constant_values=np.nan) for conv in all_convergence]
    avg_convergence = np.nanmean(padded_convergence, axis=0)
    std_convergence = np.nanstd(padded_convergence, axis=0)
    
    unique_solutions = []
    for sol, errors in solutions.items():
        unique_solutions.append((sol, min(errors)))
    
    unique_solutions.sort(key=lambda x: x[1])
    
    avg_run_time = mean(run_times)
    print(f"\nTiempo promedio por ejecución: {avg_run_time:.2f} segundos")
    
    return unique_solutions, avg_convergence, std_convergence

def plot_convergence(avg_convergence, std_convergence):
    plt.figure(figsize=(10, 6))
    iterations = np.arange(1, len(avg_convergence) + 1)
    
    plt.plot(iterations, avg_convergence, label='Error Promedio', lw=2)
    plt.fill_between(iterations, 
                    avg_convergence - std_convergence,
                    avg_convergence + std_convergence,
                    alpha=0.2, label='Desviación Estándar')
    
    plt.yscale('log')
    plt.xlabel('Iteración')
    plt.ylabel('Error (log scale)')
    plt.title('Curva de Convergencia Promedio')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

num_runs = 50
aco_params = {
    'num_ants': 100,
    'iterations': 200,
    'evaporation': 0.5,
    'alpha': 1,
    'beta': 2,
    'elite': 5
}

results, avg_conv, std_conv = multiple_runs(num_runs=num_runs, **aco_params)

plot_convergence(avg_conv, std_conv)

errors = [error for _, error in results]
solutions = [sol for sol, _ in results]

print("\nAnálisis Estadístico:")
avg_solution = [mean([sol[i] for sol in solutions]) for i in range(4)]
avg_error = GearProblem(1/6.931).evaluate(avg_solution)
print(f"- Promedio de soluciones: {avg_solution}")
print(f"- Error del promedio: {avg_error:.2e}")

solution_counts = Counter(tuple(sol) for sol in solutions)
most_common_solution, count = solution_counts.most_common(1)[0]
print(f"- Solución más común: {most_common_solution}")
print(f"- Veces que aparece la solución más común: {count}")
most_common_solution_error = GearProblem(1/6.931).evaluate(most_common_solution)
print(f"- Error de la solución más común: {most_common_solution_error:.2e}")

modes = [stats.mode([sol[i] for sol in solutions], keepdims=True)[0][0] for i in range(4)]
mode_error = GearProblem(1/6.931).evaluate(modes)
print(f"- Moda de cada letra: {modes}")
print(f"- Error de la moda: {mode_error:.2e}")

avg_std_dev = np.mean(std_conv)
print(f"- Desviación estándar promedio: {avg_std_dev:.2e}")

print("\nTop 3 Soluciones:")
for i, (solution, error) in enumerate(results[:3]):
    A, B, D, F = solution
    ratio = (B * D) / (A * F)
    print(f"{i+1}. [A:{A} B:{B} D:{D} F:{F}]")
    print(f"   Error: {error:.2e} | Relación: {ratio:.6f} (Objetivo: {1/6.931:.6f})")