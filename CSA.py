import numpy as np
import matplotlib.pyplot as plt
import time

# Definimos la función objetivo
def objective_function(x):
    return (1/6.931 - (x[2] * x[1]) / (x[0] * x[3]))**2

# Parámetros del CSA
n_crows = 30        # número de cuervos (población)
n_iterations = 200  # número de iteraciones
flight_length = 2
awareness_probability = 0.1

lower_bound = 12
upper_bound = 60
dimension = 4  # porque x1, x2, x3, x4

error_threshold = 1e-6  # Umbral de error

# Función para inicializar posiciones
def initialize_population():
    return np.random.uniform(lower_bound, upper_bound, (n_crows, dimension))

# Función para ajustar dentro de límites
def check_bounds(position):
    return np.clip(position, lower_bound, upper_bound)

# CSA principal
def CSA():
    start_time = time.time()
    
    crows = initialize_population()
    memory = np.copy(crows)
    best_crow = None
    best_score = np.inf
    convergence_curve = []

    for iter in range(n_iterations):
        new_positions = np.copy(crows)

        for i in range(n_crows):
            r = np.random.randint(0, n_crows)
            if np.random.rand() > awareness_probability:
                # Sigue a otro cuervo
                new_position = crows[i] + flight_length * np.random.rand(dimension) * (memory[r] - crows[i])
            else:
                # Movimiento aleatorio
                new_position = np.random.uniform(lower_bound, upper_bound, dimension)
            
            new_position = check_bounds(new_position)

            # Evaluación
            if objective_function(new_position) < objective_function(memory[i]):
                memory[i] = new_position

        # Actualizar cuervos
        crows = np.copy(memory)

        # Actualizar el mejor
        for crow in crows:
            score = objective_function(crow)
            if score < best_score:
                best_score = score
                best_crow = crow

        convergence_curve.append(best_score)
        if best_score <= error_threshold:
            break

    end_time = time.time()
    execution_time = end_time - start_time

    return best_crow, best_score, convergence_curve, execution_time

# Ejecutamos CSA varias veces para estadísticas
n_runs = 50
all_best_scores = []
all_execution_times = []
best_solutions = []
all_convergences = []

for run in range(n_runs):
    best_crow, best_score, convergence_curve, exec_time = CSA()
    all_best_scores.append(best_score)
    all_execution_times.append(exec_time)
    best_solutions.append(best_crow)
    all_convergences.append(convergence_curve)

# Resultados estadísticos
average_best = np.mean(all_best_scores)
std_best = np.std(all_best_scores)
average_time = np.mean(all_execution_times)

# Mejor solución encontrada
best_run_idx = np.argmin(all_best_scores)
best_solution = best_solutions[best_run_idx]
best_objective = all_best_scores[best_run_idx]

solutions_array = np.array(best_solutions)
avg_solution = np.mean(solutions_array, axis=0)

print("\n--- Resultados CSA ---")
print(f"Mejor solución encontrada: {best_solution}")
print(f"Valor de la función objetivo: {best_objective}")
print(f"Promedio de soluciones: {np.round(avg_solution, 2)}")
print(f"Promedio de mejores soluciones: {average_best}")
print(f"Desviación estándar: {std_best}")
print(f"Tiempo promedio de ejecución: {average_time} segundos")

sorted_indices = np.argsort(all_best_scores)[:3]
for i, idx in enumerate(sorted_indices):
    sol = best_solutions[idx]
    ratio = (sol[2]*sol[1])/(sol[0]*sol[3])
    print(f"\nTop {i+1}:")
    print(f"Variables: A={sol[0]}, B={sol[1]}, D={sol[2]}, F={sol[3]}")
    print(f"Error: {all_best_scores[idx]:.2e} | Relación: {ratio:.6f}")

# Verificación de restricciones
print("\nVerificación de restricciones:")
print(f"Variables dentro de [12, 60]: {np.all((best_solution >= 12) & (best_solution <= 60))}")

# Graficar la curva de convergencia (mejor corrida)
plt.figure(figsize=(10,6))
plt.plot(all_convergences[best_run_idx], label='CSA')
plt.title('Curva de Convergencia CSA')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función objetivo')
plt.legend()
plt.grid()
plt.show()