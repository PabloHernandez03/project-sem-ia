import numpy as np
import matplotlib.pyplot as plt
import time

# Función objetivo con penalización fuerte
def cost(x):
    x1, x2, x3, x4 = x
    f = 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 ** 2 + 3.1661 * x1 ** 2 * x4 + 19.84 * x1 ** 2 * x3
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -np.pi * x3 ** 2 * x4 - (4/3) * np.pi * x3 ** 3 + 1296000
    g4 = x4 - 240
    penalty = sum(max(0, g) ** 2 for g in [g1, g2, g3, g4])
    return f + 1e6 * penalty

# Parámetros PSO
n_particles, n_iters = 60, 300
w, c1, c2 = 0.7, 1.4, 1.4

# Límites de las variables
bounds = np.array([
    [0.01, 5.0],    # x1
    [0.01, 5.0],    # x2
    [10.0, 200.0],  # x3
    [10.0, 240.0]   # x4
])

# Inicialización
np.random.seed(1)
pos = np.random.uniform(bounds[:,0], bounds[:,1], (n_particles, 4))
vel = np.zeros_like(pos)
pbest_pos = pos.copy()
pbest_val = np.array([cost(p) for p in pos])
gbest_idx = np.argmin(pbest_val)
gbest_pos = pbest_pos[gbest_idx].copy()
gbest_val = pbest_val[gbest_idx]

# Métricas a recolectar
history = []
avg_fitness_per_iter = []
std_fitness_per_iter = []

start_time = time.time()

# Bucle PSO
for _ in range(n_iters):
    fitness_vals = np.array([cost(p) for p in pos])
    avg_fitness_per_iter.append(fitness_vals.mean())
    std_fitness_per_iter.append(fitness_vals.std())
    
    for i in range(n_particles):
        r1, r2 = np.random.rand(4), np.random.rand(4)
        vel[i] = w * vel[i] + c1 * r1 * (pbest_pos[i] - pos[i]) + c2 * r2 * (gbest_pos - pos[i])
        pos[i] = np.clip(pos[i] + vel[i], bounds[:,0], bounds[:,1])

        val = cost(pos[i])
        if val < pbest_val[i]:
            pbest_val[i] = val
            pbest_pos[i] = pos[i].copy()
            if val < gbest_val:
                gbest_val = val
                gbest_pos = pos[i].copy()
    history.append(gbest_val)

end_time = time.time()
total_time = end_time - start_time
time_per_iter = total_time / n_iters

# Cálculo de métricas finales
fitness_promedio = avg_fitness_per_iter[-1]
std_fitness = std_fitness_per_iter[-1]
mejor_fitness = gbest_val
mejor_solucion = gbest_pos
promedio_soluciones = pos.mean(axis=0)

# Mostrar resultados de métricas
def print_metrics():
    print("===== Resultados Finales PSO =====")
    print(f"Tiempo total de ejecución      : {total_time:.4f} seg")
    print(f"Tiempo promedio por iteración  : {time_per_iter:.6f} seg")
    print(f"Fitness promedio final         : {fitness_promedio:.4f}")
    print(f"Desviación estándar final      : {std_fitness:.4f}")
    print(f"Mejor fitness                  : {mejor_fitness:.4f}")
    print(f"Promedio de soluciones (x1,x2,x3,x4): {promedio_soluciones}")

# Mostrar solución PSO mejorada
def print_best_solution():
    print("\n\nSolución PSO:")
    print(f"x1 (espesor del recipiente) = {mejor_solucion[0]:.6f}")
    print(f"x2 (espesor de la cabeza) = {mejor_solucion[1]:.6f}")
    print(f"x3 (radio del recipiente) = {mejor_solucion[2]:.6f}")
    print(f"x4 (longitud de la capa) = {mejor_solucion[3]:.6f}")
    print(f"Costo mínimo = {mejor_fitness:.4f}")

if __name__ == "__main__":
    print_metrics()
    print_best_solution()

    # Gráficas
    plt.figure(figsize=(15, 5))
    # Convergencia del mejor costo
    plt.subplot(1, 3, 1)
    plt.plot(history, linewidth=1)
    plt.title("Convergencia: Mejor Costo")
    plt.xlabel("Iteración")
    plt.ylabel("Costo")
    plt.grid(True)
    # Evolución promedio y desviación
    plt.subplot(1, 3, 2)
    plt.plot(avg_fitness_per_iter, label='Fitness Promedio')
    plt.plot(std_fitness_per_iter, label='Desviación Estándar')
    plt.title("Estadísticas del Fitness")
    plt.xlabel("Iteración")
    plt.legend()
    plt.grid(True)
    # Mejor fitness acumulado
    plt.subplot(1, 3, 3)
    plt.plot(np.minimum.accumulate(avg_fitness_per_iter), label='Mejor Fitness Acumulado')
    plt.title("Mejor Fitness Promedio Acumulado")
    plt.xlabel("Iteración")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
