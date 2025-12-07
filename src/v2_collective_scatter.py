"""
KNN Paralelo - Versión 2: Collective Operations (Mejora de Comunicación)

Propósito:
    Demostrar cómo las operaciones colectivas (Scatter, Bcast, Gather) optimizan
    el ancho de banda (β) y reducen la latencia comparado con P2P.

Estrategia:
    - Usa MPI_Scatter para distribuir X_test eficientemente
    - Usa MPI_Bcast para replicar X_train a todos los procesos
    - Usa MPI_Gather para recolectar resultados
    - Aún usa bucles Python para cálculo de distancias (no vectorizado)

Modelo de Costo:
    T_comm ≈ log(p) × (α + N×β)
    donde p = procesos, α = latencia, β = inverso del ancho de banda
    
Mejora sobre v1:
    - Comunicación en árbol logarítmico reduce latencia
    - Mejor uso del ancho de banda de red
"""

from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time

def euclidean_distance(a, b):
    """Calcula distancia euclidiana entre dos vectores"""
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(test_point, X_train, y_train, k):
    """Predice la clase de un punto usando KNN (con bucles Python)"""
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parámetro k para KNN
    k = 3
    
    # Variables iniciales
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    X_test_chunks = None
    
    # Sincronización inicial
    comm.Barrier()
    start_total_time = MPI.Wtime()
    
    # ============================================
    # FASE 1: I/O (Solo Master)
    # ============================================
    if rank == 0:
        start_io_time = MPI.Wtime()
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        end_io_time = MPI.Wtime()
        io_time = end_io_time - start_io_time
        
        print(f"=== KNN Parallel v2 (Collective Operations) ===")
        print(f"Processes: {size}")
        print(f"Dataset: N={len(X_train)}, M={len(X_test)}, d={X_train.shape[1]}")
        
        # Dividir X_test para scatter
        X_test_chunks = np.array_split(X_test, size)
    else:
        io_time = 0.0
    
    # ============================================
    # FASE 2: COMUNICACIÓN (Operaciones Colectivas)
    # ============================================
    
    # Timing: Broadcast de datos de entrenamiento
    start_bcast_time = MPI.Wtime()
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    end_bcast_time = MPI.Wtime()
    bcast_time = end_bcast_time - start_bcast_time
    
    # Timing: Scatter de datos de prueba
    start_scatter_time = MPI.Wtime()
    local_X_test = comm.scatter(X_test_chunks, root=0)
    end_scatter_time = MPI.Wtime()
    scatter_time = end_scatter_time - start_scatter_time
    
    # ============================================
    # FASE 3: CÓMPUTO LOCAL (Con bucles Python)
    # ============================================
    start_compute_time = MPI.Wtime()
    
    # Nota: Aún usa bucles Python (no vectorizado)
    local_predictions = []
    for test_point in local_X_test:
        pred = knn_predict(test_point, X_train, y_train, k)
        local_predictions.append(pred)
    
    end_compute_time = MPI.Wtime()
    compute_time = end_compute_time - start_compute_time
    
    # ============================================
    # FASE 4: RECOLECCIÓN (Gather)
    # ============================================
    start_gather_time = MPI.Wtime()
    all_predictions = comm.gather(local_predictions, root=0)
    end_gather_time = MPI.Wtime()
    gather_time = end_gather_time - start_gather_time
    
    # Sincronización final
    comm.Barrier()
    end_total_time = MPI.Wtime()
    total_time = end_total_time - start_total_time
    
    # ============================================
    # RESULTADOS (Solo Master)
    # ============================================
    if rank == 0:
        # Combinar predicciones
        flat_predictions = [item for sublist in all_predictions for item in sublist]
        accuracy = np.mean(np.array(flat_predictions) == y_test)
        
        # Tiempos de comunicación
        total_comm_time = bcast_time + scatter_time + gather_time
        
        # Resultados
        print(f"\nResults:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nTiming Breakdown:")
        print(f"Total Time: {total_time:.4f} sec")
        print(f"I/O Time: {io_time:.4f} sec")
        print(f"Bcast Time: {bcast_time:.4f} sec")
        print(f"Scatter Time: {scatter_time:.4f} sec")
        print(f"Compute Time: {compute_time:.4f} sec")
        print(f"Gather Time: {gather_time:.4f} sec")
        print(f"Total Comm Time: {total_comm_time:.4f} sec")
        print(f"\nNote: Collective operations reduce latency via tree-based communication")
        
        # Guardar resultados en CSV
        with open("results_strong_scaling.csv", "a", encoding='utf-8') as f:
            f.write(f"v2_collective_scatter,{size},{accuracy:.4f},{total_time:.4f},"
                   f"{io_time:.4f},{bcast_time:.4f},{scatter_time:.4f},"
                   f"{compute_time:.4f},{gather_time:.4f}\n")

if __name__ == "__main__":
    main()
