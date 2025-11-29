from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time
import sys

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(test_point, X_train, y_train, k):
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    X_train = None
    y_train = None
    X_test = None
    y_test = None
    
    # Sincronizar antes de empezar a medir tiempo total
    comm.Barrier()
    start_total_time = time.time()

    if rank == 0:
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        X_test_chunks = np.array_split(X_test, size)
    else:
        X_test_chunks = None

    # Comunicación
    start_comm_time = time.time()
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    local_X_test = comm.scatter(X_test_chunks, root=0)
    end_comm_time = time.time()
    comm_time = end_comm_time - start_comm_time

    # Cómputo
    k = 3
    start_compute_time = time.time()
    local_predictions = [knn_predict(x, X_train, y_train, k) for x in local_X_test]
    end_compute_time = time.time()
    compute_time = end_compute_time - start_compute_time

    # Recolección
    start_gather_time = time.time()
    all_predictions = comm.gather(local_predictions, root=0)
    end_gather_time = time.time()
    gather_time = end_gather_time - start_gather_time
    
    total_comm_time = comm_time + gather_time

    # Finalizar medición total
    comm.Barrier()
    end_total_time = time.time()
    total_time = end_total_time - start_total_time

    if rank == 0:
        flat_predictions = [item for sublist in all_predictions for item in sublist]
        accuracy = np.mean(flat_predictions == y_test)
        
        # Recolectar tiempos de todos los procesos para promediar o tomar el máximo
        # Usaremos el máximo tiempo de cómputo entre procesos como el tiempo de cómputo paralelo efectivo
        # Y el tiempo total del rank 0
        
        print(f"Processes: {size}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Total Time: {total_time:.4f} sec")
        print(f"Compute Time (Rank 0): {compute_time:.4f} sec")
        print(f"Comm Time (Rank 0): {total_comm_time:.4f} sec")
        
        # Guardar resultados en un archivo para análisis posterior (opcional, pero útil)
        # Formato: p, accuracy, total_time, compute_time, comm_time
        with open("results_log.csv", "a") as f:
            f.write(f"{size},{accuracy},{total_time},{compute_time},{total_comm_time}\n")

if __name__ == "__main__":
    main()
