from mpi4py import MPI
from sklearn.datasets import make_classification
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

    # Argumentos: n_samples
    n_samples = 1000 # Default
    if len(sys.argv) > 1:
        n_samples = int(sys.argv[1])

    X_train = None
    y_train = None
    X_test = None
    y_test = None
    
    comm.Barrier()
    start_total_time = time.time()

    if rank == 0:
        # Generar datos sintéticos
        # n_features=64 para simular digits
        X, y = make_classification(n_samples=n_samples, n_features=64, n_classes=10, n_informative=40, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_chunks = np.array_split(X_test, size)
    else:
        X_test_chunks = None

    # Comunicación
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    local_X_test = comm.scatter(X_test_chunks, root=0)

    # Cómputo
    k = 3
    local_predictions = [knn_predict(x, X_train, y_train, k) for x in local_X_test]

    # Recolección
    all_predictions = comm.gather(local_predictions, root=0)
    
    comm.Barrier()
    end_total_time = time.time()
    total_time = end_total_time - start_total_time

    if rank == 0:
        flat_predictions = [item for sublist in all_predictions for item in sublist]
        accuracy = np.mean(flat_predictions == y_test)
        
        # Output CSV format: n_samples, processes, time, accuracy
        print(f"{n_samples},{size},{total_time:.4f},{accuracy:.4f}")

if __name__ == "__main__":
    main()
