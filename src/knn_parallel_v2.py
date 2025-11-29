from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
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
    
    if rank == 0:
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        # Dividir X_test para scatter
        X_test_chunks = np.array_split(X_test, size)
    else:
        X_test_chunks = None

    # Broadcast de datos de entrenamiento (necesarios en todos los nodos)
    # bcast envía el mismo dato a todos los procesos
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    
    # Scatter de datos de prueba
    local_X_test = comm.scatter(X_test_chunks, root=0)

    # Parámetro k
    k = 3

    # Cálculo local
    if rank == 0:
        print(f"Rank {rank}: Starting local predictions for {len(local_X_test)} samples...")
    
    local_predictions = [knn_predict(x, X_train, y_train, k) for x in local_X_test]

    # Gather resultados
    all_predictions = comm.gather(local_predictions, root=0)

    if rank == 0:
        # Aplanar
        flat_predictions = [item for sublist in all_predictions for item in sublist]
        
        # Verificar exactitud (solo posible si tenemos y_test en root)
        accuracy = np.mean(flat_predictions == y_test)
        print(f"Rank {rank}: Predictions gathered. Accuracy: {accuracy:.4f}")
        print("Beta 2 Test Completed Successfully.")

if __name__ == "__main__":
    main()
