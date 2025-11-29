from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import sys

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    X_train = None
    y_train = None
    X_test = None
    y_test = None
    
    # Solo el proceso raíz carga los datos
    if rank == 0:
        print(f"Rank {rank}: Loading data...")
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        print(f"Rank {rank}: Data loaded. Test set size: {len(X_test)}")
        
        # Determinar chunks para scatter
        # Nota: scatter de mpi4py divide automáticamente si es una lista, 
        # pero para arrays de numpy es mejor usar Scatter o dividir manualmente si los tamaños no son uniformes.
        # Para simplificar Beta 1, usaremos array_split de numpy y scatter de objetos (lento pero fácil)
        # o Scatterv para eficiencia. Usaremos scatter de objetos para Beta 1 por simplicidad.
        X_test_chunks = np.array_split(X_test, size)
    else:
        X_test_chunks = None

    # Distribuir datos de prueba
    # comm.scatter envía un elemento de la lista a cada proceso
    local_X_test = comm.scatter(X_test_chunks, root=0)

    print(f"Rank {rank}: Received {len(local_X_test)} test samples.")

    # Simular procesamiento
    local_results = [f"Pred_from_{rank}_{i}" for i in range(len(local_X_test))]

    # Recolectar resultados
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        # Aplanar lista de listas
        flat_results = [item for sublist in all_results for item in sublist]
        print(f"Rank {rank}: Gathered {len(flat_results)} results.")
        print("Beta 1 Test Completed Successfully.")

if __name__ == "__main__":
    main()
