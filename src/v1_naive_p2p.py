"""
KNN Paralelo - Versión 1: Naive Point-to-Point (Línea Base Ineficiente)

Propósito:
    Demostrar el cuello de botella de latencia (α) cuando se usa comunicación
    punto a punto bloqueante en lugar de operaciones colectivas.

Estrategia:
    - Master lee el dataset y envía cada punto de prueba individualmente
    - Usa send/recv bloqueantes dentro de un bucle
    - Alta latencia debido a múltiples operaciones de comunicación
    - Demuestra por qué las operaciones colectivas son necesarias

Modelo de Costo:
    T_comm ≈ M × p × α (latencia dominante)
    donde M = puntos de prueba, p = procesos, α = latencia por mensaje
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
    """Predice la clase de un punto usando KNN"""
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
    
    # Sincronización inicial
    comm.Barrier()
    start_total_time = MPI.Wtime()
    
    if rank == 0:
        # ============================================
        # MASTER PROCESS (Rank 0)
        # ============================================
        
        # Timing: I/O (carga de datos)
        start_io_time = MPI.Wtime()
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        end_io_time = MPI.Wtime()
        io_time = end_io_time - start_io_time
        
        print(f"=== KNN Parallel v1 (Naive P2P) ===")
        print(f"Processes: {size}")
        print(f"Dataset: N={len(X_train)}, M={len(X_test)}, d={X_train.shape[1]}")
        
        # Dividir datos de prueba entre workers
        M = len(X_test)
        chunk_size = M // size
        
        # Caso especial: solo 1 proceso (secuencial)
        if size == 1:
            start_compute_time = MPI.Wtime()
            all_predictions = [knn_predict(x, X_train, y_train, k) for x in X_test]
            end_compute_time = MPI.Wtime()
            compute_time = end_compute_time - start_compute_time
            send_time = 0.0
            recv_time = 0.0
        else:
            # Timing: Comunicación P2P (envío de datos)
            start_send_time = MPI.Wtime()
            
            # Enviar X_train y y_train a todos los workers (punto a punto)
            for worker in range(1, size):
                comm.send(X_train, dest=worker, tag=1)
                comm.send(y_train, dest=worker, tag=2)
            
            # Enviar puntos de prueba uno por uno a cada worker
            for worker in range(1, size):
                start_idx = worker * chunk_size
                end_idx = (worker + 1) * chunk_size if worker < size - 1 else M
                
                # Enviar cada punto individualmente (INEFICIENTE)
                for i in range(start_idx, end_idx):
                    comm.send(X_test[i], dest=worker, tag=100 + i)
            
            end_send_time = MPI.Wtime()
            send_time = end_send_time - start_send_time
            
            # Master también procesa su porción
            start_compute_time = MPI.Wtime()
            master_predictions = []
            for i in range(chunk_size):
                pred = knn_predict(X_test[i], X_train, y_train, k)
                master_predictions.append(pred)
            end_compute_time = MPI.Wtime()
            compute_time = end_compute_time - start_compute_time
            
            # Timing: Recepción de resultados (punto a punto)
            start_recv_time = MPI.Wtime()
            all_predictions = master_predictions
            
            for worker in range(1, size):
                start_idx = worker * chunk_size
                end_idx = (worker + 1) * chunk_size if worker < size - 1 else M
                worker_predictions = []
                
                # Recibir cada predicción individualmente (INEFICIENTE)
                for i in range(start_idx, end_idx):
                    pred = comm.recv(source=worker, tag=200 + i)
                    worker_predictions.append(pred)
                
                all_predictions.extend(worker_predictions)
            
            end_recv_time = MPI.Wtime()
            recv_time = end_recv_time - start_recv_time
        
        # Calcular accuracy
        accuracy = np.mean(np.array(all_predictions) == y_test)
        
        # Timing total
        comm.Barrier()
        end_total_time = MPI.Wtime()
        total_time = end_total_time - start_total_time
        
        # Tiempos de comunicación
        p2p_comm_time = send_time + recv_time
        
        # Resultados
        print(f"\nResults:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nTiming Breakdown:")
        print(f"Total Time: {total_time:.4f} sec")
        print(f"I/O Time: {io_time:.4f} sec")
        print(f"P2P Send Time: {send_time:.4f} sec")
        print(f"Compute Time: {compute_time:.4f} sec")
        print(f"P2P Recv Time: {recv_time:.4f} sec")
        print(f"Total P2P Comm: {p2p_comm_time:.4f} sec")
        print(f"\nNote: High P2P communication time demonstrates latency bottleneck (α)")
        
        # Guardar resultados en CSV
        with open("results_strong_scaling.csv", "a", encoding='utf-8') as f:
            f.write(f"v1_naive_p2p,{size},{accuracy:.4f},{total_time:.4f},"
                   f"{io_time:.4f},0.0000,{send_time:.4f},{compute_time:.4f},"
                   f"{recv_time:.4f}\n")
    
    else:
        # ============================================
        # WORKER PROCESSES (Rank > 0)
        # ============================================
        
        # Recibir datos de entrenamiento
        X_train = comm.recv(source=0, tag=1)
        y_train = comm.recv(source=0, tag=2)
        
        # Calcular cuántos puntos de prueba procesará este worker
        M = 360  # Tamaño conocido del test set
        chunk_size = M // size
        start_idx = rank * chunk_size
        end_idx = (rank + 1) * chunk_size if rank < size - 1 else M
        
        # Recibir puntos de prueba uno por uno
        local_X_test = []
        for i in range(start_idx, end_idx):
            test_point = comm.recv(source=0, tag=100 + i)
            local_X_test.append(test_point)
        
        # Procesar localmente
        local_predictions = []
        for test_point in local_X_test:
            pred = knn_predict(test_point, X_train, y_train, k)
            local_predictions.append(pred)
        
        # Enviar resultados uno por uno
        for i, pred in enumerate(local_predictions):
            comm.send(pred, dest=0, tag=200 + start_idx + i)
        
        comm.Barrier()

if __name__ == "__main__":
    main()
