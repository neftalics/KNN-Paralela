import subprocess
import os
import sys

def run_experiment(p):
    print(f"Running with {p} processes...")
    # Asumimos que mpiexec está en el PATH
    cmd = ["mpiexec", "-n", str(p), "python", "src/knn_parallel_final.py"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running with {p} processes:")
        print(e.stderr)

def main():
    # Inicializar archivo de log
    with open("results_log.csv", "w") as f:
        f.write("processes,accuracy,total_time,compute_time,comm_time\n")

    # Definir número de procesos a probar
    # Nota: El máximo depende de los cores lógicos de la máquina del usuario.
    # Probaremos 1, 2, 4, 8 (ajustar según disponibilidad)
    process_counts = [1, 2, 4, 8]
    
    # Verificar cuántos cores tiene la máquina
    import multiprocessing
    max_cores = multiprocessing.cpu_count()
    print(f"Detected {max_cores} logical cores.")
    
    process_counts = [p for p in process_counts if p <= max_cores]
    if not process_counts:
        process_counts = [1] # Al menos 1

    for p in process_counts:
        run_experiment(p)

    print("Experiments completed. Results saved to results_log.csv")

if __name__ == "__main__":
    main()
