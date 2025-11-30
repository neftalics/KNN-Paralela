import subprocess
import os

def run_scaling_experiment():
    # Tamaños de problema a probar
    sample_sizes = [1000, 2000, 4000, 8000]
    # Número de procesos fijo para esta prueba (e.g., 4)
    p = 4
    
    results = []
    
    print(f"Running scaling experiments with p={p}...")
    
    for n in sample_sizes:
        cmd = ["mpiexec", "-n", str(p), "python", "src/knn_parallel_synthetic.py", str(n)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse output: n_samples, processes, time, accuracy
            # La salida puede tener líneas extra de MPI, buscamos la línea que parece CSV
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if "," in line:
                    parts = line.split(',')
                    if len(parts) == 4:
                        print(f"N={n}: Time={parts[2]}s")
                        results.append(line)
        except subprocess.CalledProcessError as e:
            print(f"Error with N={n}: {e.stderr}")

    # Guardar log
    with open("scaling_log.csv", "w") as f:
        f.write("n_samples,processes,total_time,accuracy\n")
        for line in results:
            f.write(line + "\n")
            
    print("Scaling experiments completed.")

if __name__ == "__main__":
    run_scaling_experiment()
