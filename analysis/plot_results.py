import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_results():
    try:
        df = pd.read_csv("results_log.csv")
    except FileNotFoundError:
        print("results_log.csv not found. Run experiments first.")
        return

    # Ordenar por número de procesos
    df = df.sort_values(by="processes")
    
    p = df["processes"]
    total_time = df["total_time"]
    compute_time = df["compute_time"]
    comm_time = df["comm_time"]
    
    # 1. Execution Time vs Processes
    plt.figure(figsize=(10, 6))
    plt.plot(p, total_time, marker='o', label='Total Time')
    plt.plot(p, compute_time, marker='s', label='Compute Time')
    plt.plot(p, comm_time, marker='^', label='Communication Time')
    plt.xlabel("Number of Processes (p)")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time Analysis")
    plt.legend()
    plt.grid(True)
    plt.savefig("analysis/time_vs_processes.png")
    plt.close()

    # 2. Speedup
    # Speedup = T_sequential / T_parallel
    # Usaremos T_total con p=1 como base si existe, sino el T_sequential teórico
    if 1 in p.values:
        t_seq = df[df["processes"] == 1]["total_time"].values[0]
        speedup = t_seq / total_time
        
        plt.figure(figsize=(10, 6))
        plt.plot(p, speedup, marker='o', label='Measured Speedup')
        plt.plot(p, p, '--', color='gray', label='Ideal Speedup') # Ideal linear speedup
        plt.xlabel("Number of Processes (p)")
        plt.ylabel("Speedup")
        plt.title("Speedup Analysis")
        plt.legend()
        plt.grid(True)
        plt.savefig("analysis/speedup.png")
        plt.close()

    # 3. FLOPs (Estimación)
    # FLOPs para KNN ~ N_test * N_train * D * 3 (resta, cuadrado, suma) aprox
    # D = 64 (8x8 images)
    # N_test ~ 360, N_train ~ 1437 (con split 0.2)
    # Esto es constante para el problema fijo.
    # FLOPs/sec = Total FLOPs / Compute Time
    
    # Hardcoded sizes for estimation based on standard load_digits
    n_samples = 1797
    n_test = int(n_samples * 0.2) # ~360
    n_train = n_samples - n_test  # ~1437
    d = 64
    ops_per_distance = d * 3 # very rough estimate
    total_flops = n_test * n_train * ops_per_distance
    
    flops_per_sec = total_flops / total_time
    
    plt.figure(figsize=(10, 6))
    plt.plot(p, flops_per_sec, marker='o', color='green')
    plt.xlabel("Number of Processes (p)")
    plt.ylabel("FLOPs/sec")
    plt.title("Performance (FLOPs/sec)")
    plt.grid(True)
    plt.savefig("analysis/flops.png")
    plt.close()

    print("Plots generated in 'analysis/' directory.")

if __name__ == "__main__":
    plot_results()
