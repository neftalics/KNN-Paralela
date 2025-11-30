import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def set_custom_style():
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = '#121212'
    plt.rcParams['figure.facecolor'] = '#121212'
    plt.rcParams['grid.color'] = '#444444'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['text.color'] = '#e0e0e0'
    plt.rcParams['axes.labelcolor'] = '#e0e0e0'
    plt.rcParams['xtick.color'] = '#e0e0e0'
    plt.rcParams['ytick.color'] = '#e0e0e0'
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8

def annotate_points(ax, x, y, unit=""):
    for i, txt in enumerate(y):
        ax.annotate(f"{txt:.2f}{unit}", (x.iloc[i], y.iloc[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', color='white', fontsize=9)

def plot_results():
    set_custom_style()
    
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
    
    # Color palette
    primary_color = '#d9f99d' # Light yellow-green
    secondary_color = '#7dd3fc' # Light blue
    tertiary_color = '#fca5a5' # Light red
    
    # 1. Execution Time vs Processes
    plt.figure(figsize=(10, 6))
    plt.plot(p, total_time, marker='o', color=primary_color, label='Total Time')
    plt.plot(p, compute_time, marker='s', color=secondary_color, label='Compute Time')
    plt.plot(p, comm_time, marker='^', color=tertiary_color, label='Communication Time')
    plt.xlabel("Number of Processes (p)")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    annotate_points(plt.gca(), p, total_time, "s")
    plt.savefig("analysis/time_vs_processes.png", dpi=300)
    plt.close()

    # 2. Speedup
    # Speedup = T_sequential / T_parallel
    if 1 in p.values:
        t_seq = df[df["processes"] == 1]["total_time"].values[0]
        speedup = t_seq / total_time
        
        plt.figure(figsize=(10, 6))
        plt.plot(p, speedup, marker='o', color=primary_color, label='Measured Speedup')
        plt.plot(p, p, '--', color='gray', alpha=0.5, label='Ideal Speedup') 
        
        # Fill area below curve
        plt.fill_between(p, speedup, color=primary_color, alpha=0.1)
        
        plt.xlabel("Number of Processes (p)")
        plt.ylabel("Speedup")
        plt.title("Speedup Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)
        annotate_points(plt.gca(), p, speedup, "x")
        plt.savefig("analysis/speedup.png", dpi=300)
        plt.close()

    # 3. FLOPs (Estimación)
    n_samples = 1797
    n_test = int(n_samples * 0.2) 
    n_train = n_samples - n_test 
    d = 64
    ops_per_distance = d * 3 
    total_flops = n_test * n_train * ops_per_distance
    
    flops_per_sec = total_flops / total_time
    
    plt.figure(figsize=(10, 6))
    plt.plot(p, flops_per_sec, marker='o', color=secondary_color)
    plt.fill_between(p, flops_per_sec, color=secondary_color, alpha=0.1)
    plt.xlabel("Number of Processes (p)")
    plt.ylabel("FLOPs/sec")
    plt.title("Performance (FLOPs/sec)")
    plt.grid(True, alpha=0.3)
    # Annotate with scientific notation for FLOPs
    for i, val in enumerate(flops_per_sec):
        plt.annotate(f"{val:.1e}", (p.iloc[i], val), 
                     textcoords="offset points", xytext=(0,10), ha='center', color='white', fontsize=9)
    plt.savefig("analysis/flops.png", dpi=300)
    plt.close()

    # 4. Efficiency
    if 1 in p.values:
        efficiency = speedup / p
        
        plt.figure(figsize=(10, 6))
        plt.plot(p, efficiency, marker='o', color=tertiary_color, label='Efficiency')
        plt.plot(p, [1]*len(p), '--', color='gray', alpha=0.5, label='Ideal Efficiency (1.0)')
        plt.fill_between(p, efficiency, color=tertiary_color, alpha=0.1)
        plt.xlabel("Number of Processes (p)")
        plt.ylabel("Efficiency")
        plt.title("Efficiency Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)
        annotate_points(plt.gca(), p, efficiency)
        plt.savefig("analysis/efficiency.png", dpi=300)
        plt.close()

    # 5. Scalability (Time vs N)
    try:
        df_scale = pd.read_csv("scaling_log.csv")
        n = df_scale["n_samples"]
        t_scale = df_scale["total_time"]
        
        plt.figure(figsize=(10, 6))
        plt.plot(n, t_scale, marker='o', color=primary_color)
        plt.fill_between(n, t_scale, color=primary_color, alpha=0.1)
        plt.xlabel("Dataset Size (N)")
        plt.ylabel("Execution Time (s)")
        plt.title(f"Scalability with Problem Size (p={df_scale['processes'].iloc[0]})")
        plt.grid(True, alpha=0.3)
        annotate_points(plt.gca(), n, t_scale, "s")
        plt.savefig("analysis/scalability_n.png", dpi=300)
        plt.close()
    except FileNotFoundError:
        print("scaling_log.csv not found. Skipping scalability plot.")

    print("Plots generated in 'analysis/' directory with Dark Theme.")

if __name__ == "__main__":
    plot_results()
