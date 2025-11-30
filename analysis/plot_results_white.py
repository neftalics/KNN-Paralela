import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def set_white_style():
    """Set white background style for report"""
    plt.style.use('default')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['grid.color'] = '#cccccc'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.2

def annotate_points(ax, x, y, unit=""):
    """Annotate points on plot with black text"""
    for i, txt in enumerate(y):
        ax.annotate(f"{txt:.2f}{unit}", (x.iloc[i], y.iloc[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', 
                    color='black', fontsize=9, fontweight='bold')

def plot_results_white():
    """Generate white background plots for report"""
    set_white_style()
    
    try:
        df = pd.read_csv("results_log.csv")
    except FileNotFoundError:
        print("results_log.csv not found. Run experiments first.")
        return

    # Sort by number of processes
    df = df.sort_values(by="processes")
    
    p = df["processes"]
    total_time = df["total_time"]
    compute_time = df["compute_time"]
    comm_time = df["comm_time"]
    
    # Professional color palette for white background
    primary_color = '#2563eb'    # Blue
    secondary_color = '#16a34a'  # Green
    tertiary_color = '#dc2626'   # Red
    
    # 1. Execution Time vs Processes
    plt.figure(figsize=(10, 6))
    plt.plot(p, total_time, marker='o', color=primary_color, label='Total Time', linewidth=2.5)
    plt.plot(p, compute_time, marker='s', color=secondary_color, label='Compute Time', linewidth=2.5)
    plt.plot(p, comm_time, marker='^', color=tertiary_color, label='Communication Time', linewidth=2.5)
    plt.xlabel("Number of Processes (p)", fontsize=12, fontweight='bold')
    plt.ylabel("Time (seconds)", fontsize=12, fontweight='bold')
    plt.title("Execution Time Analysis", fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    annotate_points(plt.gca(), p, total_time, "s")
    plt.tight_layout()
    plt.savefig("docs/images_report/time_vs_processes.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 2. Speedup
    if 1 in p.values:
        t_seq = df[df["processes"] == 1]["total_time"].values[0]
        speedup = t_seq / total_time
        
        plt.figure(figsize=(10, 6))
        plt.plot(p, speedup, marker='o', color=primary_color, label='Measured Speedup', linewidth=2.5)
        plt.plot(p, p, '--', color='#6b7280', alpha=0.7, linewidth=2, label='Ideal Speedup') 
        
        # Fill area below curve
        plt.fill_between(p, speedup, color=primary_color, alpha=0.15)
        
        plt.xlabel("Number of Processes (p)", fontsize=12, fontweight='bold')
        plt.ylabel("Speedup", fontsize=12, fontweight='bold')
        plt.title("Speedup Analysis", fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        annotate_points(plt.gca(), p, speedup, "x")
        plt.tight_layout()
        plt.savefig("docs/images_report/speedup.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # 3. FLOPs (Estimation)
    n_samples = 1797
    n_test = int(n_samples * 0.2) 
    n_train = n_samples - n_test 
    d = 64
    ops_per_distance = d * 3 
    total_flops = n_test * n_train * ops_per_distance
    
    flops_per_sec = total_flops / total_time
    
    plt.figure(figsize=(10, 6))
    plt.plot(p, flops_per_sec, marker='o', color=secondary_color, linewidth=2.5)
    plt.fill_between(p, flops_per_sec, color=secondary_color, alpha=0.15)
    plt.xlabel("Number of Processes (p)", fontsize=12, fontweight='bold')
    plt.ylabel("FLOPs/sec", fontsize=12, fontweight='bold')
    plt.title("Performance (FLOPs/sec)", fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    # Annotate with scientific notation for FLOPs
    for i, val in enumerate(flops_per_sec):
        plt.annotate(f"{val:.1e}", (p.iloc[i], val), 
                     textcoords="offset points", xytext=(0,10), ha='center', 
                     color='black', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig("docs/images_report/flops.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 4. Efficiency
    if 1 in p.values:
        efficiency = speedup / p
        
        plt.figure(figsize=(10, 6))
        plt.plot(p, efficiency, marker='o', color=tertiary_color, label='Efficiency', linewidth=2.5)
        plt.plot(p, [1]*len(p), '--', color='#6b7280', alpha=0.7, linewidth=2, label='Ideal Efficiency (1.0)')
        plt.fill_between(p, efficiency, color=tertiary_color, alpha=0.15)
        plt.xlabel("Number of Processes (p)", fontsize=12, fontweight='bold')
        plt.ylabel("Efficiency", fontsize=12, fontweight='bold')
        plt.title("Efficiency Analysis", fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        annotate_points(plt.gca(), p, efficiency)
        plt.tight_layout()
        plt.savefig("docs/images_report/efficiency.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # 5. Scalability (Time vs N)
    try:
        df_scale = pd.read_csv("scaling_log.csv")
        n = df_scale["n_samples"]
        t_scale = df_scale["total_time"]
        
        plt.figure(figsize=(10, 6))
        plt.plot(n, t_scale, marker='o', color=primary_color, linewidth=2.5)
        plt.fill_between(n, t_scale, color=primary_color, alpha=0.15)
        plt.xlabel("Dataset Size (N)", fontsize=12, fontweight='bold')
        plt.ylabel("Execution Time (s)", fontsize=12, fontweight='bold')
        plt.title(f"Scalability with Problem Size (p={df_scale['processes'].iloc[0]})", 
                  fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        annotate_points(plt.gca(), n, t_scale, "s")
        plt.tight_layout()
        plt.savefig("docs/images_report/scalability_n.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    except FileNotFoundError:
        print("scaling_log.csv not found. Skipping scalability plot.")

    print("✓ White background plots generated in 'docs/images_report/' directory.")

if __name__ == "__main__":
    plot_results_white()
