"""
Generador de Gráficos Robusto - KNN Paralelo (Tema Blanco)

Versión mejorada con mejor manejo de errores y tipos de datos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def set_white_style():
    """Configura estilo blanco para gráficos"""
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
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.2

def get_version_style(version):
    """Retorna color y marker para cada versión"""
    styles = {
        'v1_naive_p2p': {'color': '#dc2626', 'marker': 's', 'label': 'v1: Naive P2P'},
        'v2_collective_scatter': {'color': '#2563eb', 'marker': '^', 'label': 'v2: Collective Ops'},
        'v3_final_optimized': {'color': '#16a34a', 'marker': 'o', 'label': 'v3: Optimized'}
    }
    return styles.get(version, {'color': 'black', 'marker': 'x', 'label': version})

def load_and_clean_data(csv_file):
    """Carga y limpia los datos"""
    df = pd.read_csv(csv_file)
    
    # Convertir a tipos numéricos
    numeric_cols = ['processes', 'accuracy', 'total_time', 'io_time', 
                    'bcast_time', 'scatter_time', 'compute_time', 'gather_time']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Eliminar filas con NaN
    df = df.dropna()
    
    return df

def aggregate_data(df):
    """Agrupa datos por versión y procesos, promediando múltiples runs"""
    grouped = df.groupby(['version', 'processes'], as_index=False).agg({
        'accuracy': 'mean',
        'total_time': 'mean',
        'io_time': 'mean',
        'bcast_time': 'mean',
        'scatter_time': 'mean',
        'compute_time': 'mean',
        'gather_time': 'mean'
    })
    
    return grouped

def plot_all_comparisons(df, output_dir):
    """Genera todos los gráficos de comparación"""
    
    # Parámetros del dataset
    N = 1437
    M = 360
    d = 64
    total_flops = M * N * 3 * d
    
    # 1. Time Comparison
    try:
        plt.figure(figsize=(12, 7))
        
        for version in df['version'].unique():
            version_data = df[df['version'] == version].sort_values('processes')
            style = get_version_style(version)
            
            plt.plot(version_data['processes'], version_data['total_time'],
                    marker=style['marker'], color=style['color'], 
                    label=style['label'], linewidth=2.5, markersize=10)
        
        plt.xlabel('Number of Processes (p)', fontsize=13)
        plt.ylabel('Total Time (seconds)', fontsize=13)
        plt.title('Execution Time Comparison - Strong Scaling', fontsize=15, fontweight='bold')
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_dir / 'time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: time_comparison.png")
    except Exception as e:
        print(f"✗ Error in time_comparison: {e}")
    
    # 2. Speedup Comparison
    try:
        plt.figure(figsize=(12, 7))
        
        for version in df['version'].unique():
            version_data = df[df['version'] == version].sort_values('processes')
            style = get_version_style(version)
            
            # Calcular speedup
            t_seq = version_data[version_data['processes'] == 1]['total_time'].values[0]
            speedup = t_seq / version_data['total_time']
            
            plt.plot(version_data['processes'], speedup,
                    marker=style['marker'], color=style['color'], 
                    label=style['label'], linewidth=2.5, markersize=10)
        
        # Línea ideal
        ideal_p = np.array([1, 2, 4, 8])
        plt.plot(ideal_p, ideal_p, '--', color='gray', alpha=0.7, 
                 label='Ideal Speedup', linewidth=2)
        
        plt.xlabel('Number of Processes (p)', fontsize=13)
        plt.ylabel('Speedup', fontsize=13)
        plt.title('Speedup Comparison - Strong Scaling', fontsize=15, fontweight='bold')
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_dir / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: speedup_comparison.png")
    except Exception as e:
        print(f"✗ Error in speedup_comparison: {e}")
    
    # 3. Efficiency Comparison
    try:
        plt.figure(figsize=(12, 7))
        
        for version in df['version'].unique():
            version_data = df[df['version'] == version].sort_values('processes')
            style = get_version_style(version)
            
            # Calcular eficiencia
            t_seq = version_data[version_data['processes'] == 1]['total_time'].values[0]
            speedup = t_seq / version_data['total_time']
            efficiency = speedup / version_data['processes']
            
            plt.plot(version_data['processes'], efficiency,
                    marker=style['marker'], color=style['color'], 
                    label=style['label'], linewidth=2.5, markersize=10)
        
        # Línea ideal
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, 
                    label='Ideal Efficiency', linewidth=2)
        
        plt.xlabel('Number of Processes (p)', fontsize=13)
        plt.ylabel('Efficiency', fontsize=13)
        plt.title('Efficiency Comparison - Strong Scaling', fontsize=15, fontweight='bold')
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, alpha=0.4)
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(output_dir / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: efficiency_comparison.png")
    except Exception as e:
        print(f"✗ Error in efficiency_comparison: {e}")
    
    # 4. FLOPs Performance
    try:
        plt.figure(figsize=(12, 7))
        
        for version in df['version'].unique():
            version_data = df[df['version'] == version].sort_values('processes')
            style = get_version_style(version)
            
            # Calcular GFLOPs/sec
            gflops_per_sec = (total_flops / version_data['compute_time']) / 1e9
            
            plt.plot(version_data['processes'], gflops_per_sec,
                    marker=style['marker'], color=style['color'], 
                    label=style['label'], linewidth=2.5, markersize=10)
        
        plt.xlabel('Number of Processes (p)', fontsize=13)
        plt.ylabel('GFLOPs/sec', fontsize=13)
        plt.title('Computational Performance', fontsize=15, fontweight='bold')
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_dir / 'flops_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: flops_performance.png")
    except Exception as e:
        print(f"✗ Error in flops_performance: {e}")
    
    # 5. Time Breakdown (solo v3)
    try:
        v3_data = df[df['version'] == 'v3_final_optimized'].sort_values('processes')
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        processes = v3_data['processes']
        width = 0.6
        
        # Componentes de tiempo
        io_time = v3_data['io_time']
        bcast_time = v3_data['bcast_time']
        scatter_time = v3_data['scatter_time']
        compute_time = v3_data['compute_time']
        gather_time = v3_data['gather_time']
        
        # Stacked bar chart
        p1 = ax.bar(processes, io_time, width, label='I/O', color='#fca5a5')
        p2 = ax.bar(processes, bcast_time, width, bottom=io_time, 
                    label='Broadcast', color='#93c5fd')
        p3 = ax.bar(processes, scatter_time, width, 
                    bottom=io_time+bcast_time, label='Scatter', color='#c4b5fd')
        p4 = ax.bar(processes, compute_time, width, 
                    bottom=io_time+bcast_time+scatter_time, 
                    label='Compute', color='#86efac')
        p5 = ax.bar(processes, gather_time, width, 
                    bottom=io_time+bcast_time+scatter_time+compute_time, 
                    label='Gather', color='#fde68a')
        
        ax.set_xlabel('Number of Processes (p)', fontsize=13)
        ax.set_ylabel('Time (seconds)', fontsize=13)
        ax.set_title('Time Breakdown - v3 Optimized', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.4, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'time_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: time_breakdown.png")
    except Exception as e:
        print(f"✗ Error in time_breakdown: {e}")
    
    # 6. Amdahl Validation (solo v3)
    try:
        v3_data = df[df['version'] == 'v3_final_optimized'].sort_values('processes')
        
        plt.figure(figsize=(12, 7))
        
        # Speedup medido
        t_seq = v3_data[v3_data['processes'] == 1]['total_time'].values[0]
        speedup_measured = t_seq / v3_data['total_time']
        
        plt.plot(v3_data['processes'], speedup_measured, 
                 marker='o', color='#16a34a', label='Measured Speedup (v3)',
                 linewidth=2.5, markersize=10)
        
        # Modelos teóricos de Amdahl
        p_range = np.linspace(1, 8, 100)
        colors = ['#dc2626', '#ea580c', '#16a34a', '#ca8a04', '#7c3aed']
        
        for i, f in enumerate([0.1, 0.2, 0.31, 0.4, 0.5]):
            speedup_amdahl = 1 / (f + (1 - f) / p_range)
            linestyle = '-' if f == 0.31 else '--'
            alpha = 1.0 if f == 0.31 else 0.6
            linewidth = 2.5 if f == 0.31 else 1.5
            label = f'Amdahl f={f}' + (' (expected)' if f == 0.31 else '')
            
            plt.plot(p_range, speedup_amdahl, linestyle=linestyle, 
                    alpha=alpha, linewidth=linewidth, label=label, color=colors[i])
        
        # Speedup ideal
        plt.plot([1, 2, 4, 8], [1, 2, 4, 8], '--', color='gray', 
                 alpha=0.7, label='Ideal (f=0)', linewidth=2)
        
        plt.xlabel('Number of Processes (p)', fontsize=13)
        plt.ylabel('Speedup', fontsize=13)
        plt.title("Amdahl's Law Validation - v3 Optimized", fontsize=15, fontweight='bold')
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_dir / 'amdahl_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: amdahl_validation.png")
    except Exception as e:
        print(f"✗ Error in amdahl_validation: {e}")

def main():
    print("="*70)
    print("Plot Generator - KNN Parallel (White Theme) - Robust Version")
    print("="*70)
    print()
    
    # Leer datos
    csv_file = "results_strong_scaling.csv"
    if not Path(csv_file).exists():
        print(f"ERROR: {csv_file} not found!")
        return
    
    print(f"Reading data from {csv_file}...")
    df_raw = load_and_clean_data(csv_file)
    
    print(f"  Raw data points: {len(df_raw)}")
    print(f"  Versions: {df_raw['version'].unique()}")
    print(f"  Processes: {sorted(df_raw['processes'].unique())}")
    print()
    
    # Agregar datos
    print("Aggregating data (averaging multiple runs)...")
    df = aggregate_data(df_raw)
    
    print(f"  Aggregated data points: {len(df)}")
    print()
    
    # Crear directorio de salida
    output_dir = Path("docs/images_report")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar estilo
    set_white_style()
    
    # Generar gráficos
    print("Generating plots...")
    print()
    
    plot_all_comparisons(df, output_dir)
    
    print()
    print("="*70)
    print("Plot Generation Complete!")
    print("="*70)
    print(f"Plots saved to: {output_dir}/")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
