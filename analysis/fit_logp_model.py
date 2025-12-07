"""
Ajuste del Modelo LogP - KNN Paralelo

Propósito:
    Ajustar los parámetros α (latencia) y β (inverso del ancho de banda)
    del modelo LogP a partir de los datos experimentales de comunicación.

Modelo Teórico:
    T_comm(p) ≈ log₂(p) × (α + N×β)
    
    donde:
    - p = número de procesos
    - α = latencia por mensaje (segundos)
    - β = tiempo por byte transmitido (segundos/byte)
    - N = tamaño del mensaje en bytes

Estrategia:
    1. Leer CSV con resultados experimentales
    2. Extraer tiempos de comunicación (bcast + gather)
    3. Ajustar curva usando scipy.optimize.curve_fit
    4. Generar gráfico de ajuste teórico vs experimental
    5. Reportar valores de α y β

Output:
    - Parámetros α y β ajustados
    - Gráfico de ajuste (logp_fit.png)
    - Métricas de bondad de ajuste (R²)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

def logp_model(p, alpha, beta, N):
    """
    Modelo LogP para tiempo de comunicación.
    
    Args:
        p: número de procesos
        alpha: latencia (segundos)
        beta: tiempo por byte (segundos/byte)
        N: tamaño del mensaje (bytes)
    
    Returns:
        Tiempo de comunicación estimado
    """
    return np.log2(p) * (alpha + N * beta)

def fit_logp_parameters(processes, comm_times, message_size):
    """
    Ajusta parámetros α y β del modelo LogP.
    
    Args:
        processes: array de número de procesos
        comm_times: array de tiempos de comunicación medidos
        message_size: tamaño del mensaje en bytes
    
    Returns:
        tuple (alpha, beta, r_squared)
    """
    # Función de ajuste con N fijo
    def model_func(p, alpha, beta):
        return logp_model(p, alpha, beta, message_size)
    
    # Ajustar curva
    # Bounds: alpha > 0, beta > 0
    try:
        params, covariance = curve_fit(
            model_func, 
            processes, 
            comm_times,
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=10000
        )
        
        alpha, beta = params
        
        # Calcular R²
        predictions = model_func(processes, alpha, beta)
        ss_res = np.sum((comm_times - predictions) ** 2)
        ss_tot = np.sum((comm_times - np.mean(comm_times)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return alpha, beta, r_squared
    
    except Exception as e:
        print(f"Error in curve fitting: {e}")
        return None, None, None

def plot_logp_fit(processes, comm_times, alpha, beta, message_size, 
                  version_name, output_file):
    """
    Genera gráfico del ajuste del modelo LogP.
    """
    plt.figure(figsize=(10, 6))
    
    # Datos experimentales
    plt.scatter(processes, comm_times, s=100, alpha=0.7, 
                label='Experimental Data', color='blue', zorder=3)
    
    # Modelo ajustado
    p_range = np.linspace(1, max(processes), 100)
    fitted_times = logp_model(p_range, alpha, beta, message_size)
    plt.plot(p_range, fitted_times, 'r-', linewidth=2, 
             label=f'LogP Model Fit', zorder=2)
    
    # Configuración del gráfico
    plt.xlabel('Number of Processes (p)', fontsize=12)
    plt.ylabel('Communication Time (seconds)', fontsize=12)
    plt.title(f'LogP Model Fit - {version_name}\n' + 
              f'α = {alpha:.6f} sec, β = {beta:.9f} sec/byte', 
              fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Anotaciones
    textstr = f'Model: T_comm ≈ log₂(p) × (α + N×β)\n'
    textstr += f'α (latency) = {alpha*1000:.3f} ms\n'
    textstr += f'β (per byte) = {beta*1e9:.3f} ns/byte\n'
    textstr += f'Message size = {message_size/1024:.1f} KB'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    plt.close()

def estimate_message_size():
    """
    Estima el tamaño del mensaje para broadcast.
    
    Para nuestro dataset:
    - X_train: 1437 × 64 × 8 bytes (float64) = 735,744 bytes
    - y_train: 1437 × 8 bytes (int64) = 11,496 bytes
    Total ≈ 747,240 bytes ≈ 730 KB
    """
    N_train = 1437
    d = 64
    bytes_per_float = 8  # float64
    
    X_train_size = N_train * d * bytes_per_float
    y_train_size = N_train * bytes_per_float
    
    total_size = X_train_size + y_train_size
    
    return total_size

def main():
    print("=" * 70)
    print("LogP Model Fitting - KNN Parallel")
    print("=" * 70)
    print()
    
    # Verificar que existe el archivo de resultados
    csv_file = "results_strong_scaling.csv"
    if not Path(csv_file).exists():
        print(f"ERROR: {csv_file} not found!")
        print("Please run the experiments first: python scripts/run_experiments.py")
        return
    
    # Leer datos
    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Estimar tamaño del mensaje
    message_size = estimate_message_size()
    print(f"Estimated message size: {message_size:,} bytes ({message_size/1024:.1f} KB)")
    print()
    
    # Procesar cada versión
    versions = df['version'].unique()
    
    for version in versions:
        print("=" * 70)
        print(f"Fitting LogP model for: {version}")
        print("=" * 70)
        
        # Filtrar datos de esta versión
        version_data = df[df['version'] == version]
        
        # Agrupar por número de procesos y promediar
        grouped = version_data.groupby('processes').agg({
            'bcast_time': 'mean',
            'gather_time': 'mean'
        }).reset_index()
        
        # Tiempo total de comunicación colectiva
        grouped['comm_time'] = grouped['bcast_time'] + grouped['gather_time']
        
        processes = grouped['processes'].values
        comm_times = grouped['comm_time'].values
        
        print(f"\nData points:")
        for p, t in zip(processes, comm_times):
            print(f"  p={p}: T_comm={t:.4f} sec")
        
        # Ajustar modelo
        print(f"\nFitting LogP model...")
        alpha, beta, r_squared = fit_logp_parameters(
            processes, comm_times, message_size
        )
        
        if alpha is not None:
            print(f"\nResults:")
            print(f"  α (latency):        {alpha:.6f} sec = {alpha*1000:.3f} ms")
            print(f"  β (per byte):       {beta:.9f} sec/byte = {beta*1e9:.3f} ns/byte")
            print(f"  R² (goodness):      {r_squared:.4f}")
            
            # Interpretación
            print(f"\nInterpretation:")
            bandwidth_mbps = (1 / beta) / 1e6 if beta > 0 else 0
            print(f"  Effective bandwidth: {bandwidth_mbps:.2f} MB/sec")
            print(f"  Latency overhead:    {alpha*1000:.3f} ms per collective operation")
            
            # Generar gráfico
            output_file = f"docs/images_report/logp_fit_{version}.png"
            plot_logp_fit(processes, comm_times, alpha, beta, message_size,
                         version, output_file)
            
        else:
            print("  ERROR: Could not fit model")
        
        print()
    
    print("=" * 70)
    print("LogP Model Fitting Complete!")
    print("=" * 70)
    print()
    print("Plots saved to: docs/images_report/")
    print()

if __name__ == "__main__":
    main()
