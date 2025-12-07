"""
Calculadora de FLOPs - KNN Paralelo

Propósito:
    Calcular el número de operaciones de punto flotante (FLOPs) reales
    basados en las dimensiones del dataset para validar gráficos de rendimiento.

Derivación:
    Distancia Euclidiana: d(x,y) = √(Σ(xᵢ-yᵢ)²)
    
    Operaciones por distancia:
    - d restas: (xᵢ - yᵢ)
    - d multiplicaciones: (xᵢ - yᵢ)²
    - (d-1) sumas: Σ
    - 1 raíz cuadrada: √
    
    Total ≈ 3d FLOPs por distancia (para d grande)
    
    FLOPs totales del algoritmo:
    - Para clasificar 1 punto de prueba: N × 3d
    - Para M puntos de prueba: M × N × 3d

Dataset:
    - N = 1437 (puntos de entrenamiento)
    - M = 360 (puntos de prueba)
    - d = 64 (dimensiones)
    - k = 3 (vecinos)

Output:
    - FLOPs totales
    - MFLOPs
    - GFLOPs
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def calculate_flops(N, M, d, k=3):
    """
    Calcula FLOPs para el algoritmo KNN.
    
    Args:
        N: Número de puntos de entrenamiento
        M: Número de puntos de prueba
        d: Dimensiones del espacio de características
        k: Número de vecinos (no afecta significativamente los FLOPs)
    
    Returns:
        dict con diferentes métricas de FLOPs
    """
    # FLOPs por cálculo de distancia euclidiana
    flops_per_distance = 3 * d  # resta + mult + suma (aproximación)
    
    # FLOPs para clasificar un punto de prueba
    flops_per_test_point = N * flops_per_distance
    
    # FLOPs totales para todo el dataset de prueba
    total_flops = M * N * flops_per_distance
    
    # Operaciones adicionales (argpartition, bincount, etc.)
    # Son O(N) o O(k), despreciables comparado con O(N×M×d)
    overhead_flops = M * (N + k)  # Aproximación conservadora
    
    total_flops_with_overhead = total_flops + overhead_flops
    
    return {
        'flops_per_distance': flops_per_distance,
        'flops_per_test_point': flops_per_test_point,
        'total_flops': total_flops,
        'overhead_flops': overhead_flops,
        'total_flops_with_overhead': total_flops_with_overhead,
        'mflops': total_flops / 1e6,
        'gflops': total_flops / 1e9
    }

def load_dataset_dimensions():
    """Carga el dataset real y obtiene sus dimensiones"""
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
    
    N, d = X_train.shape
    M = len(X_test)
    
    return N, M, d

def calculate_gflops_from_time(total_flops, execution_time):
    """Calcula GFLOPs/sec dado el tiempo de ejecución"""
    return (total_flops / execution_time) / 1e9

def main():
    print("=" * 70)
    print("FLOPs Calculator - KNN Parallel")
    print("=" * 70)
    print()
    
    # Cargar dimensiones del dataset real
    print("Loading dataset to get dimensions...")
    N, M, d = load_dataset_dimensions()
    
    print(f"Dataset dimensions:")
    print(f"  N (training points): {N}")
    print(f"  M (test points):     {M}")
    print(f"  d (dimensions):      {d}")
    print()
    
    # Calcular FLOPs
    k = 3
    flops = calculate_flops(N, M, d, k)
    
    print("=" * 70)
    print("FLOPs Calculation")
    print("=" * 70)
    print()
    
    print("Theoretical Model:")
    print(f"  FLOPs per distance:        {flops['flops_per_distance']:,}")
    print(f"  FLOPs per test point:      {flops['flops_per_test_point']:,}")
    print()
    
    print("Total FLOPs:")
    print(f"  Core computation:          {flops['total_flops']:,}")
    print(f"  Overhead (sorting, etc):   {flops['overhead_flops']:,}")
    print(f"  Total with overhead:       {flops['total_flops_with_overhead']:,}")
    print()
    
    print("Formatted:")
    print(f"  MFLOPs:                    {flops['mflops']:.2f}")
    print(f"  GFLOPs:                    {flops['gflops']:.4f}")
    print()
    
    # Ejemplos de rendimiento
    print("=" * 70)
    print("Performance Examples (GFLOPs/sec)")
    print("=" * 70)
    print()
    
    example_times = [5.0, 2.5, 1.25, 0.625]  # Tiempos de ejemplo
    example_processes = [1, 2, 4, 8]
    
    print("Example execution times and corresponding performance:")
    print(f"{'Processes':<12} {'Time (sec)':<15} {'GFLOPs/sec':<15} {'Speedup':<10}")
    print("-" * 70)
    
    base_time = example_times[0]
    for p, t in zip(example_processes, example_times):
        gflops_per_sec = calculate_gflops_from_time(flops['total_flops'], t)
        speedup = base_time / t
        print(f"{p:<12} {t:<15.4f} {gflops_per_sec:<15.6f} {speedup:<10.2f}x")
    
    print()
    
    # Validación teórica
    print("=" * 70)
    print("Theoretical Validation")
    print("=" * 70)
    print()
    
    print("Formula: FLOPs = M × N × 3d")
    print(f"  M = {M}")
    print(f"  N = {N}")
    print(f"  d = {d}")
    print(f"  3d = {3*d}")
    print()
    print(f"  {M} × {N} × {3*d} = {flops['total_flops']:,}")
    print()
    
    # Información para el reporte
    print("=" * 70)
    print("For LaTeX Report")
    print("=" * 70)
    print()
    print("Use this in your report:")
    print(f"\\text{{FLOPs}}_{{\\text{{total}}}} = M \\times N \\times 3d")
    print(f"  = {M} \\times {N} \\times {3*d}")
    print(f"  = {flops['total_flops']:,}")
    print(f"  \\approx {flops['mflops']:.1f} \\text{{ MFLOPs}}")
    print()

if __name__ == "__main__":
    main()
