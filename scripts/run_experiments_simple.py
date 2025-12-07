"""
Script Simplificado de Experimentación - KNN Paralelo

Ejecuta experimentos de forma más controlada para evitar problemas de encoding.
"""

import subprocess
import os
import sys
import time
from pathlib import Path

def run_single_experiment(version, processes):
    """Ejecuta un experimento individual y retorna si fue exitoso"""
    cmd = ["mpiexec", "-n", str(processes), "python", f"src/{version}.py"]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            check=True,
            timeout=120,
            encoding='utf-8',
            errors='replace'  # Reemplazar caracteres problemáticos
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def main():
    print("="*60)
    print("KNN Parallel - Benchmarking (Simplified)")
    print("="*60)
    print()
    
    # Configuración
    VERSIONS = ["v1_naive_p2p", "v2_collective_scatter", "v3_final_optimized"]
    PROCESSES = [1, 2, 4, 8]
    RUNS = 3  # Reducido a 3 para ser más rápido
    
    # Crear CSV
    csv_file = "results_strong_scaling.csv"
    with open(csv_file, "w", encoding='utf-8') as f:
        f.write("version,processes,accuracy,total_time,io_time,bcast_time,"
                "scatter_time,compute_time,gather_time\n")
    
    print(f"Created: {csv_file}")
    print()
    
    total = len(VERSIONS) * len(PROCESSES) * RUNS
    current = 0
    success_count = 0
    fail_count = 0
    
    for version in VERSIONS:
        print(f"\n{'='*60}")
        print(f"Version: {version}")
        print(f"{'='*60}")
        
        for p in PROCESSES:
            print(f"\n  Testing with {p} processes...")
            
            for run in range(1, RUNS + 1):
                current += 1
                print(f"    Run {run}/{RUNS} ({current}/{total})...", end=" ", flush=True)
                
                success, output = run_single_experiment(version, p)
                
                if success:
                    print("OK")
                    success_count += 1
                else:
                    print(f"FAIL: {output[:50]}")
                    fail_count += 1
                
                time.sleep(0.5)
    
    print()
    print("="*60)
    print("Benchmarking Complete!")
    print("="*60)
    print(f"Total experiments: {total}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Results saved to: {csv_file}")
    print()

if __name__ == "__main__":
    main()
