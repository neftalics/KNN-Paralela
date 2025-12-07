"""
Pipeline Completo de Análisis - KNN Paralelo

Ejecuta todo el análisis después de los experimentos:
1. Calcula FLOPs teóricos
2. Ajusta modelo LogP
3. Genera gráficas (tema blanco para reporte)
4. Genera resumen de resultados
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path, description):
    """Ejecuta un script y reporta el resultado"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='replace'
        )
        
        print(result.stdout)
        print(f"✓ {description} completado exitosamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error en {description}:")
        print(e.stderr)
        return False

def main():
    print("="*70)
    print("Pipeline de Análisis Completo - KNN Paralelo")
    print("="*70)
    print()
    
    # Verificar que existen los datos
    csv_file = Path("results_strong_scaling.csv")
    if not csv_file.exists():
        print("ERROR: results_strong_scaling.csv no encontrado!")
        print("Por favor ejecuta primero: python scripts/run_experiments_simple.py")
        sys.exit(1)
    
    print(f"✓ Datos encontrados: {csv_file}")
    print()
    
    # Pipeline de análisis
    steps = [
        ("analysis/calculate_flops.py", "Cálculo de FLOPs Teóricos"),
        ("analysis/fit_logp_model.py", "Ajuste del Modelo LogP"),
        ("analysis/plot_results_white_robust.py", "Generación de Gráficas (Tema Blanco)")
    ]
    
    results = {}
    
    for script, description in steps:
        success = run_script(script, description)
        results[description] = success
    
    # Resumen final
    print()
    print("="*70)
    print("Resumen del Pipeline")
    print("="*70)
    
    all_success = all(results.values())
    
    for description, success in results.items():
        status = "✓ OK" if success else "✗ FAIL"
        print(f"{status}: {description}")
    
    print()
    
    if all_success:
        print("="*70)
        print("¡Pipeline completado exitosamente!")
        print("="*70)
        print()
        print("Gráficas generadas en: docs/images_report/")
        print()
        print("Gráficas disponibles:")
        print("  - time_comparison.png")
        print("  - speedup_comparison.png")
        print("  - efficiency_comparison.png")
        print("  - flops_performance.png")
        print("  - time_breakdown.png")
        print("  - amdahl_validation.png")
        print("  - logp_fit_v1_naive_p2p.png")
        print("  - logp_fit_v2_collective_scatter.png")
        print("  - logp_fit_v3_final_optimized.png")
        print()
        print("Siguiente paso:")
        print("  Integrar las gráficas en docs/reporte_proyecto.tex")
        print()
    else:
        print("✗ Algunos pasos fallaron. Revisa los errores arriba.")
        sys.exit(1)

if __name__ == "__main__":
    main()
