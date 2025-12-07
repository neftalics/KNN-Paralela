"""
Gráficas Comparativas Simples: Digits vs MNIST
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Leer datos
df_digits = pd.read_csv('results_strong_scaling.csv')
df_mnist = pd.read_csv('results_mnist.csv')

# Filtrar y promediar
df_d = df_digits[df_digits['version'] == 'v3_final_optimized'].copy()
df_d['processes'] = pd.to_numeric(df_d['processes'])
df_d['total_time'] = pd.to_numeric(df_d['total_time'])
df_d['compute_time'] = pd.to_numeric(df_d['compute_time'])
df_d['bcast_time'] = pd.to_numeric(df_d['bcast_time'])
df_d['scatter_time'] = pd.to_numeric(df_d['scatter_time'])
df_d['gather_time'] = pd.to_numeric(df_d['gather_time'])

df_d_avg = df_d.groupby('processes').mean(numeric_only=True).reset_index()

df_m = df_mnist.copy()
df_m['processes'] = pd.to_numeric(df_m['processes'])
df_m['total_time'] = pd.to_numeric(df_m['total_time'])
df_m['compute_time'] = pd.to_numeric(df_m['compute_time'])
df_m['bcast_time'] = pd.to_numeric(df_m['bcast_time'])
df_m['scatter_time'] = pd.to_numeric(df_m['scatter_time'])
df_m['gather_time'] = pd.to_numeric(df_m['gather_time'])

output_dir = Path('docs/images_report')
output_dir.mkdir(parents=True, exist_ok=True)

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# 1. Speedup Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Digits
p_d = df_d_avg['processes'].values
t_d = df_d_avg['total_time'].values
speedup_d = t_d[0] / t_d

ax1.plot(p_d, speedup_d, 'o-', color='#dc2626', linewidth=2.5, markersize=10, label='Digits')
ax1.plot([1, 2, 4, 8], [1, 2, 4, 8], '--', color='gray', alpha=0.7, linewidth=2, label='Ideal')
ax1.set_xlabel('Procesos (p)', fontsize=13)
ax1.set_ylabel('Speedup', fontsize=13)
ax1.set_title('Digits: Speedup Limitado\n(Dataset Pequeño)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 9)

# MNIST
p_m = df_m['processes'].values
t_m = df_m['total_time'].values
speedup_m = t_m[0] / t_m

ax2.plot(p_m, speedup_m, 's-', color='#16a34a', linewidth=2.5, markersize=10, label='MNIST')
ax2.plot([1, 2, 4, 8], [1, 2, 4, 8], '--', color='gray', alpha=0.7, linewidth=2, label='Ideal')
ax2.set_xlabel('Procesos (p)', fontsize=13)
ax2.set_ylabel('Speedup', fontsize=13)
ax2.set_title('MNIST: Speedup Excelente\n(Dataset Grande)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 9)

plt.tight_layout()
plt.savefig(output_dir / 'comparison_speedup_digits_vs_mnist.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ comparison_speedup_digits_vs_mnist.png")

# 2. Ratio Cómputo/Comunicación
fig, ax = plt.subplots(figsize=(12, 7))

comm_d = df_d_avg['bcast_time'] + df_d_avg['scatter_time'] + df_d_avg['gather_time']
comp_d = df_d_avg['compute_time']
ratio_d = comp_d / comm_d
ratio_d = ratio_d.replace([np.inf, -np.inf], 100)

comm_m = df_m['bcast_time'] + df_m['scatter_time'] + df_m['gather_time']
comp_m = df_m['compute_time']
ratio_m = comp_m / comm_m
ratio_m = ratio_m.replace([np.inf, -np.inf], 100)

ax.plot(p_d, ratio_d, 'o-', color='#dc2626', linewidth=2.5, markersize=10, label='Digits (pequeño)')
ax.plot(p_m, ratio_m, 's-', color='#16a34a', linewidth=2.5, markersize=10, label='MNIST (grande)')
ax.axhline(y=5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Umbral mínimo (R=5)')
ax.axhline(y=20, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Umbral óptimo (R=20)')

ax.set_xlabel('Procesos (p)', fontsize=13)
ax.set_ylabel('Ratio Cómputo/Comunicación', fontsize=13)
ax.set_title('Criterio de Decisión: Ratio Cómputo/Comunicación', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

ax.text(6, 2, 'Digits: Comunicación\ndomina (R<5)\n→ Usar p=1', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='#fca5a5', alpha=0.7))
ax.text(6, 25, 'MNIST: Cómputo\ndomina (R>20)\n→ Usar p=8', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#86efac', alpha=0.7))

plt.tight_layout()
plt.savefig(output_dir / 'comparison_ratio_compute_comm.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ comparison_ratio_compute_comm.png")

# 3. Eficiencia
fig, ax = plt.subplots(figsize=(12, 7))

eff_d = speedup_d / p_d
eff_m = speedup_m / p_m

ax.plot(p_d, eff_d, 'o-', color='#dc2626', linewidth=2.5, markersize=10, label='Digits')
ax.plot(p_m, eff_m, 's-', color='#16a34a', linewidth=2.5, markersize=10, label='MNIST')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Ideal')
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Umbral 50%')

ax.set_xlabel('Procesos (p)', fontsize=13)
ax.set_ylabel('Eficiencia', fontsize=13)
ax.set_title('Eficiencia: Digits vs MNIST', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(output_dir / 'comparison_efficiency_digits_vs_mnist.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ comparison_efficiency_digits_vs_mnist.png")

print("\n" + "="*70)
print("RESUMEN COMPARATIVO")
print("="*70)
print("\nDigits (N×M = 517,320):")
for i in range(len(p_d)):
    print(f"  p={int(p_d[i])}: Speedup={speedup_d[i]:.2f}x, Eficiencia={eff_d[i]:.1%}, Ratio C/C={ratio_d.iloc[i]:.2f}")

print("\nMNIST (N×M = 600,000,000):")
for i in range(len(p_m)):
    print(f"  p={int(p_m[i])}: Speedup={speedup_m[i]:.2f}x, Eficiencia={eff_m[i]:.1%}, Ratio C/C={ratio_m.iloc[i]:.2f}")

print("\n" + "="*70)
print("CONCLUSIÓN")
print("="*70)
print("✓ Digits:  Usar p=1 (ratio C/C < 5, overhead domina)")
print("✓ MNIST:   Usar p=4-8 (ratio C/C > 20, cómputo domina)")
print("="*70)
print("\n✓ Todas las gráficas generadas exitosamente!")
