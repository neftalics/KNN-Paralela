# Paralelización de KNN con MPI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MPI](https://img.shields.io/badge/MPI-mpi4py-green.svg)](https://mpi4py.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Implementación paralela del algoritmo **K-Nearest Neighbors (KNN)** utilizando **MPI (Message Passing Interface)** en Python. Este proyecto demuestra técnicas de paralelización para algoritmos de Machine Learning mediante el modelo SPMD (Single Program, Multiple Data).

## 👥 Autores

- **Fabián Alvarado Ramos**
- **Eduardo Miguel Salas Palacios**
- **Neftalí Calixto Rojas**

**Institución:** Proyecto Universitario de Computación Paralela

---

## 📋 Descripción del Proyecto

El algoritmo KNN tiene una complejidad computacional de O(N×M), donde N es el número de puntos de entrenamiento y M el número de puntos de prueba. Este proyecto paraleliza el cálculo de distancias distribuyendo los puntos de prueba entre múltiples procesos usando MPI.

### Características Principales

- ✅ Implementación secuencial de referencia
- ✅ **Tres versiones con narrativa de optimización clara**:
  - **v1 (Naive P2P)**: Demuestra cuello de botella de latencia (α)
  - **v2 (Collective Ops)**: Optimiza ancho de banda con operaciones colectivas (β)
  - **v3 (Optimized)**: Vectorización completa para minimizar fracción serial (f)
- ✅ Modelo maestro-trabajador con topología DAG
- ✅ **Strong Scaling**: N fijo (1437), p variable (1, 2, 4, 8)
- ✅ Análisis completo de rendimiento (Speedup, Eficiencia, FLOPs)
- ✅ **Validación teórica**: LogP, Amdahl, PRAM

---

## 🔬 Marco Teórico

### Modelo LogP (Comunicación)

El tiempo de comunicación en operaciones colectivas sigue el modelo LogP:

```
T_comm(p) ≈ log₂(p) × (α + N×β)
```

donde:
- **α** = latencia por mensaje (segundos)
- **β** = tiempo por byte (segundos/byte)
- **p** = número de procesos
- **N** = tamaño del mensaje (bytes)

**Validación**: El script `fit_logp_model.py` ajusta α y β a partir de datos experimentales.

### Ley de Amdahl (Escalabilidad)

El speedup máximo está limitado por la fracción serial:

```
Speedup_max = 1 / (f + (1-f)/p)
```

donde:
- **f** = fracción serial del código (≈ 0.31 en v3)
- **p** = número de procesos

**Implicación**: Con f=0.31, el speedup máximo teórico es ~3.23x (incluso con infinitos procesos).

### Formalismo PRAM

El código v3 usa comentarios estilo PRAM (Parallel Random Access Machine):
- `BEGIN PARALLEL SECTION`: Inicio de región paralela
- `SYNC`: Punto de sincronización
- Modelo CREW (Concurrent Read, Exclusive Write)

---

## 📖 Historia de Optimización

### v1: Naive Point-to-Point (Línea Base Ineficiente)

**Problema**: Comunicación punto a punto bloqueante en bucles.

```python
# Master envía cada punto individualmente
for worker in range(1, size):
    for test_point in X_test_chunks[worker]:
        comm.send(test_point, dest=worker)  # Alta latencia
```

**Cuello de botella**: Latencia (α) dominante. Cada `send`/`recv` incurre en overhead de latencia.

**Modelo de costo**: `T_comm ≈ M × p × α`

### v2: Collective Operations (Mejora de Comunicación)

**Solución**: Usar operaciones colectivas MPI.

```python
# Operaciones colectivas optimizadas
X_train = comm.bcast(X_train, root=0)      # Broadcast
local_X_test = comm.scatter(X_test_chunks, root=0)  # Scatter
all_predictions = comm.gather(local_predictions, root=0)  # Gather
```

**Mejora**: Comunicación en árbol logarítmico reduce latencia.

**Modelo de costo**: `T_comm ≈ log(p) × (α + N×β)`

**Limitación**: Aún usa bucles Python para cálculo de distancias (no vectorizado).

### v3: Final Optimized (Vectorización Completa)

**Solución**: Vectorización NumPy completa.

```python
# Cálculo vectorizado de distancias (sin bucles Python)
distances = np.sqrt(np.sum((X_train - test_point)**2, axis=1))
k_indices = np.argpartition(distances, k)[:k]
```

**Mejora**: Minimiza fracción serial (f ≈ 0.31).

**Optimizaciones**:
- Operaciones vectorizadas NumPy (aprovecha BLAS/LAPACK)
- `argpartition` en lugar de `argsort` (O(N) vs O(N log N))
- Timing detallado para validar Amdahl

---

## 📁 Estructura del Repositorio

```
Paralela-Proyecto/
├── src/                          # Código fuente
│   ├── knn_sequential.py         # Versión secuencial (baseline)
│   ├── v1_naive_p2p.py          # v1: Comunicación P2P ineficiente
│   ├── v2_collective_scatter.py  # v2: Operaciones colectivas
│   ├── v3_final_optimized.py     # v3: Vectorización completa
│   └── old/                      # Versiones anteriores (legacy)
│
├── scripts/                      # Scripts de experimentación
│   ├── run_experiments.sh        # Benchmarking completo (Bash)
│   └── run_experiments.py        # Benchmarking completo (Python/Windows)
│
├── analysis/                     # Scripts de análisis
│   ├── calculate_flops.py        # Calculadora de FLOPs
│   ├── fit_logp_model.py         # Ajuste del modelo LogP
│   ├── plot_results.py           # Gráficas (tema oscuro)
│   └── plot_results_white.py     # Gráficas (tema blanco)
│
├── docs/                         # Documentación
│   ├── reporte_proyecto.tex      # Reporte académico (LaTeX)
│   ├── presentacion.html         # Presentación HTML
│   ├── images/                   # Gráficas tema oscuro
│   └── images_report/            # Gráficas tema blanco
│
├── results_strong_scaling.csv    # Resultados de Strong Scaling
└── README.md                     # Este archivo
```

---

## 🚀 Instalación y Configuración

### Requisitos Previos

- **Python 3.8+**
- **MPI Implementation:**
  - Windows: [MS-MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
  - Linux/Mac: OpenMPI o MPICH

### Dependencias de Python

```bash
pip install mpi4py numpy scikit-learn matplotlib pandas
```

### Verificar Instalación de MPI

```bash
mpiexec --version
```

---

## 💻 Uso

### 1. Ejecución Secuencial (Baseline)

```bash
python src/knn_sequential.py
```

**Salida esperada:**
```
Accuracy (Sequential): 0.9833
Execution Time: 5.02 seconds
```

### 2. Ejecución de Versiones Paralelas

#### v1: Naive Point-to-Point
```bash
mpiexec -n 4 python src/v1_naive_p2p.py
```

#### v2: Collective Operations
```bash
mpiexec -n 4 python src/v2_collective_scatter.py
```

#### v3: Final Optimized
```bash
mpiexec -n 4 python src/v3_final_optimized.py
```

**Salida esperada (v3 con 4 procesos):**
```
=== KNN Parallel v3 (Final Optimized - Vectorized) ===
Processes: 4
Dataset: N=1437, M=360, d=64
Theoretical FLOPs: 99,532,800 (99.53 MFLOPs)

Results:
Accuracy: 0.9833

Timing Breakdown:
Total Time:        2.09 sec
I/O Time:          0.15 sec (7.2%)
Bcast Time:        0.08 sec (3.8%)
Scatter Time:      0.02 sec (1.0%)
Compute Time:      1.75 sec (83.7%)
Gather Time:       0.01 sec (0.5%)
Total Comm Time:   0.11 sec (5.3%)

Performance Metrics:
Serial Fraction (f): 0.0718
GFLOPs/sec:          0.0569
```

### 3. Ejecutar Experimentos Completos

#### Opción A: Script Python (Windows/Linux/Mac)
```bash
python scripts/run_experiments.py
```

#### Opción B: Script Bash (Linux/Mac)
```bash
bash scripts/run_experiments.sh
```

**Configuración**:
- Versiones: v1, v2, v3
- Procesos: p ∈ {1, 2, 4, 8}
- Runs: 5 ejecuciones por configuración
- Output: `results_strong_scaling.csv`

### 4. Análisis de Resultados

#### Calcular FLOPs teóricos:
```bash
python analysis/calculate_flops.py
```

#### Ajustar modelo LogP:
```bash
python analysis/fit_logp_model.py
```

#### Generar gráficas (tema oscuro para presentación):
```bash
python analysis/plot_results.py
```

#### Generar gráficas (tema blanco para reporte):
```bash
python analysis/plot_results_white.py
```

**Gráficas generadas:**
- `time_comparison.png` - Comparación de tiempos entre versiones
- `speedup_comparison.png` - Speedup vs ideal
- `efficiency_comparison.png` - Eficiencia del clúster
- `flops_performance.png` - Rendimiento computacional (GFLOPs/sec)
- `time_breakdown.png` - Breakdown de tiempos (v3)
- `amdahl_validation.png` - Validación de Ley de Amdahl
- `logp_fit_*.png` - Ajuste del modelo LogP por versión

---

## 📊 Resultados Principales

### Dataset Utilizado
- **Nombre:** Digits (scikit-learn)
- **Características:** Imágenes 8×8 píxeles (64 dimensiones)
- **Muestras totales:** 1797
- **Train/Test split:** 80/20 (1437 train, 360 test)
- **Clases:** 10 (dígitos 0-9)

### Métricas de Rendimiento

| Procesos | Tiempo (s) | Speedup | Eficiencia | FLOPs/s |
|----------|------------|---------|------------|---------|
| 1        | 5.02       | 1.00x   | 100%       | 19.8M   |
| 2        | 2.68       | 1.87x   | 94%        | 37.1M   |
| 4        | 2.09       | 2.40x   | 60%        | 47.6M   |
| 8        | 2.01       | 2.50x   | 31%        | 49.5M   |

### Hallazgos Clave

- ✅ **Precisión idéntica:** 0.9833 (secuencial vs paralelo)
- ✅ **Reducción de tiempo:** 60% (5.02s → 2.01s)
- ✅ **Configuración óptima:** 4 procesos (mejor balance speedup/eficiencia)
- ⚠️ **Saturación:** A partir de 8 procesos, el overhead de comunicación domina
- 📈 **Complejidad confirmada:** O(N²) verificada experimentalmente

---

## 🔬 Metodología

### Arquitectura Maestro-Trabajador

```
┌─────────┐
│  Main   │  (Rank 0: Scatter datos)
└────┬────┘
     │
  ┌──┴──┬──────┬──────┐
  │     │      │      │
┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐
│ T1│ │ T2│ │ T3│ │ T4│  (Workers: Cómputo local)
└─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
  │     │      │      │
  └──┬──┴──────┴──────┘
     │
┌────▼────┐
│   End   │  (Rank 0: Gather resultados)
└─────────┘
```

### Operaciones MPI Utilizadas

1. **`MPI_Scatter`**: Distribuir X_test entre procesos
2. **`MPI_Bcast`**: Enviar X_train, y_train a todos
3. **`MPI_Gather`**: Recolectar predicciones parciales

### Cálculo de FLOPs

Distancia euclidiana: `d(x,y) = √(Σ(xᵢ-yᵢ)²)`

- **FLOPs por distancia:** 3d (d restas + d multiplicaciones + d sumas)
- **FLOPs totales:** M × N × 3d = 360 × 1437 × 192 ≈ **99.5 MFLOPs**

---

## 📖 Documentación

### Reporte Académico

El reporte completo en LaTeX se encuentra en `docs/reporte_proyecto.tex` e incluye:

- ✅ Introducción y justificación
- ✅ Modelo PRAM y topología DAG
- ✅ Desarrollo incremental (3 versiones beta)
- ✅ Análisis de complejidad teórica normalizada
- ✅ Resultados experimentales completos
- ✅ Derivación de FLOPs desde distancia euclidiana
- ✅ Análisis de escalabilidad
- ✅ Conclusiones y mejoras propuestas
- ✅ Bibliografía con impacto descrito

### Compilar el Reporte

```bash
cd docs
pdflatex reporte_proyecto.tex
```

---

## 🎯 Mejoras Futuras

1. **Optimización de Comunicación:** Usar comunicación punto a punto para datasets grandes
2. **Estructuras de Datos Avanzadas:** Implementar KD-trees o Ball Trees
3. **Paralelización Híbrida:** Combinar MPI + OpenMP
4. **Balanceo Dinámico:** Distribución adaptativa de carga
5. **GPU Acceleration:** Porting a CUDA/OpenCL

---

## 📚 Referencias

1. **MPI Forum**. "MPI: A Message-Passing Interface Standard". Version 4.0, 2021.
2. **Gropp, W., Lusk, E., & Skjellum, A.** "Using MPI: Portable Parallel Programming with the Message-Passing Interface". MIT Press, 2014.
3. **mpi4py Documentation**. https://mpi4py.readthedocs.io/
4. **Scikit-learn**. "K-Nearest Neighbors". https://scikit-learn.org/stable/modules/neighbors.html
5. **Pacheco, P.** "An Introduction to Parallel Programming". Morgan Kaufmann, 2011.

---

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

---

## 🤝 Contribuciones

Este es un proyecto académico. Para preguntas o sugerencias, contactar a los autores.

---

## 📞 Contacto

**Repositorio:** https://github.com/neftalics/KNN-Paralela

---

**Última actualización:** Noviembre 2025
